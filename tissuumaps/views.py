import pyvips

from collections import OrderedDict
from flask import Flask, abort, make_response, render_template, url_for,  request, Response, jsonify, send_from_directory

import json
import io
import openslide
from openslide import ImageSlide, OpenSlide, OpenSlideError, OpenSlideUnsupportedFormatError, open_slide
from openslide.deepzoom import DeepZoomGenerator
import os
import threading, time
from threading import Lock
from functools import wraps
import imghdr
import importlib
import encodings.idna

from tissuumaps import app
import gzip

def check_auth(username, password):
    if username == "username" and password == "password":
        return True
    return False

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not "path" in kwargs.keys():
            return f(*args, **kwargs)
        path = os.path.abspath(os.path.join(app.basedir, kwargs["path"]))
        activeFolder = os.path.dirname(path)
        while (os.path.dirname(activeFolder) != activeFolder and not os.path.isfile(activeFolder + "/auth")):
            activeFolder = os.path.dirname(activeFolder)
        if os.path.isfile(activeFolder + "/auth"):
            with open(activeFolder + "/auth", 'r') as file:
                data = file.read().replace('\n', '')
                user, password = [u.strip() for u in data.split(";")]
            auth = request.authorization
            if not auth or not (user == auth.username and password == auth.password):
                return authenticate()
            return f(*args, **kwargs)
        else:
            return f(*args, **kwargs)
    return decorated

class PILBytesIO(io.BytesIO):
    def fileno(self):
        '''Classic PIL doesn't understand io.UnsupportedOperation.'''
        raise AttributeError('Not supported')

class ImageConverter():
    def __init__(self, inputImage, outputImage):
        self.inputImage = inputImage
        self.outputImage = outputImage
    
    def convert (self):
        print ("Converting:",self.inputImage, self.outputImage, os.path.isfile(self.outputImage))
        if not os.path.isfile(self.outputImage):
            def convertThread():
                try:
                    imgVips = pyvips.Image.new_from_file(self.inputImage)
                    minVal = imgVips.percent(0)
                    maxVal = imgVips.percent(99)
                    if minVal == maxVal:
                        minVal = 0
                        maxVal = 255
                    imgVips = (255.* (imgVips - minVal)) / (maxVal - minVal)
                    imgVips = (imgVips < 0).ifthenelse(0, imgVips)
                    imgVips = (imgVips > 255).ifthenelse(255, imgVips)
                    imgVips = imgVips.scaleimage()
                    imgVips.tiffsave(self.outputImage, pyramid=True, tile=True, tile_width=256, tile_height=256, properties=True, bitdepth=8)
                except: 
                    print ("Impossible to convert image using VIPS:")
                    import traceback
                    print (traceback.format_exc())
                self.convertDone = True
            self.convertDone = False
            threading.Thread(target=convertThread,daemon=True).start()
            while(not self.convertDone):
                time.sleep(0.02)
        return self.outputImage

class _SlideCache(object):
    def __init__(self, cache_size, dz_opts):
        self.cache_size = cache_size
        self.dz_opts = dz_opts
        self._lock = Lock()
        self._cache = OrderedDict()

    def get(self, path):
        with self._lock:
            if path in self._cache:
                # Move to end of LRU
                slide = self._cache.pop(path)
                self._cache[path] = slide
                return slide
        
        osr = OpenSlide(path)
        #try:
        #    osr = OpenSlide(path)
        #except:
        #    osr = ImageSlide(path)
            #Fix for 16 bits tiff files
            # if osr._image.getextrema()[1] > 256:
            #     osr._image = osr._image.point(lambda i:i*(1./256)).convert('L')
        
        slide = DeepZoomGenerator(osr, **self.dz_opts)
        slide.osr = osr
        
        slide.associated_images = {}
        for name, image in slide.osr.associated_images.items():
            slide.associated_images[name] = DeepZoomGenerator(ImageSlide(image))

        try:
            mpp_x = osr.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = osr.properties[openslide.PROPERTY_NAME_MPP_Y]
            slide.properties = osr.properties
            slide.mpp = (float(mpp_x) + float(mpp_y)) / 2
        except (KeyError, ValueError):
            try:
                if osr.properties["tiff.ResolutionUnit"]=="centimetre":
                    numerator = 10000 # microns in CM
                else:
                    numerator = 25400 # Microns in Inch
                mpp_x = numerator / float(osr.properties["tiff.XResolution"])
                mpp_y = numerator / float(osr.properties["tiff.YResolution"])
                slide.properties = osr.properties
                slide.mpp = (float(mpp_x) + float(mpp_y)) / 2
            except:
                slide.mpp = 0
        try:
            slide.properties = slide.properties
        except:
            slide.properties = osr.properties
        slide.tileLock = Lock()
        with self._lock:
            if path not in self._cache:
                while len(self._cache) >= self.cache_size:
                    self._cache.popitem(last=False)
                self._cache[path] = slide
        return slide


class _Directory(object):
    def __init__(self, basedir, relpath='', max_depth=4, filter=None):
        self.name = os.path.basename(relpath)
        self.children = []
        if max_depth != 0:
            try:
                for name in sorted(os.listdir(os.path.join(basedir, relpath))):
                    if ".tissuumaps" in name:
                        continue
                    if "private" in name:
                        continue
                    cur_relpath = os.path.join(relpath, name)
                    cur_path = os.path.join(basedir, cur_relpath)
                    if os.path.isdir(cur_path):
                        cur_dir = _Directory(basedir, cur_relpath, max_depth=max_depth-1, filter=filter)
                        if cur_dir.children:
                            self.children.append(cur_dir)
                    elif filter != None and filter not in name:
                        continue
                    elif OpenSlide.detect_format(cur_path):
                        self.children.append(_SlideFile(cur_relpath))
                    elif imghdr.what(cur_path):
                        self.children.append(_SlideFile(cur_relpath))
                    elif ".tmap" in cur_path:
                        self.children.append(_SlideFile(cur_relpath))
                    
            except:
                pass


class _SlideFile(object):
    def __init__(self, relpath):
        self.name = os.path.basename(relpath)
        self.url_path = relpath.replace("\\","/")

def setup(app):
    app.basedir = os.path.abspath(app.config['SLIDE_DIR'])
    config_map = {
        'DEEPZOOM_TILE_SIZE': 'tile_size',
        'DEEPZOOM_OVERLAP': 'overlap',
        'DEEPZOOM_LIMIT_BOUNDS': 'limit_bounds',
    }
    opts = dict((v, app.config[k]) for k, v in config_map.items())
    app.cache = _SlideCache(app.config['SLIDE_CACHE_SIZE'], opts)

@app.before_first_request
def _setup():
    setup(app)

@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    if app.config["isStandalone"]:
        return render_template('standalone/files.html', message="Impossible to load this file"), 404
    else:
        return render_template('server/files.html', root_dir=_Directory(app.basedir, max_depth=app.config['FOLDER_DEPTH']), message="Impossible to load this file"), 404
    

def _get_slide(path):
    path = os.path.abspath(os.path.join(app.basedir, path))
    if not path.startswith(app.basedir):
        # Directory traversal
        abort(404)
    if not os.path.exists(path):
        abort(404)
    try:
        slide = app.cache.get(path)
        slide.filename = os.path.basename(path)
        return slide
    except:
        if ".tissuumaps" in path:
            abort(404)
        try:
            newpath = os.path.dirname(path) + "/.tissuumaps/" + os.path.splitext(os.path.basename(path))[0] + ".tif"
            if not os.path.isdir(os.path.dirname(path) + "/.tissuumaps/"):
                os.makedirs(os.path.dirname(path) + "/.tissuumaps/")
            path = ImageConverter(path,newpath).convert()
            #imgPath = imgPath.replace("\\","/")
            return _get_slide(path)
        except:
            import traceback
            print (traceback.format_exc())
            abort(404)

@app.route('/')
@requires_auth
def index():
    #return render_template('files.html', root_dir=_Directory(app.basedir, max_depth=app.config['FOLDER_DEPTH']))
    if app.config["isStandalone"]:
        return render_template('standalone/files.html')
    else:
        return render_template('server/files.html', root_dir=_Directory(app.basedir, max_depth=app.config['FOLDER_DEPTH']))
        

@app.route('/web/<path:path>')
@requires_auth
def base_static(path):
    completePath = os.path.abspath(os.path.join(app.basedir, path))
    directory = os.path.dirname(completePath) + "/web/"
    filename = os.path.basename(completePath)
    return send_from_directory(directory, filename)

@app.route('/<path:path>')
@requires_auth
def slide(path):
    slide = _get_slide(path)
    slide_url = url_for('dzi', path=path)
    slide_properties = slide.properties
    
    associated_urls = dict((name, url_for('dzi_asso', path=path, associated_name=name)) for name in slide.associated_images.keys())
    #folder_dir = _Directory(os.path.abspath(app.basedir)+"/",
    #                        os.path.dirname(path))
    #
    if app.config["isStandalone"]:
        return render_template('standalone/tissuumaps.html', plugins=app.config["PLUGINS"], slide_url=slide_url, slide_filename=slide.filename, slide_mpp=slide.mpp, properties=slide_properties)
    else:
        folder_dir = _Directory(os.path.abspath(app.basedir)+"/",
                                os.path.dirname(path))
        if "private" in path:
            root_dir = _Directory(os.path.abspath(app.basedir)+"/", os.path.dirname(path), max_depth=app.config['FOLDER_DEPTH'])
        else:
            root_dir = _Directory(app.basedir, max_depth=app.config['FOLDER_DEPTH'])
        return render_template('server/tissuumaps.html', plugins=app.config["PLUGINS"], slide_url=slide_url, slide_filename=slide.filename, slide_mpp=slide.mpp, properties=slide_properties, root_dir=root_dir, folder_dir=folder_dir)

@app.route('/ping')
@requires_auth
def ping():
    return make_response("pong")

@app.route('/<path:path>.tmap', methods=['GET', 'POST'])
@requires_auth
def tmapFile(path):
    jsonFilename = os.path.abspath(os.path.join(app.basedir, path) + ".tmap")
    if request.method == 'POST':
        state = request.get_json(silent=False)
        with open(jsonFilename,"w") as jsonFile:
            json.dump(state, jsonFile, indent=4, sort_keys=True)
        return state
    else:
        if os.path.isfile(jsonFilename):
            try:
                with open(jsonFilename,"r") as jsonFile:
                    state = json.load(jsonFile)
            except:
                import traceback
                print (traceback.format_exc())
                abort(404)
        else:
            abort(404)
        if "plugins" in state.keys():
            plugins = state["plugins"]
        else:
            plugins = []
        
        if app.config["isStandalone"]:
            return render_template('standalone/tissuumaps.html', plugins=app.config["PLUGINS"], jsonProject=state)
        else:
            folder_dir = _Directory(os.path.abspath(app.basedir)+"/",
                                    os.path.dirname(path))
            if "private" in path:
                root_dir = _Directory(os.path.abspath(app.basedir)+"/", os.path.dirname(path), max_depth=app.config['FOLDER_DEPTH'], filter=".tmap")
            else:
                root_dir = _Directory(app.basedir, max_depth=app.config['FOLDER_DEPTH'])
            
            return render_template('server/tissuumaps.html', plugins=plugins, jsonProject=state, root_dir=root_dir, folder_dir=folder_dir)


@app.route('/<path:path>.csv')
@requires_auth
def csvFile(path):
    completePath = os.path.abspath(os.path.join(app.basedir, path) + ".csv")
    directory = os.path.dirname(completePath)
    filename = os.path.basename(completePath)
    if os.path.isfile(completePath):
        # Temporary fix for gz files without csv
        if os.path.isfile(completePath + ".gz"):
            os.rename(completePath + ".gz", completePath + ".cgz")

        if not os.path.isfile(completePath + ".cgz"):
            with open(completePath, 'rb') as f_in, gzip.open(completePath + ".cgz", 'wb', compresslevel=9) as f_out:
                f_out.writelines(f_in)
        response = make_response(send_from_directory(directory, filename + ".cgz"))
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Vary'] = 'Accept-Encoding'
        response.headers['Transfer-Encoding'] = 'gzip'
        response.headers['Content-Length'] = os.path.getsize(completePath + ".cgz")
        response.headers['Content-Type'] = 'text/csv; charset=UTF-8'
        return response
    else:
        abort(404)
    
@app.route('/<path:path>.json')
@requires_auth
def jsonFile(path):
    completePath = os.path.abspath(os.path.join(app.basedir, path) + ".json")
    directory = os.path.dirname(completePath)
    filename = os.path.basename(completePath)
    if os.path.isfile(completePath):
        return send_from_directory(directory, filename)
    else:
        abort(404)

@app.route('/<path:path>.dzi')
@requires_auth
def dzi(path):
    slide = _get_slide(path)
    format = app.config['DEEPZOOM_FORMAT']
    resp = make_response(slide.get_dzi(format))
    resp.mimetype = 'application/xml'
    return resp

@app.route('/<path:path>.dzi/<path:associated_name>')
@requires_auth
def dzi_asso(path,associated_name):
    slide = _get_slide(path)
    associated_image = slide.osr.associated_images[associated_name]
    dzg = DeepZoomGenerator(ImageSlide(associated_image))
    format = app.config['DEEPZOOM_FORMAT']
    resp = make_response(dzg.get_dzi(format))
    resp.mimetype = 'application/xml'
    return resp


@app.route('/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>')
def tile(path, level, col, row, format):
    slide = _get_slide(path)
    format = format.lower()
    #if format != 'jpeg' and format != 'png':
    #    # Not supported by Deep Zoom
    #    abort(404)
    try:
        with slide.tileLock:
            tile = slide.get_tile(level, (col, row))
    except ValueError:
        # Invalid level or coordinates
        abort(404)
    buf = PILBytesIO()
    tile.save(buf, format, quality=app.config['DEEPZOOM_TILE_QUALITY'])
    resp = make_response(buf.getvalue())
    resp.mimetype = 'image/%s' % format
    resp.cache_control.max_age = 1209600
    resp.cache_control.public = True
    return resp

@app.route('/<path:path>.dzi/<path:associated_name>_files/<int:level>/<int:col>_<int:row>.<format>')
def tile_asso(path, associated_name, level, col, row, format):
    slide = _get_slide(path).associated_images[associated_name]
    format = format.lower()
    if format != 'jpeg' and format != 'png':
        # Not supported by Deep Zoom
        abort(404)
    try:
        tile = slide.get_tile(level, (col, row))
    except ValueError:
        # Invalid level or coordinates
        abort(404)
    buf = PILBytesIO()
    tile.save(buf, format, quality=app.config['DEEPZOOM_TILE_QUALITY'])
    resp = make_response(buf.getvalue())
    resp.mimetype = 'image/%s' % format
    return resp

def load_plugin(name):
    mod = importlib.import_module("."+name,package="tissuumaps.plugins")
    return mod

@app.route('/plugin/<path:pluginName>.js')
def runPlugin(pluginName):
    directory = "plugins"
    filename = pluginName + ".js"
    completePath = os.path.abspath(os.path.join(directory, pluginName + ".js"))
    directory = os.path.dirname(completePath)
    filename = os.path.basename(completePath)
    if os.path.isfile(completePath):
        return send_from_directory(directory, filename)
    else:
        print (completePath, "is not an existing file.")
        abort(404)

@app.route('/plugin/<path:pluginName>/<path:method>', methods=['GET', 'POST'])
def pluginJS(pluginName, method):
    print ("runPlugin", pluginName, method)
    print (request.method)
    
    pluginModule = load_plugin(pluginName)
    pluginInstance = pluginModule.Plugin(app)
    pluginMethod = getattr(pluginInstance, method)
    if request.method == 'POST':
        content = request.get_json(silent=False)
        return pluginMethod(content)
    else:
        content = request.args
        return pluginMethod(content)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'misc/favicon.ico', mimetype='image/vnd.microsoft.icon')
