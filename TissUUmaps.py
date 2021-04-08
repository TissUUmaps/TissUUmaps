#!/usr/bin/env python
#
# flaskTissUUmaps - a minimal python server for TissUUmaps using Flask
#
# This library is free software; you can redistribute it and/or modify it
# under the terms of version 3.0 of the GNU General Public License
# as published by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this library; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import encodings.idna

from collections import OrderedDict
from flask import Flask, abort, make_response, render_template, url_for,  request, Response, jsonify, send_from_directory
from pathlib import Path

from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QPlainTextEdit, QDialog, QSplashScreen
from PyQt5.QtWebChannel import QWebChannel
from PyQt5 import QtGui 
from PyQt5.QtGui import QDesktopServices


#from threading import Timer
import threading, time
import sys
import socket

import json
from io import BytesIO
import openslide
from openslide import ImageSlide, OpenSlide, OpenSlideError, open_slide
from openslide.deepzoom import DeepZoomGenerator
import os
from optparse import OptionParser
from threading import Lock
from functools import wraps
import imghdr
import urllib.parse
import urllib.request

import PIL
PIL.Image.MAX_IMAGE_PIXELS = 93312000000

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
        return f(*args, **kwargs) #Comment this line to add authentifaction
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

SLIDE_DIR = '.'
SLIDE_CACHE_SIZE = 10
DEEPZOOM_FORMAT = 'jpeg'
DEEPZOOM_TILE_SIZE = 254
DEEPZOOM_OVERLAP = 1
DEEPZOOM_LIMIT_BOUNDS = True
DEEPZOOM_TILE_QUALITY = 75

FOLDER_DEPTH = 4

# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    template_folder=os.path.join(sys._MEIPASS, 'templates')
    os.chdir(sys._MEIPASS)
elif __file__:
    template_folder="templates"
print ("template_folder",template_folder)
app = Flask(__name__,template_folder=template_folder)
app.config.from_object(__name__)
app.config.from_envvar('DEEPZOOM_MULTISERVER_SETTINGS', silent=True)

class PILBytesIO(BytesIO):
    def fileno(self):
        '''Classic PIL doesn't understand io.UnsupportedOperation.'''
        raise AttributeError('Not supported')


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
        try:
            osr = OpenSlide(path)
        except:
            osr = ImageSlide(path)
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
    def __init__(self, basedir, relpath='', max_depth=4):
        self.name = os.path.basename(relpath)
        self.children = []
        if max_depth != 0:
            try:
                for name in sorted(os.listdir(os.path.join(basedir, relpath))):
                    cur_relpath = os.path.join(relpath, name)
                    cur_path = os.path.join(basedir, cur_relpath)
                    if os.path.isdir(cur_path):
                        cur_dir = _Directory(basedir, cur_relpath, max_depth=max_depth-1)
                        if cur_dir.children:
                            self.children.append(cur_dir)
                    elif OpenSlide.detect_format(cur_path):
                        self.children.append(_SlideFile(cur_relpath))
                    elif imghdr.what(cur_path):
                        self.children.append(_SlideFile(cur_relpath))
                    
            except:
                pass


class _SlideFile(object):
    def __init__(self, relpath):
        self.name = os.path.basename(relpath)
        self.url_path = relpath.replace("\\","/")

def setup(app):
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
    return render_template('files.html', root_dir=_Directory(app.basedir, max_depth=app.config['FOLDER_DEPTH']), message="Impossible to load this file"), 404

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
    except OpenSlideError:
        abort(404)


@app.route('/TmapsState/<path:path>', methods=['GET', 'POST'])
@requires_auth
def setTmapsState(path):
    jsonFilename = os.path.abspath(os.path.join(app.basedir, path))
    jsonFilename = os.path.splitext(jsonFilename)[0]+'.tmap'
    print (request.method)
    
    if request.method == 'POST':
        state = request.get_json(silent=False)
        print (state["Markers"]["_nameAndLetters"])
        # we save the state in a tmap file
        with open(jsonFilename,"w") as jsonFile:
            json.dump(state, jsonFile)
    else:
        if os.path.isfile(jsonFilename):
            with open(jsonFilename,"r") as jsonFile:
                state = json.load(jsonFile)
        else:
            return jsonify({})
    return jsonify(state)

@app.route('/')
@requires_auth
def index():
    #return render_template('files.html', root_dir=_Directory(app.basedir, max_depth=app.config['FOLDER_DEPTH']))
    return render_template('files.html')

@app.route('/<path:path>')
@requires_auth
def slide(path):
    state_filename = "/TmapsState/" + path.replace("\\","\\\\") 

    slide = _get_slide(path)
    slide_url = url_for('dzi', path=path)
    slide_properties = slide.properties
    
    associated_urls = dict((name, url_for('dzi_asso', path=path, associated_name=name)) for name in slide.associated_images.keys())
    #folder_dir = _Directory(os.path.abspath(app.basedir)+"/",
    #                        os.path.dirname(path))
    #return render_template('tissuumaps.html', associated=associated_urls, slide_url=slide_url, state_filename=state_filename, slide_filename=slide.filename, slide_mpp=slide.mpp, properties=slide_properties, root_dir=_Directory(app.basedir, max_depth=app.config['FOLDER_DEPTH']), folder_dir=folder_dir)
    return render_template('tissuumaps.html', associated=associated_urls, slide_url=slide_url, state_filename=state_filename, slide_filename=slide.filename, slide_mpp=slide.mpp, properties=slide_properties)

@app.route('/<path:path>.csv')
@requires_auth
def csvFile(path):
    completePath = os.path.abspath(os.path.join(app.basedir, path) + ".csv")
    directory = os.path.dirname(completePath)
    filename = os.path.basename(completePath)
    if os.path.isfile(completePath):
        return send_from_directory(directory, filename)
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
@requires_auth
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
@requires_auth
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

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'misc/favicon.ico', mimetype='image/vnd.microsoft.icon')

class CustomWebEnginePage(QWebEnginePage):
    """ Custom WebEnginePage to customize how we handle link navigation """

    def acceptNavigationRequest(self, url,  _type, isMainFrame):
        if _type == QWebEnginePage.NavigationTypeLinkClicked:
            QDesktopServices.openUrl(url)
            return False
        return True

class textWindow(QDialog):
    def __init__(self, parent, title, message):
        QDialog.__init__(self, parent)

        self.setMinimumSize(QSize(700, 500))    
        self.setWindowTitle(title) 

        # Add text field
        self.b = QPlainTextEdit(self)
        self.b.setMinimumSize (650,450)
        self.b.setReadOnly(True)
        self.b.insertPlainText(message)
        self.b.move(10,10)
        self.b.resize(400,200)
    
class webEngine(QWebEngineView):
    def __init__(self, qt_app, app, args):
        super().__init__()
        self.app = app
        self.setMinimumSize(800,400)
        self.setContextMenuPolicy(Qt.NoContextMenu)
        self.lastdir = str(Path.home())
        self.setPage(CustomWebEnginePage(self))
        self.webchannel = QWebChannel()
        self.page().setWebChannel(self.webchannel)
        self.webchannel.registerObject('backend', self)
        self.location = None
        
        self.setWindowTitle("TissUUmaps")
        self.resize(1024, 800)
        self.setZoomFactor(1.0)
        self.page().profile().clearHttpCache()
        
        self.setWindowIcon(QtGui.QIcon('static/misc/favicon.ico')) 
        self.showMaximized()

    def run (self):
        sys.exit(qt_app.exec_())

    def setLocation (self, location):
        self.location = location
        while True:
            try:
                if (urllib.request.urlopen(self.location).getcode() == 200):
                    break
            except:
                pass
            time.sleep(0.1)
        print ("loading page ", self.location)
        if (len(args) > 0):
            if not self.openImagePath(args[0]):
                self.load(QUrl(self.location))
        else:
            self.load(QUrl(self.location))
            
    @pyqtSlot(str)
    def getProperties(self, path):
        try:
            path = urllib.parse.unquote(path)[:-4]
            print (path)
            slide = _get_slide(path)
            propString = "\n".join([n + ": " + v for n,v in slide.properties.items()])
        except:
            propString = ""
        
        messageBox = textWindow(self,os.path.basename(path) + " properties", propString)
        messageBox.show()
        
    @pyqtSlot()
    def openImage(self):
        home = str(Path.home())
        
        folderpath = QFileDialog.getOpenFileName(self, 'Select a File',self.lastdir)[0]
        self.openImagePath(folderpath)
    
    def openImagePath (self, folderpath):
        print (folderpath)
        try:
            oldBaseDir = app.basedir
        except AttributeError:
            oldBaseDir = ""
        self.lastdir = os.path.dirname(folderpath)
        if not folderpath:
            return
        parts = Path(folderpath).parts
        if (not hasattr(app, 'cache')):
            setup(app)
        app.basedir = parts[0]
        imgPath = os.path.join(*parts[1:])
        try:
            _get_slide(imgPath)
        except:
            app.basedir = oldBaseDir
            import traceback
            print (traceback.format_exc())
            QMessageBox.about(self, "Error", "TissUUmaps did not manage to open this image.")

            return False
        print (app.basedir, self.location + imgPath)
        self.load(QUrl(self.location + imgPath))
        self.setWindowTitle("TissUUmaps - " + os.path.basename(folderpath))
        return True

    @pyqtSlot()
    def exit(self):
        self.close()
        #sys.exit()

    @pyqtSlot(result="QJsonObject")
    def addLayer(self):
        folderpath = QFileDialog.getOpenFileName(self, 'Select a File')[0]
        if not folderpath:
            returnDict = {"dzi":None,"name":None}
            return returnDict
        parts = Path(folderpath).parts
        if (app.basedir != parts[0]):
            QMessageBox.about(self, "Error", "All layers must be in the same drive")
            returnDict = {"dzi":None,"name":None}
            return returnDict
        imgPath = os.path.join(*parts[1:])
        try:
            _get_slide(imgPath)
        except:
            QMessageBox.about(self, "Error", "TissUUmaps did not manage to open this image.")
            returnDict = {"dzi":None,"name":None}
            return returnDict
        returnDict = {
            "dzi":"/"+imgPath + ".dzi",
            "name":os.path.basename(imgPath)
        }
        print ("returnDict", returnDict)
        return returnDict
    
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
        
if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] [slide-directory]')
    parser.add_option('-B', '--ignore-bounds', dest='DEEPZOOM_LIMIT_BOUNDS',
                default=False, action='store_false',
                help='display entire scan area')
    parser.add_option('-c', '--config', metavar='FILE', dest='config',
                help='config file')
    parser.add_option('-d', '--debug', dest='DEBUG', action='store_true',
                help='run in debugging mode (insecure)')
    parser.add_option('-e', '--overlap', metavar='PIXELS',
                dest='DEEPZOOM_OVERLAP', type='int',
                help='overlap of adjacent tiles [1]')
    parser.add_option('-f', '--format', metavar='{jpeg|png}',
                dest='DEEPZOOM_FORMAT',
                help='image format for tiles [jpeg]')
    parser.add_option('-l', '--listen', metavar='ADDRESS', dest='host',
                default='127.0.0.1',
                help='address to listen on [127.0.0.1]')
    parser.add_option('-p', '--port', metavar='PORT', dest='port',
                type='int', default=5000,
                help='port to listen on [5000]')
    parser.add_option('-Q', '--quality', metavar='QUALITY',
                dest='DEEPZOOM_TILE_QUALITY', type='int',
                help='JPEG compression quality [75]')
    parser.add_option('-s', '--size', metavar='PIXELS',
                dest='DEEPZOOM_TILE_SIZE', type='int',
                help='tile size [254]')
    parser.add_option('-D', '--depth', metavar='LEVELS',
                dest='FOLDER_DEPTH', type='int',
                help='folder depth search for opening files [4]')

    (opts, args) = parser.parse_args()
    # Load config file if specified

    if opts.config is not None:
        app.config.from_pyfile(opts.config)
    # Overwrite only those settings specified on the command line
    for k in dir(opts):
        if not k.startswith('_') and getattr(opts, k) is None:
            delattr(opts, k)
    app.config.from_object(opts)
    # Set slide directory
    #try:
    #    app.config['SLIDE_DIR'] = args[0]
    #except IndexError:
    #    pass
    #Timer(0.01,lambda: ui("http://127.0.0.1:5000/", app)).start()
    qInstallMessageHandler(lambda x,y,z: None)

    qt_app = QApplication([])
    
    logo = QtGui.QPixmap('static/misc/design/logo.png')
    logo = logo.scaledToWidth(512, Qt.SmoothTransformation)
    splash = QSplashScreen(logo, Qt.WindowStaysOnTopHint)

    desktop = qt_app.desktop()
    scrn = desktop.screenNumber(QtGui.QCursor.pos())
    currentDesktopsCenter = desktop.availableGeometry(scrn).center()
    splash.move(currentDesktopsCenter - splash.rect().center())

    # can display startup information

    splash.show()

    #splash.showMessage('Loading TissUUmaps...',Qt.AlignBottom | Qt.AlignCenter,Qt.white)

    qt_app.processEvents()
    port = 5000
    print ("Starting port detection")
    while (is_port_in_use(port)):
        port += 1
        if port == 6000:
            exit(0)
    print ("Ending port detection", port)

    def flaskThread():
        app.run(host=opts.host, port=port, threaded=True, debug=False)
    
    threading.Thread(target=flaskThread,daemon=True).start()
    
    
    ui = webEngine(qt_app, app, args)
    ui.setLocation ("http://127.0.0.1:" + str(port) + "/")
    
    QTimer.singleShot(1000, splash.close)
    ui.run()
    #threading.Thread(target=flaskThread,daemon=True).start()
    #app.run(host="0.0.0.0", port=opts.port, threaded=False, debug=False)
