# Python default library
import base64
import gzip

# External libraries
import imghdr
import importlib
import io
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from functools import wraps
from threading import Lock
from urllib.parse import parse_qs, urlparse

import pyvips

# Flask dependencies
from flask import (
    Response,
    _request_ctx_stack,
    abort,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from tifffile import TiffFile

import tissuumaps.tiffslide as tiffslide
from tissuumaps import app
from tissuumaps.flask_filetree import filetree
from tissuumaps.tiffslide import OpenSlide
from tissuumaps.tiffslide.deepzoom import DeepZoomGenerator

# import openslide  # isort: skip
# from openslide import ImageSlide, OpenSlide  # isort: skip
# from openslide.deepzoom import DeepZoomGenerator  # isort: skip


def _fnfilter(filename):
    return True
    if OpenSlide.detect_format(filename):
        return True
    elif imghdr.what(filename):
        return True
    elif ".tmap" in filename.lower():
        return True
    return False


def _dfilter(filename):
    if "private" in filename:
        return False
    if ".tissuumaps" in filename:
        return False
    return True


ft = filetree.make_blueprint(
    app=app, register=False, dfilter=_dfilter, fnfilter=_fnfilter
)
app.register_blueprint(ft, url_prefix="/filetree")


def check_auth(username, password):
    if username == "username" and password == "password":
        return True
    return False


def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
        "Could not verify your access level for that URL.\n"
        "You have to login with proper credentials",
        401,
        {"WWW-Authenticate": 'Basic realm="Login Required"'},
    )


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if _request_ctx_stack.top.request.args.get("path"):
            path = os.path.abspath(
                os.path.join(
                    app.basedir, _request_ctx_stack.top.request.args.get("path"), "fake"
                )
            )
        elif not "path" in kwargs.keys():
            path = getPathFromReferrer(_request_ctx_stack.top.request, "")
        else:
            path = os.path.abspath(os.path.join(app.basedir, kwargs["path"]))
        activeFolder = os.path.dirname(path)
        while os.path.dirname(activeFolder) != activeFolder and not os.path.isfile(
            activeFolder + "/auth"
        ):
            activeFolder = os.path.dirname(activeFolder)
        if os.path.isfile(activeFolder + "/auth"):
            with open(activeFolder + "/auth", "r") as file:
                data = file.read().replace("\n", "")
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
        """Classic PIL doesn't understand io.UnsupportedOperation."""
        raise AttributeError("Not supported")


class ImageConverter:
    def __init__(self, inputImage, outputImage):
        self.inputImage = inputImage
        self.outputImage = outputImage

    def convert(self):
        logging.debug(" ".join(["Converting: ", self.inputImage, self.outputImage]))
        if not os.path.isfile(self.outputImage):

            def convertThread():
                try:
                    imgVips = pyvips.Image.new_from_file(self.inputImage)
                    minVal = imgVips.percent(0.5)
                    maxVal = imgVips.percent(99.5)
                    if minVal == maxVal:
                        minVal = 0
                        maxVal = 255
                    if minVal < 0 or maxVal > 255:
                        logging.debug(
                            f"Rescaling image {self.inputImage}: {minVal} - {maxVal} to 0 - 255"
                        )
                        imgVips = (255.0 * (imgVips - minVal)) / (maxVal - minVal)
                        imgVips = (imgVips < 0).ifthenelse(0, imgVips)
                        imgVips = (imgVips > 255).ifthenelse(255, imgVips)
                    imgVips = imgVips.scaleimage()
                    imgVips.tiffsave(
                        self.outputImage,
                        pyramid=True,
                        tile=True,
                        tile_width=256,
                        tile_height=256,
                        compression="jpeg",
                        Q=95,
                        properties=True,
                    )
                except:
                    logging.error("Impossible to convert image using VIPS:")
                    import traceback

                    logging.error(traceback.format_exc())
                self.convertDone = True

            self.convertDone = False
            threading.Thread(target=convertThread, daemon=True).start()
            while not self.convertDone:
                time.sleep(0.02)
        return self.outputImage

    def convertToDZI(self):
        if not os.path.isfile(self.outputImage):

            def convertThread():
                try:
                    imgVips = pyvips.Image.new_from_file(self.inputImage)
                    minVal = imgVips.percent(0.5)
                    maxVal = imgVips.percent(99.5)
                    if minVal == maxVal:
                        minVal = 0
                        maxVal = 255
                    if imgVips.percent(0.01) < 0 or imgVips.percent(99.99) > 255:
                        imgVips = (255.0 * (imgVips - minVal)) / (maxVal - minVal)
                        imgVips = (imgVips < 0).ifthenelse(0, imgVips)
                        imgVips = (imgVips > 255).ifthenelse(255, imgVips)
                        imgVips = imgVips.scaleimage()
                    imgVips.dzsave(
                        os.path.basename(self.outputImage),
                        dirname=os.path.dirname(self.outputImage),
                        suffix=".jpg",
                        background=0,
                        depth="onepixel",
                        overlap=0,
                        tile_size=256,
                    )

                except:
                    logging.error("Impossible to convert image using VIPS:")
                    import traceback

                    logging.error(traceback.format_exc())
                self.convertDone = True

            self.convertDone = False
            threading.Thread(target=convertThread, daemon=True).start()
            while not self.convertDone:
                time.sleep(0.02)
        return self.outputImage


class _SlideCache(object):
    def __init__(self, cache_size, dz_opts):
        self.cache_size = cache_size
        self.dz_opts = dz_opts
        self._lock = Lock()
        self._cache = OrderedDict()

    def get(self, path, originalPath=None):
        with self._lock:
            if path in self._cache:
                # Move to end of LRU
                slide = self._cache.pop(path)
                self._cache[path] = slide
                return slide

        osr = OpenSlide(path)
        # try:
        #    osr = OpenSlide(path)
        # except:
        #    osr = ImageSlide(path)
        # Fix for 16 bits tiff files
        # if osr._image.getextrema()[1] > 256:
        #     osr._image = osr._image.point(lambda i:i*(1./256)).convert('L')

        slide = DeepZoomGenerator(osr, **self.dz_opts)
        slide.osr = osr

        slide.associated_images = {}
        for name, image in slide.osr.associated_images.items():
            slide.associated_images[name] = image

        try:
            mpp_x = osr.properties[tiffslide.PROPERTY_NAME_MPP_X]
            mpp_y = osr.properties[tiffslide.PROPERTY_NAME_MPP_Y]
            slide.properties = osr.properties
            slide.mpp = float(mpp_x)
        except (KeyError, ValueError, TypeError):
            try:
                if osr.properties["tiff.ResolutionUnit"] == "centimetre":
                    numerator = 10000  # microns in CM
                else:
                    numerator = 25400  # Microns in Inch
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
        if originalPath:
            slide.properties = {"Path": originalPath}
        with self._lock:
            if path not in self._cache:
                while len(self._cache) >= self.cache_size:
                    self._cache.popitem(last=False)
                self._cache[path] = slide
        return slide


class _SlideFile(object):
    def __init__(self, relpath):
        self.name = os.path.basename(relpath)
        self.url_path = relpath.replace("\\", "/")


def setup(app):
    app.basedir = os.path.abspath(app.config["SLIDE_DIR"])
    config_map = {
        "DEEPZOOM_TILE_SIZE": "tile_size",
        "DEEPZOOM_OVERLAP": "overlap",
        "DEEPZOOM_LIMIT_BOUNDS": "limit_bounds",
    }
    opts = dict((v, app.config[k]) for k, v in config_map.items())
    app.cache = _SlideCache(app.config["SLIDE_CACHE_SIZE"], opts)


@app.before_first_request
def _setup():
    setup(app)


@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    # return render_template("tissuumaps.html", isStandalone=app.config["isStandalone"], message="Impossible to load this file", readOnly=app.config["READ_ONLY"])
    return redirect("/404"), 404, {"Refresh": "1; url=/404"}


def _get_slide(path, page=None, originalPath=None):
    path = os.path.abspath(os.path.join(app.basedir, path))
    if not path.startswith(app.basedir):
        # Directory traversal
        abort(404)
    if not os.path.exists(path):
        abort(404)
    try:
        slide = app.cache.get(path, originalPath)
        slide.filename = os.path.basename(path)
        return slide
    except:
        if ".tissuumaps" in path:
            abort(404)
        try:

            import traceback

            logging.error(traceback.format_exc())
            newpath = (
                os.path.dirname(path)
                + "/.tissuumaps/"
                + os.path.splitext(os.path.basename(path))[0]
                + ".tif"
            )
            os.makedirs(os.path.dirname(path) + "/.tissuumaps/", exist_ok=True)
            tifpath = ImageConverter(path, newpath).convert()
            return _get_slide(tifpath, path)
        except:
            import traceback

            logging.error(traceback.format_exc())
            abort(404)


@app.route("/")
@requires_auth
def index():
    return render_template(
        "tissuumaps.html",
        isStandalone=app.config["isStandalone"],
        readOnly=app.config["READ_ONLY"],
    )


@app.route("/web/<path:path>")
@requires_auth
def base_static(path):
    completePath = os.path.abspath(os.path.join(app.basedir, path))
    directory = os.path.dirname(completePath) + "/web/"
    filename = os.path.basename(completePath)
    return send_from_directory(directory, filename)


def int_to_rgb(v):
    rgba = [x / 255 * 100 for x in int(v).to_bytes(4, signed=True, byteorder="big")]
    return ",".join(str(int(x)) for x in rgba[:3])


@app.route("/<path:filename>")
@requires_auth
def slide(filename):
    path = request.args.get("path")
    if not path:
        path = "./"
    path = os.path.abspath(os.path.join(app.basedir, path, filename))
    try:
        slide = _get_slide(path)
        print (slide.properties.keys())
        tif = slide._osr.ts_tifffile
        # print (len(tif.pages), len(tif.series))
        if len(tif.series[0].pages) > 1:
            # ome = ome_types.from_xml(tif.pages[0].tags["ImageDescription"].value)
            # for annot in ome.structured_annotations:
            # pageNames = [f"Page {i}" for i in range(len(tif.pages))]
            # try:
            #    print (annot.dict())
            #    print (annot.value.OriginalMetadata.Key)
            #    if annot.Value.OriginalMetadata.Key == "Name":
            #        pageNames = json.loads(annot.Value.OriginalMetadata.Value)
            # except:
            #    import traceback

            #    logging.error(traceback.format_exc())
            #for p in tif.series[0].pages:
            #    print (p.tags)
            import xml.etree.ElementTree

            try:
                omexml_string = tif.series[0].pages[0].description.strip()
                try:
                    root = xml.etree.ElementTree.parse(
                        io.BytesIO(omexml_string.encode("utf-16"))
                    )
                except:
                    root = xml.etree.ElementTree.parse(
                        io.BytesIO(omexml_string.encode("utf-8"))
                    )
                namespaces = {
                    "ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"
                }
                channels = root.findall(
                    "ome:Image[1]/ome:Pixels/ome:Channel", namespaces
                )
                channel_names = [c.attrib["Name"] for c in channels]
                color_names = [int_to_rgb(c.attrib["Color"]) for c in channels]
            except:
                channel_names = [
                    f"Channel_{pindex}"
                    for pindex, page in enumerate(tif.series[0].pages)
                ]
                color_names = [
                    "100,100,100" for pindex, page in enumerate(tif.series[0].pages)
                ]
                import traceback

                logging.error(traceback.format_exc())
            slide_url = os.path.basename(path) + ".dzi"
            jsonProject = {
                "filters": ["Color", "Contrast"],
                "compositeMode": "lighter",
                "layerFilters": {
                    str(pindex): [
                        {
                            "name": "Color",
                            "value": color_names[pindex % len(color_names)],
                        }
                    ]
                    for pindex, page in enumerate(tif.series[0].pages)
                },
                "layers": [
                    {
                        "name": channel_names[pindex],
                        "tileSource": os.path.basename(path) + f"__p{pindex}.dzi",
                    }
                    for pindex, page in enumerate(tif.series[0].pages)
                ],
                "mpp": slide.properties["tiffslide.mpp-x"]
            }
        else:
            raise Exception("Only one page, so back to normal layers.")
    except:
        import traceback

        logging.error(traceback.format_exc())
        # slide = _get_slide(path)
        # slide = _get_slide(path)
        slide_url = os.path.basename(path) + ".dzi"  # url_for("dzi", path=path)
        jsonProject = {
            "layers": [{"name": os.path.basename(path), "tileSource": slide_url}]
        }
    return render_template(
        "tissuumaps.html",
        plugins=[p["module"] for p in app.config["PLUGINS"]],
        jsonProject=jsonProject,
        isStandalone=app.config["isStandalone"],
        readOnly=app.config["READ_ONLY"],
    )


@app.route("/ping")
@requires_auth
def ping():
    return make_response("pong")


def getPathFromReferrer(request, filename):
    try:
        parsed_url = urlparse(request.referrer)
        path = parse_qs(parsed_url.query)["path"][0]
        path = os.path.abspath(os.path.join(app.basedir, path, filename))
    except:
        path = os.path.abspath(os.path.join(app.basedir, filename))
    if not path:
        path = os.path.abspath(os.path.join(app.basedir, filename))
    logging.debug(f"Path from referrer: {path}")
    return path


@app.route("/<path:path>/<string:filename>.tmap", methods=["GET", "POST"])
@requires_auth
def tmapFile_old(path, filename):
    return redirect(url_for("tmapFile", filename=filename) + "?path=" + path)


@app.route("/<string:filename>.tmap", methods=["GET", "POST"])
@requires_auth
def tmapFile(filename):
    path = request.args.get("path")
    if not path:
        path = "./"
    jsonFilename = os.path.abspath(os.path.join(app.basedir, path, filename) + ".tmap")

    if request.method == "POST" and not app.config["READ_ONLY"]:
        state = request.get_json(silent=False)
        with open(jsonFilename, "w") as jsonFile:
            json.dump(state, jsonFile, indent=4, sort_keys=True)
        return state
    else:
        if os.path.isfile(jsonFilename):
            try:
                with open(jsonFilename, "r") as jsonFile:
                    state = json.load(jsonFile)
            except:
                import traceback

                logging.error(traceback.format_exc())
                abort(404)
        else:
            abort(404)
        if "plugins" in state.keys():
            plugins = []
        else:
            plugins = [p["module"] for p in app.config["PLUGINS"]]

        return render_template(
            "tissuumaps.html",
            plugins=plugins,
            jsonProject=state,
            isStandalone=app.config["isStandalone"],
            readOnly=app.config["READ_ONLY"],
        )


@app.route("/<path:completePath>.csv")
@requires_auth
def csvFile(completePath):
    completePath = os.path.join(app.basedir, completePath + ".csv")
    directory = os.path.dirname(completePath)
    filename = os.path.basename(completePath)
    if os.path.isfile(completePath):
        # We can not gzip csv files with the PapaParse library
        return send_from_directory(directory, filename)

        # We keep the old gzip code anyway:
        # gz files have to be names cgz for some browser
        if os.path.isfile(completePath + ".gz"):
            os.rename(completePath + ".gz", completePath + ".cgz")

        generate_cgz = False
        if not os.path.isfile(completePath + ".cgz"):
            generate_cgz = True
        elif os.path.getmtime(completePath) > os.path.getmtime(completePath + ".cgz"):
            # In this case, the csv file has been recently modified and the cgz file is
            # stale, so it must be regenerated.
            generate_cgz = True
        if generate_cgz:
            with open(completePath, "rb") as f_in, gzip.open(
                completePath + ".cgz", "wb", compresslevel=9
            ) as f_out:
                f_out.writelines(f_in)

        response = make_response(send_from_directory(directory, filename + ".cgz"))
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Vary"] = "Accept-Encoding"
        response.headers["Transfer-Encoding"] = "gzip"
        response.headers["Content-Length"] = os.path.getsize(completePath + ".cgz")
        response.headers["Content-Type"] = "text/csv; charset=UTF-8"
        return response
    else:
        abort(404)


@app.route("/<path:completePath>.json")
@requires_auth
def jsonFile(completePath):
    completePath = os.path.join(app.basedir, completePath + ".json")
    directory = os.path.dirname(completePath)
    filename = os.path.basename(completePath)
    if os.path.isfile(completePath):
        return send_from_directory(directory, filename)
    else:
        abort(404)


@app.route("/<path:path>.dzi", defaults={"page": None})
@app.route("/<path:path>__p<int:page>.dzi")
@requires_auth
def dzi(path, page):
    completePath = os.path.join(app.basedir, path)
    # Check if a .dzi file exists, else use OpenSlide:
    if os.path.isfile(completePath + ".dzi"):
        directory = os.path.dirname(completePath)
        filename = os.path.basename(completePath) + ".dzi"
        return send_from_directory(directory, filename)
    slide = _get_slide(path, page)
    format = app.config["DEEPZOOM_FORMAT"]
    resp = make_response(slide.get_dzi(format))
    resp.mimetype = "application/xml"
    return resp


@app.route("/<path:path>.dzi/info", defaults={"page": None})
@app.route("/<path:path>__p<int:page>.dzi/info")
@requires_auth
def dzi_asso(path, page):
    slide = _get_slide(path, page)
    associated_images = []
    for key, im in slide.associated_images.items():
        output = io.BytesIO()
        im.save(output, "PNG")
        b64 = base64.b64encode(output.getvalue()).decode()
        associated_images.append({"name": key, "content": b64})
        output.close()
    return render_template(
        "slide_prop.html",
        associated_images=associated_images,
        properties=slide.properties,
    )
    return resp


@app.route(
    "/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>",
    defaults={"page": None},
)
@app.route("/<path:path>__p<int:page>_files/<int:level>/<int:col>_<int:row>.<format>")
def tile(path, level, col, row, format, page):
    completePath = os.path.join(app.basedir, path)
    if os.path.isfile(f"{completePath}_files/{level}/{col}_{row}.{format}"):
        directory = f"{completePath}_files/{level}/"
        filename = f"{col}_{row}.{format}"
        return send_from_directory(directory, filename)
    slide = _get_slide(path, page)
    format = format.lower()
    # if format != 'jpeg' and format != 'png':
    #    # Not supported by Deep Zoom
    #    abort(404)
    try:
        with slide.tileLock:
            tile = slide.get_tile(level, (col, row), page)
    except ValueError:
        # Invalid level or coordinates
        logging.error("Invalid level or coordinates:")
        import traceback
        logging.error(traceback.format_exc())
        abort(404)
    buf = PILBytesIO()
    tile.save(buf, format, quality=app.config["DEEPZOOM_TILE_QUALITY"])
    resp = make_response(buf.getvalue())
    resp.mimetype = "image/%s" % format
    resp.cache_control.max_age = 1209600
    resp.cache_control.public = True
    return resp


@app.route(
    "/<path:path>.dzi/<path:associated_name>_files/<int:level>/<int:col>_<int:row>.<format>"
)
def tile_asso(path, associated_name, level, col, row, format):
    slide = _get_slide(path).associated_images[associated_name]
    format = format.lower()
    if format != "jpeg" and format != "png":
        # Not supported by Deep Zoom
        abort(404)
    try:
        tile = slide.get_tile(level, (col, row))
    except ValueError:
        # Invalid level or coordinates
        abort(404)
    buf = PILBytesIO()
    tile.save(buf, format, quality=app.config["DEEPZOOM_TILE_QUALITY"])
    resp = make_response(buf.getvalue())
    resp.mimetype = "image/%s" % format
    return resp


def load_plugin(name):
    for directory in [app.config["PLUGIN_FOLDER_USER"], app.config["PLUGIN_FOLDER"]]:
        if os.path.isfile(os.path.join(directory, name + ".py")):
            spec = importlib.util.spec_from_file_location(
                name, os.path.join(directory, name + ".py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
    return mod


@app.route("/plugins/<path:pluginName>.js")
def runPlugin(pluginName):
    for directory in [app.config["PLUGIN_FOLDER_USER"], app.config["PLUGIN_FOLDER"]]:
        filename = pluginName + ".js"
        completePath = os.path.abspath(os.path.join(directory, pluginName + ".js"))
        directory = os.path.dirname(completePath)
        filename = os.path.basename(completePath)
        if os.path.isfile(completePath):
            return send_from_directory(directory, filename)

    logging.error(completePath, "is not an existing file.")
    abort(404)


@app.route("/plugins/<path:pluginName>/<path:method>", methods=["GET", "POST"])
def pluginJS(pluginName, method):
    pluginModule = load_plugin(pluginName)
    pluginInstance = pluginModule.Plugin(app)
    pluginMethod = getattr(pluginInstance, method)
    if request.method == "POST":
        content = request.get_json(silent=False)
        return pluginMethod(content)
    else:
        content = request.args
        return pluginMethod(content)


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "misc/favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )
