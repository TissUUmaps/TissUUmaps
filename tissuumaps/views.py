# Python default library
import base64
import gzip

# External libraries
import imghdr
import importlib
import io
import json
import logging
import mimetypes
import os
import pathlib
import re
import sys
import tempfile
import threading
import time
import traceback
from collections import OrderedDict
from functools import wraps
from shutil import copyfile, copytree
from threading import Lock
from urllib.parse import parse_qs, urlparse

# Hack to get brew installe libvips to work:
# ruff: noqa: E402
os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/lib"

import pyvips

# Flask dependencies
from flask import (
    Response,
    abort,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from packaging import version
from tissuumaps_schema import current as current_schema_module
from tissuumaps_schema.utils import (
    MAJOR_SCHEMA_VERSION_MODULES,
    get_major_version,
    guess_schema_version,
)
from werkzeug.exceptions import MethodNotAllowed, NotFound
from werkzeug.routing import RequestRedirect

from tissuumaps import app, read_h5ad, tarfile_stream

import openslide  # isort: skip
from openslide import OpenSlide  # isort: skip
from openslide.deepzoom import DeepZoomGenerator  # isort: skip


def _fnfilter(filename):
    if OpenSlide.detect_format(filename):
        return True
    elif imghdr.what(filename):
        return True
    elif filename.lower().endswith(".tmap"):
        return True
    elif filename.lower().endswith(".dzi"):
        return True
    elif filename.lower().endswith(".h5ad"):
        return True
    return False


def _dfilter(filename):
    if "private" in filename:
        return False
    if ".tissuumaps" in filename:
        return False
    return True


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
        if request.args.get("path"):
            path = os.path.abspath(
                os.path.join(app.basedir, request.args.get("path"), "fake")
            )
        elif "path" not in kwargs.keys():
            path = getPathFromReferrer(request, "")
        else:
            path = os.path.abspath(os.path.join(app.basedir, kwargs["path"]))
        activeFolder = path
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
        if not os.path.isfile(self.outputImage) or (
            os.path.getmtime(self.inputImage) > os.path.getmtime(self.outputImage)
        ):

            def convertThread():
                try:
                    imgVips = pyvips.Image.new_from_file(self.inputImage)
                    min_percent = app.config["VIPS_MIN_OUTLIER_PERC"]
                    max_percent = app.config["VIPS_MAX_OUTLIER_PERC"]

                    minVal = imgVips.percent(min_percent)
                    maxVal = imgVips.percent(max_percent)

                    if app.config["VIPS_EXCLUDE_MIN_INTENSITY"]:
                        absoluteMinVal = imgVips.min()
                        imgVips_tmp = (
                            (imgVips == absoluteMinVal)
                            .bandand()
                            .ifthenelse(maxVal + 1, imgVips)
                        )
                        minVal = imgVips_tmp.percent(min_percent)

                    if minVal == maxVal:
                        minVal = 0
                        maxVal = 255
                    if (
                        app.config["VIPS_FORCE_RESCALE"]
                        or imgVips.min() < 0
                        or imgVips.max() > 255
                    ):
                        logging.debug(
                            f"Rescaling image {self.inputImage}: "
                            f"{minVal} - {maxVal} to 0 - 255"
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
                        Q=app.config["VIPS_JPEG_COMPRESSION"],
                        properties=True,
                    )
                except Exception:
                    logging.error("Impossible to convert image using VIPS:")
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
                        self.outputImage,
                        suffix=".jpg",
                        background=0,
                        depth="onepixel",
                        overlap=0,
                        tile_size=256,
                    )

                except Exception:
                    logging.error("Impossible to convert image using VIPS:")
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
        # except Exception as e:
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
            mpp_x = osr.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = osr.properties[openslide.PROPERTY_NAME_MPP_Y]
            slide.properties = osr.properties
            slide.mpp = float(mpp_x)
        except (KeyError, ValueError):
            try:
                if osr.properties["tiff.ResolutionUnit"] == "centimetre":
                    numerator = 10000  # microns in CM
                else:
                    numerator = 25400  # Microns in Inch
                mpp_x = numerator / float(osr.properties["tiff.XResolution"])
                mpp_y = numerator / float(osr.properties["tiff.YResolution"])
                slide.properties = osr.properties
                slide.mpp = (float(mpp_x) + float(mpp_y)) / 2
            except Exception:
                slide.mpp = 0
        try:
            slide.properties = slide.properties
        except Exception:
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


@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return (
        render_template(
            "tissuumaps.html",
            isStandalone=app.config["isStandalone"],
            message="File not found or corrupted.",
            readOnly=app.config["READ_ONLY"],
            version=app.config["VERSION"],
            schema_version=current_schema_module.VERSION,
        ),
        404,
    )


@app.errorhandler(500)
def internal_server_error(e):
    # note that we set the 500 status explicitly
    return (
        render_template(
            "tissuumaps.html",
            isStandalone=app.config["isStandalone"],
            message=(
                "Internal Server Error<br/>The server encountered an internal "
                "error and was unable to complete your request. Either the "
                "server is overloaded or there is an error in the application."
            ),
            readOnly=app.config["READ_ONLY"],
            version=app.config["VERSION"],
            schema_version=current_schema_module.VERSION,
        ),
        500,
    )
    return redirect("/"), 500, {"Refresh": "1; url=/"}


def _get_slide(path, originalPath=None):
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
    except Exception:
        if ".tissuumaps" in path:
            abort(404)
        try:
            newpath = (
                os.path.dirname(path)
                + "/.tissuumaps/"
                + os.path.splitext(os.path.basename(path))[0]
                + ".tif"
            )
            os.makedirs(os.path.dirname(path) + "/.tissuumaps/", exist_ok=True)
            tifpath = ImageConverter(path, newpath).convert()
            return _get_slide(tifpath, path)
        except Exception:
            logging.error(traceback.format_exc())
            abort(404)


def get_view_function(url, method="GET"):
    """Match a url and return the view and arguments
    it will be called with, or None if there is no view.
    """

    adapter = app.url_map.bind("localhost")

    try:
        match = adapter.match(url, method=method)
    except RequestRedirect as e:
        # recursively match redirects
        return get_view_function(e.new_url, method)
    except (MethodNotAllowed, NotFound):
        # no match
        return None

    try:
        # return the view function and arguments
        return app.view_functions[match[0]], match[1]
    except KeyError:
        # no view is associated with the endpoint
        return None


@app.route("/")
@requires_auth
def index():
    if app.config["DEFAULT_PROJECT"]:
        # Check if the default project exists and is accessible
        default_project = os.path.join(app.basedir, app.config["DEFAULT_PROJECT"])
        if os.path.isfile(default_project):
            view_function = get_view_function(
                "/" + app.config["DEFAULT_PROJECT"], method="GET"
            )
            if view_function:
                return view_function[0](**view_function[1])

        elif app.config["PROJECT_LIST"]:
            projectList = [
                {"name": "Select a dataset", "path": "", "selected": True}
            ] + getProjectList(os.path.dirname(default_project))
            if len(projectList) > 0:
                projectList[0]["selected"] = True
                return render_template(
                    "tissuumaps.html",
                    plugins=[],
                    jsonProject={},
                    isStandalone=app.config["isStandalone"],
                    readOnly=app.config["READ_ONLY"],
                    projectList=projectList,
                    version=app.config["VERSION"],
                    schema_version=current_schema_module.VERSION,
                )

    indexPath = os.path.abspath(os.path.join(app.basedir, "index.html"))
    if os.path.isfile(indexPath) and app.config["READ_ONLY"]:
        directory = os.path.dirname(indexPath)
        filename = os.path.basename(indexPath)
        return send_from_directory(directory, filename)
    return render_template(
        "tissuumaps.html",
        plugins=[p["module"] for p in app.config["PLUGINS"]],
        isStandalone=app.config["isStandalone"],
        readOnly=app.config["READ_ONLY"],
        version=app.config["VERSION"],
        schema_version=current_schema_module.VERSION,
    )


@app.route("/web/<path:path>")
@requires_auth
def base_static(path):
    completePath = os.path.abspath(os.path.join(app.basedir, path))
    directory = os.path.dirname(completePath) + "/web/"
    filename = os.path.basename(completePath)
    return send_from_directory(directory, filename)


@app.route("/<path:path>.html")
@requires_auth
def base_static_redirect(path):
    return redirect("/web/" + path + ".html")


@app.route("/<path:filename>")
@requires_auth
def slide(filename):
    path = request.args.get("path")
    if not path:
        path = "./"
    path = os.path.abspath(os.path.join(app.basedir, path, filename))
    if not os.path.isfile(path) and not os.path.isfile(path + ".dzi"):
        abort(404)
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
        version=app.config["VERSION"],
        schema_version=current_schema_module.VERSION,
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
    except Exception:
        path = os.path.abspath(os.path.join(app.basedir, filename))
    if not path:
        path = os.path.abspath(os.path.join(app.basedir, filename))
    logging.debug(f"Path from referrer: {path}")
    return path


@app.route("/<path:path>/<string:filename>.tmap", methods=["GET", "POST"])
@requires_auth
def tmapFile_old(path, filename):
    if path == "":
        path = "./"
    return redirect(url_for("tmapFile", filename=filename) + "?path=" + path)


@app.route(
    "/<path:path>/<string:filename>.<any(h5ad, adata, h5):ext>", methods=["GET", "POST"]
)
@requires_auth
def h5adFile_old(path, filename, ext):
    if path == "":
        path = "./"
    return redirect(url_for("h5ad", filename=filename, ext=ext) + "?path=" + path)


def getProjectList(path):
    # Add project list if app.config["PROJECT_LIST"] is True
    projectList = []
    # Browse the directory of json_filename, with only one level
    for item in sorted(os.listdir(path)):
        file = os.path.join(path, item)
        if os.path.isfile(file) and file.endswith(".tmap"):
            # Add file relative to the basedir
            file = os.path.relpath(file, app.basedir)
            project = {"name": file[:-5], "path": file, "selected": False}
            projectList.append(project)
    return projectList


@app.route("/<string:filename>.tmap", methods=["GET", "POST"])
@requires_auth
def tmapFile(filename):
    if request.args.get("dl", default=0) != 0:
        return dlTmapFile(filename)
    # Get the path from the request arguments or use the current directory
    path = request.args.get("path", default="./")

    # Create the absolute path to the JSON file
    json_filename = os.path.abspath(os.path.join(app.basedir, path, filename) + ".tmap")

    # Define error message feedback to user, None means no error message
    errorMessage = None
    warningMessage = None
    if request.method == "POST" and not app.config["READ_ONLY"]:
        # Handle POST request to update the JSON file
        state = request.get_json(silent=False)
        with open(json_filename, "w") as json_file:
            json.dump(state, json_file, indent=4, sort_keys=True)
        return state
    else:
        # Handle GET request to read the JSON file
        if os.path.isfile(json_filename):
            try:
                with open(json_filename, "r") as json_file:
                    state = json.load(json_file)

                schema_version = guess_schema_version(state)
                major_schema_version = get_major_version(schema_version)
                # If major version is newer
                if version.parse(major_schema_version) > version.parse(
                    current_schema_module.VERSION
                ):
                    warningMessage = (
                        "<b>Warning:</b> This project was created with a newer version "
                        "of TissUUmaps with breaking changes.<br/><br/>Please "
                        "upgrade your TissUUmaps version to ensure compatibility."
                    )
                    logging.error(
                        "The MAJOR schema version of the project file "
                        f"({schema_version}) is newer than the MAJOR schema version "
                        "supported by the current  TissUUmaps installation "
                        f"({current_schema_module.VERSION})"
                    )
                # Else if minor version is newer
                elif version.parse(schema_version) > version.parse(
                    current_schema_module.VERSION
                ):
                    warningMessage = (
                        "<b>Warning:</b> This project was created with a newer version "
                        "of TissUUmaps.<br/><br/>"
                        "Upgrade your TissUUmaps version to get all functionalities."
                    )
                    logging.error(
                        "The MINOR schema version of the project file "
                        f"({schema_version}) is newer than the MINOR schema version "
                        "supported by the current TissUUmaps installation "
                        f"({current_schema_module.VERSION})"
                    )
                # Else if major version is unknown
                elif major_schema_version not in MAJOR_SCHEMA_VERSION_MODULES:
                    warningMessage = (
                        "<b>Warning:</b> This project was created with an unknown "
                        "version of TissUUmaps.<br/><br/>"
                        "Upgrade your TissUUmaps version to get all functionalities."
                    )
                    logging.error(
                        f"Unsupported MAJOR version in project file: {schema_version}"
                    )
                # Else validate and upgrade the project to last version
                else:
                    old_schema_module = MAJOR_SCHEMA_VERSION_MODULES[
                        major_schema_version
                    ]

                    try:
                        old_project = old_schema_module.Project.model_validate(state)
                        project = current_schema_module.Project.upgrade(old_project)
                        state = project.model_dump(by_alias=True)
                    except Exception as e:
                        logging.error(traceback.format_exc())
                        trace = (
                            "<br>".join(traceback.format_exception_only(e))
                            .replace("\n", "<br>")
                            .replace("\r", "<br>")
                            .replace('"', '\\"')
                        )
                        warningMessage = (
                            "<b>Warning when loading tmap project:</b> <br><pre><code>"
                            + trace
                            + "</code></pre>"
                        )
            # Error when parsing the JSON file:
            except Exception as e:
                logging.error(traceback.format_exc())
                trace = (
                    "<br>".join(traceback.format_exception_only(e))
                    .replace("\n", "<br>")
                    .replace("\r", "<br>")
                    .replace('"', '\\"')
                )
                errorMessage = (
                    "<b>Error when loading tmap project:</b> <br><pre><code>"
                    + trace
                    + "</code></pre>"
                )
                state = {}
        else:
            abort(404)

        # Determine the plugins based on the state
        if "plugins" in state.keys():
            plugins = []
        else:
            plugins = [p["module"] for p in app.config["PLUGINS"]]

        # Get the list of projects in the current directory
        projectList = (
            getProjectList(os.path.dirname(json_filename))
            if app.config["PROJECT_LIST"]
            else []
        )
        file = os.path.relpath(json_filename, app.basedir)
        for p in projectList:
            if p["path"] == file:
                p["selected"] = True

        # Render the template with appropriate data
        return render_template(
            "tissuumaps.html",
            plugins=plugins,
            jsonProject=state,
            isStandalone=app.config["isStandalone"],
            readOnly=app.config["READ_ONLY"],
            projectList=projectList,
            version=app.config["VERSION"],
            schema_version=current_schema_module.VERSION,
            message=errorMessage,
            warning=warningMessage,
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


@app.route("/<path:completePath>.<any(json, geojson, pbf):ext>")
@requires_auth
def jsonFile(completePath, ext):
    completePath = os.path.join(app.basedir, completePath + "." + ext)
    directory = os.path.dirname(completePath)
    filename = os.path.basename(completePath)
    if os.path.isfile(completePath):
        return send_from_directory(directory, filename)
    else:
        abort(404)


@app.route("/<path:path>.dzi")
@requires_auth
def dzi(path):
    completePath = os.path.join(app.basedir, path)
    # Check if a .dzi file exists, else use OpenSlide:
    if os.path.isfile(completePath + ".dzi"):
        directory = os.path.dirname(completePath)
        filename = os.path.basename(completePath) + ".dzi"
        return send_from_directory(directory, filename)
    slide = _get_slide(path)
    format = app.config["DEEPZOOM_FORMAT"]
    resp = make_response(slide.get_dzi(format))
    resp.mimetype = "application/xml"
    return resp


@app.route("/<path:path>.dzi/info")
@requires_auth
def dzi_asso(path):
    slide = _get_slide(path)
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


@app.route("/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>")
def tile(path, level, col, row, format):
    completePath = os.path.join(app.basedir, path)
    if os.path.isfile(f"{completePath}_files/{level}/{col}_{row}.{format}"):
        directory = f"{completePath}_files/{level}/"
        filename = f"{col}_{row}.{format}"
        return send_from_directory(directory, filename)
    slide = _get_slide(path)
    format = format.lower()
    # if format != 'jpeg' and format != 'png':
    #    # Not supported by Deep Zoom
    #    abort(404)
    try:
        with slide.tileLock:
            tile = slide.get_tile(level, (col, row))
    except ValueError:
        # Invalid level or coordinates
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


def send_file_partial(path):
    """
    Simple wrapper around send_file which handles HTTP 206 Partial Content
    (byte ranges)
    TODO: handle all send_file args, mirror send_file's error handling
    (if it has any)
    """
    range_header = request.headers.get("Range", None)
    if not range_header:
        response = make_response(send_file(path))
        response.headers["Accept-Ranges"] = "bytes"
        return response

    size = os.path.getsize(path)
    byte1, byte2 = 0, None
    m = re.search(r"(\d+)-(\d*)", range_header)
    g = m.groups()

    if g[0]:
        byte1 = int(g[0])
    if g[1]:
        byte2 = int(g[1])

    length = size - byte1
    if byte2 is not None:
        length = byte2 - byte1

    data = None
    with open(path, "rb") as f:
        f.seek(byte1)
        data = f.read(length + 1)

    rv = Response(
        data, 206, mimetype=mimetypes.guess_type(path)[0], direct_passthrough=True
    )
    rv.headers["Accept-Ranges"] = "bytes"
    rv.headers.add(
        "Content-Range", "bytes {0}-{1}/{2}".format(byte1, byte1 + length, size)
    )

    logging.debug(
        " ".join(
            [
                "Sent!",
                str(range_header),
                str(size),
                "bytes {0}-{1}/{2}".format(byte1, byte1 + length, size),
            ]
        )
    )
    return rv


@app.route("/<string:filename>.<any(h5ad, adata, h5):ext>")
@requires_auth
def h5ad(filename, ext):
    path = request.args.get("path")
    if not path:
        path = "./"
    completePath = os.path.abspath(
        os.path.join(app.basedir, path, filename) + "." + ext
    )
    # Check if a .h5ad file exists:
    if not os.path.isfile(completePath):
        abort(404)
    if "Referer" in request.headers.keys():
        if "h5Utils_worker.js" in request.headers["Referer"]:
            return send_file_partial(completePath)

    state = read_h5ad.h5ad_to_tmap(
        app.basedir, os.path.join(path, filename) + "." + ext
    )

    plugins = [p["module"] for p in app.config["PLUGINS"]]
    return render_template(
        "tissuumaps.html",
        plugins=plugins,
        jsonProject=state,
        isStandalone=app.config["isStandalone"],
        readOnly=app.config["READ_ONLY"],
        version=app.config["VERSION"],
        schema_version=current_schema_module.VERSION,
    )


def exportToStatic(state, folderpath, previouspath):
    imgFiles = []
    otherFiles = []

    def addRelativePath(state, relativePath):
        nonlocal imgFiles, otherFiles

        def addRelativePath_aux(state, path, isImg):
            nonlocal imgFiles, otherFiles
            if path[0] not in state.keys():
                return
            if len(path) == 1:
                if path[0] not in state.keys():
                    return
                if state[path[0]] is None:
                    return
                if isinstance(state[path[0]], list):
                    if isImg:
                        imgFiles += [s for s in state[path[0]]]
                        state[path[0]] = [
                            "data/images/"
                            + os.path.basename(s.replace("/", "_").replace("\\", "_"))
                            for s in state[path[0]]
                        ]
                    else:
                        otherFiles += [relativePath + "/" + s for s in state[path[0]]]
                        state[path[0]] = [
                            "data/files/" + os.path.basename(s) for s in state[path[0]]
                        ]

                else:
                    if isImg:
                        imgFiles += [state[path[0]]]
                        state[path[0]] = "data/images/" + os.path.basename(
                            state[path[0]].replace("/", "_").replace("\\", "_")
                        )
                    else:
                        otherFiles += [state[path[0]]]
                        state[path[0]] = "data/files/" + os.path.basename(
                            state[path[0]]
                        )
                return
            else:
                if path[0] not in state.keys():
                    return
                if isinstance(state[path[0]], list):
                    for state_ in state[path[0]]:
                        addRelativePath_aux(state_, path[1:], isImg)
                else:
                    addRelativePath_aux(state[path[0]], path[1:], isImg)

        try:
            relativePath = relativePath.replace("\\", "/")
            paths = [
                ["layers", "tileSource"],
                ["markerFiles", "path"],
                ["regionFiles", "path"],
                ["regionFile"],
            ]
            for path in paths:
                addRelativePath_aux(state, path, path[0] == "layers")
        except Exception:
            logging.error(traceback.format_exc())

        return state

    if not folderpath:
        return {"success": False, "error": "Directory not found"}
    try:
        relativePath = os.path.relpath(previouspath, os.path.dirname(folderpath))
        state = addRelativePath(json.loads(state), relativePath)

        os.makedirs(os.path.join(folderpath, "data/images"), exist_ok=True)
        os.makedirs(os.path.join(folderpath, "data/files"), exist_ok=True)
        for image in imgFiles:
            image = image.replace(".dzi", "")
            ImageConverter(
                os.path.join(previouspath, image),
                os.path.join(
                    folderpath,
                    "data/images",
                    os.path.basename(image.replace("/", "_").replace("\\", "_")),
                ),
            ).convertToDZI()
        for file in set(otherFiles):
            copyfile(
                os.path.join(previouspath, file),
                os.path.join(folderpath, "data/files", os.path.basename(file)),
            )

        if getattr(sys, "frozen", False):
            mainFolderPath = sys._MEIPASS
        else:
            mainFolderPath = os.path.dirname(pathlib.Path(__file__))

        with app.app_context():
            index = render_template(
                "tissuumaps.html",
                plugins=[],
                jsonProject=state,
                isStandalone=False,
                readOnly=True,
                version=app.config["VERSION"],
                schema_version=current_schema_module.VERSION,
            )
        # Replace /static with static:
        index = index.replace('"/static/', '"static/')

        with open(folderpath + "/index.html", "w") as f:
            f.write(index)

        for dir in ["css", "js", "misc", "vendor"]:
            copytree(
                os.path.join(mainFolderPath, "static", dir),
                os.path.join(folderpath, "static", dir),
                dirs_exist_ok=True,
            )

        return {"success": True}
    except Exception:
        return {"success": False, "error": traceback.format_exc()}


def dlTmapFile(filename):
    # Get the path from the request arguments or use the current directory
    path = request.args.get("path", default="./")
    previous_path = os.path.join(app.basedir, path)
    # Create the absolute path to the JSON file
    json_filename = os.path.abspath(os.path.join(app.basedir, path, filename) + ".tmap")
    with open(json_filename, "r") as json_file:
        state = json.load(json_file)

    imgFiles = []
    otherFiles = []

    def addRelativePath(state, relativePath):
        nonlocal imgFiles, otherFiles

        def addRelativePath_aux(state, path, isImg):
            nonlocal imgFiles, otherFiles
            if path[0] not in state.keys():
                return
            if len(path) == 1:
                if path[0] not in state.keys():
                    return
                if state[path[0]] is None:
                    return
                if isinstance(state[path[0]], list):
                    if isImg:
                        imgFiles += [s for s in state[path[0]]]
                        state[path[0]] = [
                            "data/images/"
                            + os.path.basename(s.replace("/", "_").replace("\\", "_"))
                            for s in state[path[0]]
                        ]
                    else:
                        otherFiles += [relativePath + "/" + s for s in state[path[0]]]
                        state[path[0]] = [
                            "data/files/" + os.path.basename(s) for s in state[path[0]]
                        ]

                else:
                    if isImg:
                        imgFiles += [state[path[0]]]
                        state[path[0]] = "data/images/" + os.path.basename(
                            state[path[0]].replace("/", "_").replace("\\", "_")
                        )
                    else:
                        otherFiles += [state[path[0]]]
                        state[path[0]] = "data/files/" + os.path.basename(
                            state[path[0]]
                        )
                return
            else:
                if path[0] not in state.keys():
                    return
                if isinstance(state[path[0]], list):
                    for state_ in state[path[0]]:
                        addRelativePath_aux(state_, path[1:], isImg)
                else:
                    addRelativePath_aux(state[path[0]], path[1:], isImg)

        try:
            relativePath = relativePath.replace("\\", "/")
            paths = [
                ["layers", "tileSource"],
                ["markerFiles", "path"],
                ["regionFiles", "path"],
                ["regionFile"],
            ]
            for path in paths:
                addRelativePath_aux(state, path, path[0] == "layers")
        except Exception:
            logging.error(traceback.format_exc())

        return state

    state = addRelativePath(state, "./")

    # Create a temp file called "TissUUmaps_project.tmap" using tempfile
    tmp_tmap = os.path.join(tempfile.gettempdir(), "TissUUmaps_project.tmap")
    with open(tmp_tmap, "w") as f:
        json.dump(state, f)

    def stream_file():
        tar = tarfile_stream.open(mode="w|tar")
        yield from tar.header()
        for image in imgFiles:
            image = image.replace(".dzi", "")
            yield from tar.add(
                os.path.join(previous_path, image),
                os.path.join(
                    "data/images",
                    os.path.basename(image.replace("/", "_").replace("\\", "_")),
                ),
            )

        for file in set(otherFiles):
            yield from tar.add(
                os.path.join(previous_path, file),
                os.path.join("data/files", os.path.basename(file)),
            )
        yield from tar.add(tmp_tmap, os.path.basename(json_filename))
        yield from tar.footer()

    return app.response_class(
        stream_file(),
        headers={
            "Content-Disposition": "attachment; filename=TissUUmaps_project.tar",
            "filename": "TissUUmaps_project.tar",
        },
    )


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

    logging.error(completePath + " is not an existing file.")
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


@app.route("/filetree")
@requires_auth
def get_tree():
    if not app.config["READ_ONLY"]:
        return render_template("filetree.html")
    else:
        return make_response("")


@app.route("/get_file_tree")
@requires_auth
def get_file_tree():
    if not app.config["READ_ONLY"]:
        root_path = app.config["SLIDE_DIR"] + "/" + request.args.get("root", "./")
        return jsonify(get_file_tree_data(root_path))
    else:
        return jsonify([])


def get_file_tree_data(root_path):
    data = {"text": os.path.basename(root_path), "children": []}

    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)
        if os.path.isdir(item_path):
            if _dfilter(item_path):
                data["children"].append(
                    {
                        "text": item,
                        "icon": "jstree-folder",
                        "state": {"opened": False},
                        "data": {"isdirectory": True},
                        # No initial children, they will be loaded on demand
                        "children": True,
                    }
                )
        else:
            if _fnfilter(item_path):
                data["children"].append(
                    {
                        "text": item,
                        "icon": "jstree-file",
                        "data": {"isdirectory": False},
                    }
                )

    return data["children"]


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "misc/favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )
