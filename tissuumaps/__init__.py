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
import glob
import logging
import os
import pathlib
import sys

import yaml
from flask import Flask

log = logging.getLogger()
log.setLevel(logging.INFO)

SLIDE_DIR = "/mnt/data/shared/"
DEFAULT_PROJECT = False
SLIDE_CACHE_SIZE = 60

DEEPZOOM_FORMAT = "jpeg"
DEEPZOOM_TILE_SIZE = 254
DEEPZOOM_OVERLAP = 1
DEEPZOOM_LIMIT_BOUNDS = True
DEEPZOOM_TILE_QUALITY = 90

VIPS_MIN_OUTLIER_PERC = 0.5
VIPS_MAX_OUTLIER_PERC = 99.5
VIPS_FORCE_RESCALE = False
VIPS_JPEG_COMPRESSION = 85
VIPS_EXCLUDE_MIN_INTENSITY = False

PLUGINS = []
DEBUG_CLI = False
READ_ONLY = False
COLLAPSE_TOP_MENU = False
PROJECT_LIST = False

# determine if application is a script file or frozen exe
if getattr(sys, "frozen", False):
    template_folder = os.path.join(sys._MEIPASS, "templates")
    static_folder = os.path.join(sys._MEIPASS, "static")
    plugins_folder = os.path.join(sys._MEIPASS, "plugins")
    version_file = os.path.join(sys._MEIPASS, "VERSION")
    os.chdir(sys._MEIPASS)
else:
    folderPath = os.path.dirname(pathlib.Path(__file__))
    template_folder = os.path.join(folderPath, "templates")
    static_folder = os.path.join(folderPath, "static")
    plugins_folder = os.path.join(folderPath, "plugins")
    version_file = os.path.join(folderPath, "VERSION")

logging.debug("template_folder: " + template_folder)
logging.debug("static_folder: " + static_folder)
with open(version_file) as f:
    tissuumaps_version = f.read().strip()
logging.info(" * TissUUmaps version: " + tissuumaps_version)

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
app.config.from_object(__name__)
app.config["PLUGIN_FOLDER"] = plugins_folder
app.config["VERSION"] = tissuumaps_version
app.config["PLUGIN_FOLDER_USER"] = os.path.join(
    os.path.expanduser("~"), ".tissuumaps", "plugins"
)
app.config.from_envvar("TISSUUMAPS_CONF", silent=True)
userConfigFile = os.path.join(os.path.expanduser("~"), ".tissuumaps", "tissuumaps.cfg")

if os.path.isfile(userConfigFile):
    app.config.from_pyfile(userConfigFile)


def getPluginInFolder(folder):
    pluginNames = [
        os.path.splitext(os.path.basename(module))[0]
        for module in glob.glob(os.path.join(folder, "*.py"))
        if "__init__.py" not in module
    ]
    for pluginName in pluginNames:
        if pluginName in [p["module"] for p in app.config["PLUGINS"]]:
            continue
        yml = os.path.join(folder, pluginName + ".yml")
        if os.path.isfile(yml):
            with open(yml) as file:
                # The FullLoader parameter handles the conversion from YAML
                # scalar values to Python the dictionary format
                pluginInfo = yaml.load(file, Loader=yaml.FullLoader)
        else:
            pluginInfo = {"version": "0.0", "name": pluginName.replace("_", " ")}
        pluginInfo["module"] = pluginName
        app.config["PLUGINS"].append(pluginInfo)


getPluginInFolder(app.config["PLUGIN_FOLDER_USER"])
getPluginInFolder(app.config["PLUGIN_FOLDER"])

app.config["isStandalone"] = False

# must import views after app is created
from . import views  # noqa: E402

views.setup(app)
