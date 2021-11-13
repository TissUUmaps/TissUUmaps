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
import os
import pathlib
import sys
from flask import Flask
from optparse import OptionParser
import logging

SLIDE_DIR = "/mnt/data/shared/"
SLIDE_CACHE_SIZE = 10
DEEPZOOM_FORMAT = 'png'
DEEPZOOM_TILE_SIZE = 254
DEEPZOOM_OVERLAP = 1
DEEPZOOM_LIMIT_BOUNDS = True
DEEPZOOM_TILE_QUALITY = 90

FOLDER_DEPTH = 4
PLUGINS = []

# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    template_folder=os.path.join(sys._MEIPASS, 'templates')
    static_folder=os.path.join(sys._MEIPASS, 'static')
    plugins_folder=os.path.join(sys._MEIPASS, 'plugins')
    os.chdir(sys._MEIPASS)
else:
    folderPath = os.path.dirname(pathlib.Path(__file__))
    template_folder=os.path.join(folderPath, 'templates')
    static_folder=os.path.join(folderPath, 'static')
    plugins_folder=os.path.join(folderPath, 'plugins')

logging.info("template_folder: " + template_folder)
logging.info("static_folder: " + static_folder)

app = Flask(__name__,template_folder=template_folder,static_folder=static_folder)
app.config.from_object(__name__)
app.config.from_envvar('DEEPZOOM_MULTISERVER_SETTINGS', silent=True)
app.config["PLUGIN_FOLDER"] = plugins_folder

for module in glob.glob(app.config["PLUGIN_FOLDER"] + "/*.py"):
    if "__init__.py" in module:
        continue
    app.config["PLUGINS"].append(os.path.splitext(os.path.basename(module))[0])
logging.info("Plugin list:",app.config["PLUGINS"])

app.config["isStandalone"] = False

from . import views
