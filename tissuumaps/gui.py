import logging
import sys
import traceback
import warnings

debug_logger = logging.getLogger("root")
debug_logger.write = debug_logger.debug  # consider all prints as debug information
debug_logger.flush = lambda: None  # this may be called when printing
sys.stdout = debug_logger

try:
    from PyQt6 import QtGui
    from PyQt6.QtCore import *
    from PyQt6.QtGui import QAction, QDesktopServices, QStandardItem, QStandardItemModel
    from PyQt6.QtWebChannel import QWebChannel
    from PyQt6.QtWebEngineCore import *
    from PyQt6.QtWebEngineWidgets import *
    from PyQt6.QtWidgets import (
        QApplication,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QLabel,
        QLineEdit,
        QListView,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSplashScreen,
        QStyle,
    )

except ImportError:
    # dependency missing, issue a warning
    logging.error("dependency not found, please install PyQt6 to enable gui")
    logging.error(traceback.format_exc())
    import sys

    sys.exit()

import json
import os
import pathlib
import platform
import random
import re
import socket
import string
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from functools import partial
from optparse import OptionParser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Don't remove this line.  The idna encoding
# is used by getaddrinfo when dealing with unicode hostnames,
# and in some cases, there appears to be a race condition
# where threads will get a LookupError on getaddrinfo() saying
# that the encoding doesn't exist.  Using the idna encoding before
# running any CLI code (and any threads it may create) ensures that
# the encodings.idna is imported and registered in the codecs registry,
# which will stop the LookupErrors from happening.
# See: https://bugs.python.org/issue29288
"".encode("idna")

# determine if application is a script file or frozen exe
if getattr(sys, "frozen", False):
    template_folder = os.path.join(sys._MEIPASS, "templates")
    static_folder = os.path.join(sys._MEIPASS, "static")
    os.chdir(sys._MEIPASS)
else:  # if __file__:
    # template_folder="templates_standalone"
    folderPath = os.path.dirname(pathlib.Path(__file__))
    template_folder = os.path.join(folderPath, "templates")
    static_folder = os.path.join(folderPath, "static")
    os.chdir(folderPath)

from tissuumaps import views


class CustomWebEnginePage(QWebEnginePage):
    """Custom WebEnginePage to customize how we handle link navigation"""

    def acceptNavigationRequest(self, url, _type, isMainFrame):
        if _type == QWebEnginePage.NavigationTypeLinkClicked:
            QDesktopServices.openUrl(url)
            return False
        return True

    # def javaScriptConsoleMessage(self, level, msg, line, sourceID):
    #    logging.debug(
    #        "Javascript console: "
    #        + " ; ".join([str(level), str(msg), str(line), str(sourceID)])
    #    )


class textWindow(QDialog):
    def __init__(self, parent, title, message):
        QDialog.__init__(self, parent)

        self.setMinimumSize(QSize(700, 500))
        self.setWindowTitle(title)

        # Add text field
        self.b = QPlainTextEdit(self)
        self.b.setMinimumSize(650, 450)
        self.b.setReadOnly(True)
        self.b.insertPlainText(message)
        self.b.move(10, 10)
        self.b.resize(400, 200)


class SelectPluginWindow(QDialog):
    def __init__(self, app, parent=None):
        try:
            super(SelectPluginWindow, self).__init__(parent=parent)
            self.app = app
            self.setWindowTitle("Select Plugins")
            form = QFormLayout(self)
            form.addRow(QLabel("Plugin site:"))
            self.textbox = QLineEdit(self)
            self.textbox.setText(
                "https://tissuumaps.github.io/TissUUmaps/plugins/latest/"
            )
            form.addRow(self.textbox)

            # Create a button in the window
            self.button = QPushButton("Get plugins from site", self)
            self.button.move(20, 80)
            form.addRow(self.button)

            # connect button to function on_click
            self.button.clicked.connect(self.getPlugins)
            form.addRow(QLabel("Available plugins:"))
            self.listView = QListView(self)
            form.addRow(self.listView)

            buttonBox = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel,
                Qt.Orientation.Horizontal,
                self,
            )
            form.addRow(buttonBox)
            buttonBox.accepted.connect(self.accept)
            buttonBox.rejected.connect(self.reject)

            self.getPlugins()
        except:
            logging.error(traceback.format_exc())

    def getPlugins(self):
        self.url = self.textbox.text()
        try:
            response = urllib.request.urlopen(self.url + "pluginList.json")
            self.items = json.loads(response.read())
            for plugin in self.items:
                installedPlugins = [
                    p
                    for p in self.app.config["PLUGINS"]
                    if p["module"] == plugin["py"].replace(".py", "")
                ]
                if len(installedPlugins) > 0:
                    plugin["old_version"] = installedPlugins[0]["version"]
                else:
                    plugin["old_version"] = "0.0"

            model = QStandardItemModel(self.listView)
            for item in self.items:
                # create an item with a caption
                updateAvailable = False
                if str(item["old_version"]).split(".") < str(item["version"]).split(
                    "."
                ):
                    updateAvailable = True
                item["updateAvailable"] = updateAvailable
                if updateAvailable:
                    standardItem = QStandardItem(
                        item["name"] + " - v." + str(item["version"]) + " available"
                    )
                else:
                    standardItem = QStandardItem(item["name"])

                standardItem.setCheckState(
                    Qt.CheckState.Checked
                    if not updateAvailable
                    else Qt.CheckState.Unchecked
                )
                if updateAvailable:
                    standardItem.setCheckable(True)
                standardItem.setEditable(False)

                model.appendRow(standardItem)
            self.listView.setModel(model)
        except:
            try:
                QMessageBox.warning(self, "Error", traceback.format_exc())
            except:
                logging.error(traceback.format_exc())

    def itemsSelected(self):
        selected = []
        model = self.listView.model()
        i = 0
        while model.item(i):
            if model.item(i).checkState():
                selected.append(self.items[i])
            i += 1
        return selected


class MainWindow(QMainWindow):
    def __init__(self, qt_app, app, *args, **kwargs):
        super(MainWindow, self).__init__()
        self.resize(1400, 1000)
        self.app = app
        self.browser = webEngine(qt_app, app, self, *args)

        self.setCentralWidget(self.browser)

        # self.status = QStatusBar()
        # self.setStatusBar(self.status)

        self.bar = self.menuBar()
        self.recentActions = []
        self.setStyleSheet(
            """
        QMenuBar {
            border-bottom: 1px solid #911821;
        }
    """
        )
        file = self.bar.addMenu("File")

        _open = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
            "Open",
            self,
        )
        _open.setShortcut("Ctrl+O")
        file.addAction(_open)
        _open.triggered.connect(self.browser.openImage)

        recentFilesMenu = file.addMenu("Open Recent")
        recentFilesMenu.setToolTipsVisible(True)
        for _ in range(self.browser.maxRecent):
            _recentAction = QAction("", self)
            _recentAction.setVisible(False)
            _recentAction.triggered.connect(self.browser.openRecent)
            recentFilesMenu.addAction(_recentAction)
            self.recentActions.append(_recentAction)

        self.browser.updateRecent()

        _save = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            "Save project",
            self,
        )
        _save.setShortcut("Ctrl+S")
        file.addAction(_save)

        def trigger():
            self.browser.page().runJavaScript("flask.standalone.saveProject();")

        _save.triggered.connect(trigger)

        _close = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DockWidgetCloseButton),
            "Close file",
            self,
        )
        file.addAction(_close)
        _close.triggered.connect(self.browser.closeImage)

        file.addSeparator()

        _export = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView),
            "Capture viewport",
            self,
        )
        file.addAction(_export)

        def trigger():
            self.browser.page().runJavaScript("overlayUtils.savePNG();")

        _export.triggered.connect(trigger)

        _export = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DirLinkIcon),
            "Export to static webpage",
            self,
        )
        file.addAction(_export)

        def trigger():
            self.browser.page().runJavaScript("flask.standalone.exportToStatic();")

        _export.triggered.connect(trigger)

        file.addSeparator()

        _exit = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton),
            "Exit",
            self,
        )
        _exit.setShortcut("Ctrl+Q")
        file.addAction(_exit)
        _exit.triggered.connect(self.close)

        plugins = self.bar.addMenu("Plugins")
        for pluginName in [p["module"] for p in app.config["PLUGINS"]]:
            _plugin = QAction(pluginName, self)
            plugins.addAction(_plugin)

            _plugin.triggered.connect(partial(self.triggerPlugin, pluginName))

        plugins.addSeparator()
        _plugin = QAction("Add plugin", self)
        plugins.addAction(_plugin)

        _plugin.triggered.connect(self.addPlugin)

        about = self.bar.addMenu("About")
        _help = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogHelpButton),
            "Help",
            self,
        )
        about.addAction(_help)

        def trigger():
            QDesktopServices.openUrl(QUrl("https://tissuumaps.github.io/"))

        _help.triggered.connect(trigger)
        _version = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogInfoView),
            "Version",
            self,
        )
        about.addAction(_version)

        def trigger():
            if getattr(sys, "frozen", False):
                folderPath = sys._MEIPASS
            else:
                folderPath = os.path.dirname(pathlib.Path(__file__))
            with open(os.path.join(folderPath, "VERSION"), "r") as fh:
                version = fh.read()

            QMessageBox.about(self, "Information", "TissUUmaps version " + version)

        _version.triggered.connect(trigger)

        if self.app.config["DEBUG_CLI"]:
            debug = self.bar.addMenu("Debug")
            _debugpage = QAction(
                self.style().standardIcon(
                    QStyle.StandardPixmap.SP_FileDialogContentsView
                ),
                "Open external debugging",
                self,
            )
            debug.addAction(_debugpage)

            def trigger():
                QDesktopServices.openUrl(QUrl("http://localhost:5588/"))

            _debugpage.triggered.connect(trigger)

        self.showMaximized()

    def triggerPlugin(self, pName):
        logging.debug("Plugin triggered: " + pName)
        self.browser.page().runJavaScript('pluginUtils.startPlugin("' + pName + '");')

    def addPlugin(self):
        logging.debug("Adding plugins")
        try:
            dial = SelectPluginWindow(self.app, self)
            if dial.exec() == QDialog.DialogCode.Accepted:
                changed = False
                for plugin in dial.itemsSelected():
                    if plugin["updateAvailable"]:
                        changed = True
                        for type in ["py", "js", "yml"]:
                            if type in plugin.keys():
                                urlFile = dial.url + plugin[type]
                                localFile = os.path.join(
                                    self.app.config["PLUGIN_FOLDER_USER"], plugin[type]
                                )
                                os.makedirs(
                                    self.app.config["PLUGIN_FOLDER_USER"], exist_ok=True
                                )
                                urllib.request.urlretrieve(urlFile, localFile)
                if changed:
                    QMessageBox.warning(
                        self,
                        "Restart TissUUmaps",
                        "The new plugins will only be available after restarting TissUUmaps.",
                    )
        except:
            logging.error(traceback.format_exc())


class webEngine(QWebEngineView):
    def __init__(self, qt_app, app, mainWin, args):
        super().__init__()
        self.setAcceptDrops(True)
        self.qt_app = qt_app
        self.app = views.app
        self.args = args
        self.maxRecent = 25
        self.setMinimumSize(800, 400)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.lastdir = str(Path.home())
        profile = QWebEngineProfile().defaultProfile()

        # profile.setHttpCacheType(QWebEngineProfile.DiskHttpCache)
        # self.setPage(CustomWebEnginePage(profile, self))
        self.webchannel = QWebChannel()
        self.page().setWebChannel(self.webchannel)
        self.webchannel.registerObject("backend", self)
        self.location = None
        self.mainWin = mainWin

        self.mainWin.setWindowTitle("TissUUmaps")
        # self.mainWin.resize(1024, 800)
        # self.setZoomFactor(1.0)
        # profile.clearHttpCache()
        profile.downloadRequested.connect(self.on_downloadRequested)
        self.settings().setAttribute(
            QWebEngineSettings.WebAttribute.FullScreenSupportEnabled, True
        )

        def setfullscreen(request):
            if self.mainWin.windowState() & Qt.WindowState.WindowFullScreen:
                self.mainWin.showMaximized()
                self.mainWin.bar.setVisible(True)
            else:
                self.mainWin.showFullScreen()
                self.mainWin.bar.setVisible(False)
            request.accept()

        self.page().fullScreenRequested.connect(setfullscreen)

        self.mainWin.setWindowIcon(QtGui.QIcon("static/misc/favicon.ico"))

    def addRecent(self, path):
        os.makedirs(os.path.join(os.path.expanduser("~"), ".tissuumaps"), exist_ok=True)
        recentFile = os.path.join(
            os.path.expanduser("~"), ".tissuumaps", "recents.json"
        )
        if os.path.isfile(recentFile):
            try:
                with open(recentFile) as f:
                    recentFiles = json.load(f)
            except:
                logging.warning(
                    f"Impossible to load: {recentFile}. TissUUmaps will create a new one."
                )
                recentFiles = []
        else:
            recentFiles = []
        if path in recentFiles:
            recentFiles.remove(path)
        recentFiles.insert(0, path)
        recentFiles = recentFiles[: self.maxRecent]
        with open(recentFile, "w") as f:
            json.dump(recentFiles, f)
        self.updateRecent()

    def updateRecent(self):
        recentFile = os.path.join(
            os.path.expanduser("~"), ".tissuumaps", "recents.json"
        )
        if os.path.isfile(recentFile):
            try:
                with open(recentFile) as f:
                    recentFiles = json.load(f)
            except:
                logging.warning(
                    f"Impossible to load: {recentFile}. TissUUmaps will create a new one."
                )
                recentFiles = []
        else:
            recentFiles = []

        for auto in range(self.maxRecent):
            if len(recentFiles) > auto:
                self.mainWin.recentActions[auto].setText(
                    os.path.basename(recentFiles[auto])
                )
                self.mainWin.recentActions[auto].setData(recentFiles[auto])
                self.mainWin.recentActions[auto].setVisible(True)
                self.mainWin.recentActions[auto].setToolTip(recentFiles[auto])
            else:
                self.mainWin.recentActions[auto].setVisible(False)

    def openRecent(self):
        sender = self.sender()
        self.openImagePath(sender.data())

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            for link in links:
                filename, file_extension = os.path.splitext(link)
                if file_extension == ".tmap":
                    self.openImagePath(link)
                elif file_extension == ".h5ad":
                    self.openImagePath(link)
                elif file_extension == ".csv":
                    self.page().runJavaScript(f'flask.standalone.addCSV("{link}");')
                else:
                    self.page().runJavaScript(f'flask.standalone.addLayer("{link}");')
            # self.emit(SIGNAL("dropped"), links)
        else:
            event.ignore()

    def on_downloadRequested(self, download):
        old_path = os.path.join(
            download.downloadDirectory(), download.downloadFileName()
        )
        suffix = os.path.splitext(old_path)[1]
        path, _ = QFileDialog.getSaveFileName(
            self, "Save File", old_path, "*." + suffix
        )
        if path:
            download.setDownloadDirectory(os.path.dirname(path))
            download.setDownloadFileName(os.path.basename(path))
            download.accept()

            def openImageThread():
                for i in range(100):
                    if os.path.isfile(path):
                        break
                    time.sleep(0.1)
                else:
                    return

                if hasattr(os, "startfile"):
                    os.startfile(os.path.normpath(path))
                else:
                    opener = "open" if sys.platform == "darwin" else "xdg-open"
                    subprocess.call([opener, os.path.normpath(path)])

            threading.Thread(target=openImageThread, daemon=True).start()

    def setLocation(self, location):
        self.location = location
        while True:
            try:
                if urllib.request.urlopen(self.location).getcode() == 200:
                    break
            except:
                pass

            logging.error("Impossible to load: " + self.location)
            time.sleep(0.1)
        if len(self.args) > 0:
            if not self.openImagePath(os.path.abspath(self.args[0])):
                self.load(QUrl(self.location))
        else:
            self.load(QUrl(self.location))

    @pyqtSlot(str)
    def getProperties(self, path):
        try:
            path = urllib.parse.unquote(path)[:-4]
            slide = views._get_slide(path)
            propString = "\n".join([n + ": " + v for n, v in slide.properties.items()])
        except:
            propString = ""

        messageBox = textWindow(
            self, os.path.basename(path) + " properties", propString
        )
        messageBox.show()

    def openImage(self):
        folderpath = QFileDialog.getOpenFileName(self, "Select a File", self.lastdir)[0]
        self.openImagePath(folderpath)

    def closeImage(self):
        self.app.basedir = os.path.abspath(self.app.config["SLIDE_DIR"])
        self.load(QUrl(self.location))
        self.mainWin.setWindowTitle("TissUUmaps")

    @pyqtSlot(str, result="QJsonObject")
    def exportToStatic(self, state):
        try:
            parsed_url = urlparse(self.url().toString())
            previouspath = parse_qs(parsed_url.query)["path"][0]
        except:
            previouspath = "./"
        previouspath = os.path.abspath(os.path.join(self.app.basedir, previouspath))

        folderpath = QFileDialog.getExistingDirectory(
            self,
            "Select webpage directory",
            self.lastdir,
            options=QFileDialog.Option.ShowDirsOnly,
        )
        try:
            return views.exportToStatic(state, folderpath, previouspath)
        except:
            return {"success": False, "error": traceback.format_exc()}

    @pyqtSlot(str)
    def saveProject(self, state):
        def getRel(previouspath, file, newpath):
            completepath = os.path.dirname(os.path.join(previouspath, file))
            relPath = os.path.relpath(completepath, newpath)
            return os.path.join(relPath, os.path.basename(file)).replace("\\", "/")

        def addRelativePath(state, previouspath, newpath):
            def addRelativePath_aux(state, path):
                if len(path) == 1:
                    if path[0] not in state.keys():
                        return
                    if isinstance(state[path[0]], list):
                        state[path[0]] = [
                            getRel(previouspath, s, newpath) for s in state[path[0]]
                        ]
                    else:
                        state[path[0]] = getRel(previouspath, state[path[0]], newpath)
                    return
                if not path[0] in state.keys():
                    return
                else:
                    if isinstance(state[path[0]], list):
                        for state_ in state[path[0]]:
                            addRelativePath_aux(state_, path[1:])
                    else:
                        addRelativePath_aux(state[path[0]], path[1:])

            try:
                paths = [
                    ["layers", "tileSource"],
                    ["markerFiles", "path"],
                    ["regionFiles", "path"],
                    ["regionFile"],
                ]
                for path in paths:
                    addRelativePath_aux(state, path)
            except:
                logging.error(traceback.format_exc())

            return state

        folderpath = QFileDialog.getSaveFileName(self, "Save project as", self.lastdir)[
            0
        ]
        if not folderpath:
            return {}
        try:
            parsed_url = urlparse(self.url().toString())
            previouspath = parse_qs(parsed_url.query)["path"][0]
            previouspath = os.path.abspath(os.path.join(self.app.basedir, previouspath))
        except:
            previouspath = self.app.basedir
        state = addRelativePath(
            json.loads(state), previouspath, os.path.dirname(folderpath)
        )
        with open(folderpath, "w") as f:
            json.dump(state, f, indent=4)
        self.addRecent(folderpath)

    def openImagePath(self, folderpath):
        self.lastdir = os.path.dirname(folderpath)
        if not folderpath:
            return
        self.addRecent(folderpath)
        parts = Path(folderpath).parts
        if not hasattr(self.app, "cache"):
            views.setup(self.app)
        self.app.basedir = parts[0]
        imgPath = os.path.join(*parts[1:])
        imgPath = imgPath.replace("\\", "/")

        _, file_extension = os.path.splitext(folderpath)
        if file_extension == ".csv":
            logging.debug(
                " ".join(
                    [
                        "Opening csv:",
                        str(self.app.basedir),
                        str(self.location + imgPath),
                    ]
                )
            )
            self.page().runJavaScript(f'flask.standalone.addCSV("{folderpath}");')
            return True
        logging.debug(
            " ".join(
                [
                    "Opening image:",
                    str(self.app.basedir),
                    str(self.location + imgPath),
                    str(QUrl(self.location + imgPath)),
                ]
            )
        )

        filename = os.path.basename(imgPath)
        path = os.path.dirname(imgPath)
        self.page().runJavaScript("flask.server.loading('Opening image...');")
        self.load(QUrl(self.location + filename + "?path=" + path))
        self.mainWin.setWindowTitle("TissUUmaps - " + os.path.basename(folderpath))
        return True

    @pyqtSlot()
    def exit(self):
        self.close()

    @pyqtSlot(str, str, result="QJsonObject")
    def addCSV(self, path, csvpath):
        if csvpath == "":
            csvpath = QFileDialog.getOpenFileName(self, "Select a File")[0]
        if not csvpath:
            returnDict = {"markerFile": None}
            return returnDict
        parts = Path(csvpath).parts
        if parts[0] == "https:":
            imgPath = parts[-1]
            relativePath = "/".join(parts[:-1])

        else:
            if self.app.basedir != parts[0]:
                if not self.app.basedir == os.path.abspath(
                    self.app.config["SLIDE_DIR"]
                ):
                    QMessageBox.warning(
                        self, "Error", "All files must be in the same drive."
                    )
                    returnDict = {"markerFile": None}
                    return returnDict
                else:
                    self.app.basedir = parts[0]
            imgPath = os.path.join(*parts[1:])

            path = os.path.abspath(os.path.join(self.app.basedir, path))
            imgPath = os.path.abspath(os.path.join(self.app.basedir, imgPath))

            relativePath = os.path.relpath(os.path.dirname(imgPath), path)
        returnDict = {
            "markerFile": {
                "name": os.path.basename(imgPath),
                "path": relativePath + "/" + os.path.basename(imgPath),
                "uid": "".join(random.choice(string.ascii_uppercase) for _ in range(6)),
            }
        }
        return returnDict

    @pyqtSlot(str, str, result="QJsonObject")
    def addLayer(self, path, layerpath):
        if layerpath == "":
            layerpath = QFileDialog.getOpenFileName(self, "Select a File")[0]
        if not layerpath:
            returnDict = {"dzi": None, "name": None}
            return returnDict
        parts = Path(layerpath).parts
        if self.app.basedir != parts[0]:
            if not self.app.basedir == os.path.abspath(self.app.config["SLIDE_DIR"]):
                reply = QMessageBox.question(
                    self,
                    "Error",
                    "All layers must be in the same drive. Would you like to open this image only?",
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.openImagePath(layerpath)
                returnDict = {"dzi": None, "name": None}
                return returnDict
            else:
                self.openImagePath(layerpath)  # self.app.basedir = parts[0]
                returnDict = {"dzi": None, "name": None}
                return returnDict
        imgPath = os.path.join(*parts[1:])
        try:
            views._get_slide(imgPath)
        except:
            logging.error(traceback.format_exc())
            QMessageBox.about(
                self, "Error", "TissUUmaps did not manage to open this image."
            )
            returnDict = {"dzi": None, "name": None}
            return returnDict
        path = os.path.abspath(os.path.join(self.app.basedir, path))
        imgPath = os.path.abspath(os.path.join(self.app.basedir, imgPath))
        relativePath = os.path.relpath(os.path.dirname(imgPath), path)
        returnDict = {
            "dzi": relativePath + "/" + os.path.basename(imgPath) + ".dzi",
            "name": os.path.basename(imgPath),
        }
        return returnDict


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def main():
    parser = OptionParser(usage="Usage: %prog [options] [slide-directory]")
    parser.add_option(
        "-B",
        "--ignore-bounds",
        dest="DEEPZOOM_LIMIT_BOUNDS",
        default=False,
        action="store_false",
        help="display entire scan area",
    )
    parser.add_option(
        "-c", "--config", metavar="FILE", dest="config", help="config file"
    )
    parser.add_option(
        "-d",
        "--debug",
        dest="DEBUG",
        action="store_true",
        help="run in debugging mode (insecure)",
    )
    parser.add_option(
        "-e",
        "--overlap",
        metavar="PIXELS",
        dest="DEEPZOOM_OVERLAP",
        type="int",
        help="overlap of adjacent tiles [1]",
    )
    parser.add_option(
        "-f",
        "--format",
        metavar="{jpeg|png}",
        dest="DEEPZOOM_FORMAT",
        help="image format for tiles [jpeg]",
    )
    parser.add_option(
        "-l",
        "--listen",
        metavar="ADDRESS",
        dest="host",
        default="127.0.0.1",
        help="address to listen on [127.0.0.1]",
    )
    parser.add_option(
        "-p",
        "--port",
        metavar="PORT",
        dest="port",
        type="int",
        default=5432,
        help="port to listen on [5432]",
    )
    parser.add_option(
        "-Q",
        "--quality",
        metavar="QUALITY",
        dest="DEEPZOOM_TILE_QUALITY",
        type="int",
        help="JPEG compression quality [75]",
    )
    parser.add_option(
        "-s",
        "--size",
        metavar="PIXELS",
        dest="DEEPZOOM_TILE_SIZE",
        type="int",
        help="tile size [254]",
    )
    parser.add_option(
        "-D",
        "--depth",
        metavar="LEVELS",
        dest="FOLDER_DEPTH",
        type="int",
        help="folder depth search for opening files [4]",
    )

    (opts, args) = parser.parse_args()

    if opts.DEBUG:
        views.app.config["DEBUG_CLI"] = True
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.DEBUG)
        log = logging.getLogger("pyvips")
        log.setLevel(logging.DEBUG)
        log = logging.getLogger()
        log.setLevel(logging.DEBUG)
        warnings.filterwarnings("default")
        logging.debug("Debug mode")

        DEBUG_PORT = "5588"
        DEBUG_URL = "http://127.0.0.1:%s" % DEBUG_PORT
        os.environ["QTWEBENGINE_REMOTE_DEBUGGING"] = DEBUG_PORT

        # os.environ['WERKZEUG_RUN_MAIN'] = 'true'
    else:
        views.app.config["DEBUG_CLI"] = False
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        log = logging.getLogger("pyvips")
        log.setLevel(logging.ERROR)
        log = logging.getLogger()
        log.setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")
        # os.environ['WERKZEUG_RUN_MAIN'] = 'true'

    # Overwrite only those settings specified on the command line
    for k in dir(opts):
        if not k.startswith("_") and getattr(opts, k) is None:
            delattr(opts, k)
    views.app.config.from_object(opts)
    views.app.config["isStandalone"] = True

    qInstallMessageHandler(lambda x, y, z: None)

    fmt = QtGui.QSurfaceFormat()
    if platform.system() == "Darwin":
        fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setVersion(4, 1)
    fmt.setSwapBehavior(QtGui.QSurfaceFormat.SwapBehavior.DoubleBuffer)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL)

    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--no-sandbox --ignore-gpu-blacklist"
    qt_app = QApplication(sys.argv)

    logo = QtGui.QPixmap("static/misc/design/logo.png")
    logo = logo.scaledToWidth(512, Qt.TransformationMode.SmoothTransformation)
    splash = QSplashScreen(logo, Qt.WindowType.WindowStaysOnTopHint)

    # desktop = qt_app.desktop()
    # scrn = desktop.screenNumber(QtGui.QCursor.pos())
    # currentDesktopsCenter = desktop.availableGeometry(scrn).center()
    # splash.move(currentDesktopsCenter - splash.rect().center())

    splash.show()
    # splash.showMessage('Loading TissUUmaps...',Qt.AlignBottom | Qt.AlignCenter,Qt.white)

    qt_app.processEvents()

    port = opts.port
    logging.info("Starting port detection")
    while is_port_in_use(port):
        port += 1
        if port == 6000:
            exit(0)
    logging.info("Ending port detection " + str(port))

    def flaskThread():
        views.setup(views.app)
        views.app.run(host="127.0.0.1", port=port, threaded=True, debug=False)

    threading.Thread(target=flaskThread, daemon=True).start()

    ui = MainWindow(qt_app, views.app, args)
    ui.browser.setLocation("http://127.0.0.1:" + str(port) + "/")

    QTimer.singleShot(1000, splash.close)
    qt_app.exec()


if __name__ == "__main__":
    main()
