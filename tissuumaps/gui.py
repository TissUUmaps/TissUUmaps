
from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QPlainTextEdit, QDialog, QSplashScreen, QProgressDialog
from PyQt5.QtWebChannel import QWebChannel
from PyQt5 import QtGui 
from PyQt5.QtGui import QDesktopServices
from optparse import OptionParser

from pathlib import Path

import threading, time
import sys
import socket

import urllib.parse
import urllib.request
import os

from tissuumaps import views

class CustomWebEnginePage(QWebEnginePage):
    """ Custom WebEnginePage to customize how we handle link navigation """

    def acceptNavigationRequest(self, url,  _type, isMainFrame):
        if _type == QWebEnginePage.NavigationTypeLinkClicked:
            QDesktopServices.openUrl(url)
            return False
        return True
    
    #def javaScriptConsoleMessage(self, level, msg, line, sourceID):
    #    print (level, msg, line, sourceID)

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
        self.qt_app = qt_app
        self.app = views.app
        self.args = args
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
        sys.exit(self.qt_app.exec_())

    def setLocation (self, location):
        self.location = location
        while True:
            try:
                if (urllib.request.urlopen(self.location).getcode() == 200):
                    break
            except:
                pass
            
            print ("Impossible to load",self.location)
            time.sleep(0.1)
        print ("loading page ", self.location)
        if (len(self.args) > 0):
            if not self.openImagePath(self.args[0]):
                self.load(QUrl(self.location))
        else:
            self.load(QUrl(self.location))
            
    @pyqtSlot(str)
    def getProperties(self, path):
        try:
            path = urllib.parse.unquote(path)[:-4]
            print (path)
            slide = views._get_slide(path)
            propString = "\n".join([n + ": " + v for n,v in slide.properties.items()])
        except:
            propString = ""
        
        messageBox = textWindow(self,os.path.basename(path) + " properties", propString)
        messageBox.show()
        
    @pyqtSlot()
    def openImage(self):
        folderpath = QFileDialog.getOpenFileName(self, 'Select a File',self.lastdir)[0]
        self.openImagePath(folderpath)

    @pyqtSlot(result="QJsonObject")
    def saveProject(self):
        folderpath = QFileDialog.getSaveFileName(self, 'Save project as',self.lastdir)[0]
        parts = Path(folderpath).parts
        if (self.app.basedir != parts[0]):
            QMessageBox.about(self, "Error", "All layers must be in the same drive")
            returnDict = {"dzi":None,"name":None}
            return returnDict
        imgPath = os.path.join(*parts[1:])
        imgPath = imgPath.replace("\\","/") 
        returnDict = {
            "path":imgPath
        }
        return returnDict
    
    def openImagePath (self, folderpath):
        print ("openImagePath",folderpath)
        try:
            oldBaseDir = self.app.basedir
        except AttributeError:
            oldBaseDir = ""
        self.lastdir = os.path.dirname(folderpath)
        if not folderpath:
            return
        print ("openImagePath",oldBaseDir, folderpath)
        parts = Path(folderpath).parts
        if (not hasattr(self.app, 'cache')):
            setup(self.app)
        self.app.basedir = parts[0]
        imgPath = os.path.join(*parts[1:])
        imgPath = imgPath.replace("\\","/")
        try:
            if not ".tmap" in imgPath:
                views._get_slide(imgPath)
        except:
            self.app.basedir = oldBaseDir
            import traceback
            print (traceback.format_exc())
            QMessageBox.about(self, "Error", "TissUUmaps did not manage to open this image.")

            return False
        print ("Opening:", self.app.basedir, self.location + imgPath, QUrl(self.location + imgPath))
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
        if (self.app.basedir != parts[0]):
            QMessageBox.about(self, "Error", "All layers must be in the same drive")
            returnDict = {"dzi":None,"name":None}
            return returnDict
        imgPath = os.path.join(*parts[1:])
        try:
            views._get_slide(imgPath)
        except:
            import traceback
            print (traceback.format_exc())
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

def main():
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
    # Overwrite only those settings specified on the command line
    for k in dir(opts):
        if not k.startswith('_') and getattr(opts, k) is None:
            delattr(opts, k)
    views.app.config.from_object(opts)
    views.app.config["isStandalone"] = True

    qInstallMessageHandler(lambda x,y,z: None)

    fmt = QtGui.QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
    fmt.setSamples(4)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)

    vp = QtGui.QOpenGLVersionProfile(fmt)
    
    qt_app = QApplication([])

    logo = QtGui.QPixmap('static/misc/design/logo.png')
    logo = logo.scaledToWidth(512, Qt.SmoothTransformation)
    splash = QSplashScreen(logo, Qt.WindowStaysOnTopHint)

    desktop = qt_app.desktop()
    scrn = desktop.screenNumber(QtGui.QCursor.pos())
    currentDesktopsCenter = desktop.availableGeometry(scrn).center()
    splash.move(currentDesktopsCenter - splash.rect().center())

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
        views.app.run(host="127.0.0.1", port=port, threaded=True, debug=False)

    threading.Thread(target=flaskThread,daemon=True).start()

    ui = webEngine(qt_app, views.app, args)
    ui.setLocation ("http://127.0.0.1:" + str(port) + "/")

    QTimer.singleShot(1000, splash.close)
    ui.run()

if __name__ == '__main__':
    main ()