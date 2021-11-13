from tissuumaps import views
from IPython.display import HTML, Javascript
from IPython.core.display import display
import threading
import logging
import json
import click
import warnings
import os, time
from pathlib import Path

def secho(text, file=None, nl=None, err=None, color=None, **styles):
    pass

def echo(text, file=None, nl=None, err=None, color=None, **styles):
    pass

click.echo = echo
click.secho = secho
class TissUUmapsViewer ():
    def __init__(self, server, image, height=700):
        self.server = server
        self.image = image
        self.id = "tissUUmapsViewer_1"
        iframe = ('<iframe src="{src}" style="width: {width}; '
                  'height: {height}; border: none" id="{id}" allowfullscreen></iframe>')
        src = "http://localhost:%d/%s" % (self.server.port, self.image)
        self.htmlIFrame = HTML(iframe.format(width="100%", height=str(height)+"px", src=src, id=self.id))
        display(self.htmlIFrame)
        time.sleep(2)
    
    #def sendJavascript(self):
    #    display(Javascript("document.getElementById('"+self.id+"').contentWindow.postMessage({'module':'HelloWorkd'},'*');"))

class TissUUmapsServer ():
    def __init__(self, slideDir, port=5000):
        
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        log = logging.getLogger('pyvips')
        log.setLevel(logging.ERROR)
        warnings.filterwarnings('ignore')

        self.started = False
        self.port = port
        self.slideDir = slideDir
        views.app.config['SLIDE_DIR'] = slideDir

        def startServer ():
            views.app.run(host="0.0.0.0", port=self.port, threaded=True, debug=False)
        
        if TissUUmapsServer.is_port_in_use(self.port):
            log.warning (f"Port {self.port} already in use. Impossible to start TissUUmaps server.")
            return

        thread = threading.Thread(target = startServer)
        thread.setdaemon = False
        thread.start()
        self.started = True

    @staticmethod
    def is_port_in_use(port):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def viewer (self,image, height=700):
        viewer = TissUUmapsViewer(self,image,height)
        return viewer

def opentmap (path, port=5100, height=700):
    path = os.path.abspath(path)
    parts = Path(path).parts
    server = TissUUmapsServer(slideDir=parts[0], port=port)
    server.viewer(parts[-1]+"?path="+os.path.join(*parts[1:-1]), height)

def loaddata (images=[], csvFiles=[], xSelector="x", ySelector="y", keySelector=None, nameSelector=None, 
              colorSelector=None, piechartSelector=None, shapeSelector=None, scaleSelector=None, 
              fixedShape=None, scaleFactor=1,
              colormap=None,
              compositeMode="source-over",
              boundingBox=None,
              port=5100, height=700, tmapFilename="_project"):
    # make all paths absolute:
    images = [os.path.abspath(f) for f in images]
    csvFiles = [os.path.abspath(f) for f in csvFiles]
    
    # make all paths relative to project file:
    rootPath = os.path.commonpath(
        [os.path.dirname(f) for f in images] + [os.path.dirname(f) for f in csvFiles]
    )
    images = [os.path.relpath(f, rootPath) for f in images]
    csvFiles = [os.path.relpath(f, rootPath) for f in csvFiles]

    # Create json TMAP file:
    jsonTmap = {
        "compositeMode": compositeMode,
        "filename": "",
        "layers": [
            {
                "name": os.path.basename(layer),
                "tileSource": layer + ".dzi"
            } for layer in images
        ],
        "hideTabs": True,
        "markerFiles": []
    }
    if boundingBox:
        jsonTmap["boundingBox"] = {"x":boundingBox[0],"y":boundingBox[1],"width":boundingBox[2],"height":boundingBox[3]}
    if csvFiles:
        expectedHeader = {
                "X": xSelector,
                "Y": ySelector,
                "gb_col": keySelector,
                "gb_name": nameSelector,
                "cb_cmap": colormap,
                "cb_col": colorSelector,
                "scale_col": scaleSelector,
                "scale_factor": scaleFactor,
                "pie_col": piechartSelector,
                "shape_col": shapeSelector,
                "shape_fixed": fixedShape,
                "cb_gr_dict": "",
                "shape_gr_dict": "",
                "opacity": 1
            }
        expectedRadios = {
                "cb_col": colorSelector!=None,
                "cb_gr": colorSelector==None,
                "cb_gr_rand": False,
                "cb_gr_dict": False,
                "cb_gr_key": True,
                "pie_check": piechartSelector!=None,
                "scale_check": scaleSelector!=None,
                "shape_gr": shapeSelector==None and fixedShape==None,
                "shape_gr_rand": shapeSelector==None and fixedShape==None,
                "shape_gr_dict": False,
                "shape_col": shapeSelector!=None,
                "shape_fixed": fixedShape!=None
            }
        if len(csvFiles) == 1:
            csvFiles = csvFiles[0]
        jsonTmap["markerFiles"] = [{
            "comment": "All markers",
            "path": csvFiles,
            "title": "Download markers",
            "autoLoad": True,
            "hideSettings": True,
            "expectedHeader": expectedHeader,
            "expectedRadios": expectedRadios,
            "uid": "markers"
        }]
    tmapFile = os.path.join(rootPath,f"{tmapFilename}.tmap")
    print ("Creating project file", tmapFile)
    with open(tmapFile, "w") as f:
        json.dump(jsonTmap, f)
    opentmap (os.path.abspath(tmapFile), port, height)

if __name__ == '__main__':
    pass