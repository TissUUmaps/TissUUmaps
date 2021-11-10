from tissuumaps import views
from IPython.display import HTML, Javascript
from IPython.core.display import display
import threading
import logging
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
    parts = Path(path).parts
    server = TissUUmapsServer(slideDir=parts[0], port=port)
    server.viewer(parts[-1]+"?path="+os.path.join(*parts[1:-1]), height)

if __name__ == '__main__':
    pass