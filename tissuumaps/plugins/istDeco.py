from flask import Flask, abort, make_response, render_template
from openslide import OpenSlideError
import os, glob
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import pyvips
from urllib.parse import unquote

#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import importlib
#importlib.import_module('mpl_toolkits.axes_grid1').make_axes_locatable
import base64

class PILBytesIO(BytesIO):
    def fileno(self):
        '''Classic PIL doesn't understand io.UnsupportedOperation.'''
        raise AttributeError('Not supported')

class ImageConverter():
    def __init__(self, inputImage, outputImage):
        self.inputImage = inputImage
        self.outputImage = outputImage
    
    def convert (self):
        if not os.path.isfile(self.outputImage):
            try:
                imgVips = pyvips.Image.new_from_file(self.inputImage)
                minVal = imgVips.percent(10)
                maxVal = imgVips.percent(99)
                if minVal == maxVal:
                    minVal = 0
                    maxVal = 255
                print ("minVal, maxVal", minVal, maxVal)
                imgVips = (255.* (imgVips - minVal)) / (maxVal - minVal)
                imgVips = (imgVips < 0).ifthenelse(0, imgVips)
                imgVips = (imgVips > 255).ifthenelse(255, imgVips)
                print ("minVal, maxVal", imgVips.min(), imgVips.max())
                imgVips = imgVips.scaleimage()
                imgVips.tiffsave(self.outputImage, pyramid=True, tile=True, tile_width=256, tile_height=256, properties=True, bitdepth=8)
            except: 
                print ("Impossible to convert image using VIPS:")
                import traceback
                print (traceback.format_exc())
            self.convertDone = True
        return self.outputImage

class Plugin ():
    def __init__(self, app):
        self.app = app

    def _get_slide(self, path):
        path = os.path.abspath(os.path.join(self.app.basedir, path))
        print (path)
        if not path.startswith(self.app.basedir):
            # Directory traversal
            print ("Directory traversal, aborting.")
            abort(500)
        if not os.path.exists(path):
            print ("not os.path.exists, aborting.")
            abort(500)
        try:
            slide = self.app.cache.get(path)
            return slide
        except OpenSlideError:
            if ".tissuumaps" in path:
                abort(500)
            try:
                newpath = os.path.dirname(path) + "/.tissuumaps/" + os.path.basename(path)
                if not os.path.isdir(os.path.dirname(path) + "/.tissuumaps/"):
                    os.makedirs(os.path.dirname(path) + "/.tissuumaps/")
                path = ImageConverter(path,newpath).convert()
                #imgPath = imgPath.replace("\\","/")
                return self._get_slide(path)
            except:
                import traceback
                print (traceback.format_exc())
                print ("OpenSlideError, aborting.")
                abort(500)
    
    def getTile (self, path, bbox):
        path = path.replace(".dzi","")
        if path[0] == "\\" or path[0] == "/":
            path = path[1:]
        slide = self._get_slide(path)
        try:
            with slide.tileLock:
                tile = slide.osr.read_region((bbox[0],bbox[1]), 0, (bbox[2], bbox[3]))
        except ValueError:
            # Invalid level or coordinates
            print ("ValueError, aborting.")
            abort(500)
        return tile

    def getConcat(self, tiles, rounds, channels):
        singleWidth = tiles[rounds[0]][channels[0]].width
        singleHeight = tiles[rounds[0]][channels[0]].height
        
        width = len(channels) * singleWidth
        height = len(rounds) * singleHeight

        dst = Image.new('RGB', (width, height))
        for row, round in enumerate(rounds):
            for col, channel in enumerate(channels):
                try:
                    dst.paste(tiles[round][channel], (col * singleWidth, row * singleHeight))
                except:
                    pass
        return dst
        
    def getPlot (self, tiles, rounds, channels, markers, bbox):
        singleWidth = tiles[rounds[0]][channels[0]].width
        singleHeight = tiles[rounds[0]][channels[0]].height
        
        im = self.getConcat(tiles, rounds, channels).convert("L")
        fig = plt.figure(figsize=(5, 4), dpi=80)
        ax = fig.add_subplot(111)
        #plt.axis('off')
        imcolor = plt.imshow(im, cmap='Greys_r')
        
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(imcolor, fraction=0.036, pad=0.05)#, cax=cax)

        for xIndex in range(len(channels)+1):
            ax.axvline(x=singleWidth * xIndex - 0.5, color="red", linewidth=1)
        for yIndex in range(len(rounds)+1):
            ax.axhline(y=singleHeight * yIndex - 0.5, color="red", linewidth=1)
        
        for marker in markers:
            try:
                x, y = [], []
                if "rounds" in marker.keys():
                    markerRounds = marker["rounds"].split(";")
                    markerRounds = [rounds[int(m)] for m in markerRounds]
                else:
                    markerRounds = rounds
                if "channels" in marker.keys():
                    markerchannels = marker["channels"].split(";")
                    markerchannels = [channels[int(m)] for m in markerchannels]
                else:
                    markerchannels = marker["letters"]
                
                offset = marker["global_X_pos"] - bbox[0]-0.5, marker["global_Y_pos"] - bbox[1]-0.5
                for yIndex, (markerchannel, markerRound) in enumerate(zip(markerchannels, markerRounds)):
                    xIndex = channels.index(markerchannel)
                    x.append(offset[0] + singleWidth * xIndex)
                    y.append(offset[1] + singleWidth * yIndex)
                ax.plot(x, y, 'o-', label=markerchannels, color=marker["color"], markersize=5, marker="x")
            except:
                import traceback
                traceback.print_exc()
                pass
        
        ax.set_xticks([i*singleWidth + singleWidth/2-0.5 for i,_ in enumerate(channels)])
        ax.set_xticklabels(channels)
        ax.set_yticks([i*singleHeight + singleHeight/2-0.5 for i,_ in enumerate(rounds)])
        ax.set_yticklabels(rounds, rotation=90, va="center")
        ax.tick_params(axis=u'both', which=u'both',length=0)
        plt.tight_layout()

        buf = PILBytesIO()
        fig.savefig(buf)
        #plt.close(fig)
        return buf
        
    def getMatrix (self, jsonParam):
        if (not jsonParam):
            print ("No arguments, aborting.")
            abort(500)
        bbox = jsonParam["bbox"]
        layers = jsonParam["layers"]
        markers = jsonParam["markers"]
        print ("getMatrix", bbox, layers, markers)
        tiles = {}
        rounds = jsonParam["order_rounds"]
        channels = jsonParam["order_channels"]
        if rounds is None or channels is None:
            rounds = []
            channels = []
            for layer in layers:
                round, channel = layer["name"].split("_")
                if round not in rounds:
                    rounds.append(round)
                if channel not in channels:
                    channels.append(channel)
        for layer in layers:
            round, channel = layer["name"].split("_")
            if round not in tiles.keys():
                tiles[round] = {}
            tiles[round][channel] = self.getTile(layer["tileSource"], bbox)

        plot = self.getPlot(tiles, rounds, channels, markers, bbox)
        format = "png"
        img_str = base64.b64encode(plot.getvalue())
        resp = make_response(img_str)
        #resp.mimetype = 'image/%s' % format
        #resp.cache_control.max_age = 0
        #resp.cache_control.public = True
        return resp

    def importFolder (self, jsonParam):
        if (not jsonParam):
            print ("No arguments, aborting.")
            abort(500)
        relativepath = unquote(jsonParam["path"])
        print ('jsonParam["path"]', jsonParam["path"])
        if relativepath[0] == "/":
            relativepath = relativepath[1:]
        path = os.path.abspath(os.path.join(self.app.basedir, relativepath))
        absoluteRoot = os.path.abspath(self.app.basedir)
        print ("path",relativepath, path, absoluteRoot)
        tifFiles_ = glob.glob(path + "/*")
        tifFiles = []
        for tifFile in tifFiles_:
            if tifFile in tifFiles:
                continue
            try:
                self._get_slide(tifFile)
                tifFiles.append(tifFile)
            except:
                print ("impossible to read", tifFile,". Abort this file.")
                continue
        print (tifFiles)
        csvFiles = glob.glob(path + "/*.csv")
        csvFilesDesc = []
        for csvFile in csvFiles:
            filePath = csvFile.replace(absoluteRoot,"")
            if (filePath[0] != "/" and filePath[0] != "\\" ):
                filePath = "\\" + filePath
            csvFilesDesc.append({
                "path": filePath,
                "title":"Download " + os.path.basename(csvFile),
                "comment":"",
                "expectedCSV":{ "group": "target", "name": "gene", "X_col": "x", "Y_col": "y", "key": "letters" }
            })
        
        layers = []
        layerFilters = {}
        rounds = []
        channels = []
        colors = ["100,0,0","0,100,0","0,0,100","100,100,0","100,0,100","0,100,100"]
        for fileIndex, filename in enumerate(sorted(tifFiles)):
            basename = os.path.basename (filename)
            if "_" in basename:
                channel = os.path.splitext (basename)[0].split("_")[1]
                #round = os.path.basename (os.path.dirname (filename))
                round = os.path.splitext (basename)[0].split("_")[0]
            else:
                channel = os.path.splitext (basename)[0]
                #round = os.path.basename (os.path.dirname (filename))
                round = ""
            if channel not in channels:
                channels.append(channel)
            filePath = filename.replace(absoluteRoot,"")
            filePath = filePath.replace("\\","/")
            if (filePath[0] != "/"):
                filePath = "/" + filePath
            layer = {
                "name":basename,
                "path":filePath
            }
            print (channels, channel)
            print (channels.index(channel)%len(colors))
            layerFilter = [{"value": colors[channels.index(channel)%len(colors)],"name": "Color"}]
            layerFilters[fileIndex] = layerFilter
            layers.append(layer)
        jsonFile = {
            "markerFiles": csvFilesDesc,
            "CPFiles": [],
            "filters": ["Color"],
            "layers": layers,
            "layerFilters": layerFilters,
            "slideFilename": os.path.basename(path),
            "compositeMode": "lighter"
        }
        return jsonFile
        # {
        #     markerFiles: [
        #         {
        #             path: "my/server/path.csv",
        #             title: "",
        #             comment: ""
        #         }
        #     ],
        #     CPFiles: [],
        #     regionFiles: [],
        #     layers: [
        #         {
        #             name:"",
        #             path:""
        #         }
        #     ],
        #     filters: [
        #         {
        #             name:"",
        #             default:"",
        #         }
        #     ],
        #     compositeMode: ""
        # }
    
