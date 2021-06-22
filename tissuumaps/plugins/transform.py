from flask import Flask, abort, make_response, render_template
from openslide import OpenSlideError
import os, glob
from io import BytesIO
from PIL import Image
import pyvips

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
    
    def resize (self, scale):
        #if not os.path.isfile(self.outputImage):
            try:
                imgVips = pyvips.Image.new_from_file(self.inputImage)
                vips_image = imgVips.affine((scale,0,0,scale),
                    odx = 0,
                    ody = 0, 
                    extend="black",
                    oarea=[0, 0, int(vips_image.width+T[4]), int(vips_image.height+T[5])]
                )
                vips_image.tiffsave(self.outputImage, pyramid=True, tile=True, tile_width=256, tile_height=256, properties=True, bitdepth=8)
            except: 
                print ("Impossible to convert image using VIPS:")
                import traceback
                print (traceback.format_exc())
            self.convertDone = True
            return self.outputImage

    def transform (self, T):
        #if not os.path.isfile(self.outputImage):
            try:
                imgVips = pyvips.Image.new_from_file(self.inputImage)
                computedWidth, computedHeight = 0, 0
                for (x, y) in [(0,0),(imgVips.width,0),(imgVips.width,imgVips.height),(0,imgVips.height)]:
                    print (x,y, int(x*T[0]+y*T[1]+T[4]), int(x*T[2]+y*T[3]+T[5]), int(x*T[0]+y*T[1]), int(x*T[2]+y*T[3]))
                    computedWidth = max(computedWidth, int(x*T[0]+y*T[1]+T[4]))
                    computedHeight = max(computedHeight, int(x*T[2]+y*T[3]+T[5]))

                print ("computedWidth, computedHeight, imgVips.width, imgVips.height, matrix:", computedWidth, computedHeight, imgVips.width, imgVips.height, T)
                vips_image = imgVips.affine((T[0],T[1],T[2],T[3]),
                                      odx = T[4],
                                      ody = T[5], 
                                      extend="black",
                                      oarea=[0, 0, computedWidth, computedHeight]
                                     )
                #vips_image = vips_image.affine((1,0,0,1),
                #                      odx = T[4],
                #                      ody = T[5], 
                #                      extend="black",
                #                      oarea=[0, 0, int(vips_image.width+T[4]), int(vips_image.height+T[5])]
                #                     )
                vips_image.tiffsave(self.outputImage, pyramid=True, tile=True, tile_width=256, tile_height=256, properties=True, bitdepth=8)
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
            print ("OpenSlideError, aborting.")
            abort(500)
    
    def transform (self, jsonParam):
        if (not jsonParam):
            print ("No arguments, aborting.")
            abort(500)
        print (jsonParam, self.app.basedir)
        tifFile = jsonParam["path"].replace(".dzi","")
        transMatrix = jsonParam["matrix"]
        outputSuffix = jsonParam["outputSuffix"]
        if tifFile[0] == "/" or tifFile[0] == "\\":
            tifFile = tifFile[1:]
        tifFile = os.path.abspath(os.path.join(self.app.basedir, tifFile))
        print ("impossible to read", tifFile,". Trying to convert using VIPS.")
        newpath = os.path.splitext(tifFile)[0] + outputSuffix + ".tif"
        newTifFile = ImageConverter(tifFile,newpath).transform(transMatrix)
        image = self._get_slide(newTifFile)
        
        absoluteRoot = os.path.abspath(self.app.basedir)
        filePath = newTifFile.replace(absoluteRoot,"")
        filePath = filePath.replace("\\","/")
        if (filePath[0] != "/"):
            filePath = "/" + filePath
        
        jsonFile = {
            "image": filePath
        }
        return jsonFile
    