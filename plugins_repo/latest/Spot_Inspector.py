import glob
import logging
import os
from urllib.parse import unquote

from flask import abort, make_response

import tissuumaps.views as tv


class Plugin:
    def __init__(self, app):
        self.app = app

    def importFolder(self, jsonParam):
        if not jsonParam:
            logging.error("No arguments, aborting.")
            abort(500, "No arguments, aborting.")
        if jsonParam["path"] == None:
            jsonParam["path"] = "./"
        relativepath = unquote(jsonParam["path"])
        pathFormat = unquote(jsonParam["pathFormat"])
        if relativepath != "":
            if relativepath[0] == "/":
                relativepath = relativepath[1:]
        path = os.path.abspath(os.path.join(self.app.basedir, relativepath))
        absoluteRoot = os.path.abspath(self.app.basedir)
        tifFiles_ = glob.glob(path + "/" + pathFormat)
        tifFiles = []
        for tifFile in tifFiles_:
            if tifFile in tifFiles:
                continue
            try:
                tv._get_slide(tifFile)
                tifFiles.append(tifFile)
            except:
                logging.error("impossible to read " + tifFile + ". Abort this file.")
                continue
        # csvFiles = glob.glob(path + "/*.csv")
        csvFilesDesc = []
        # for csvFile in csvFiles:
        #    filePath = os.path.relpath(csvFile, path)
        #    filePath = filePath.replace("\\","/")
        #    csvFilesDesc.append({
        #        "path": filePath,
        #        "title":"Download " + os.path.basename(csvFile),
        #        "comment":"",
        #        "expectedCSV":{ "group": "target", "name": "gene", "X_col": "x", "Y_col": "y", "key": "letters" }
        #    })

        layers = []
        layerFilters = {}
        rounds = []
        channels = []
        colors = [
            "100,0,0",
            "0,100,0",
            "0,0,100",
            "100,100,0",
            "100,0,100",
            "0,100,100",
        ]
        for fileIndex, filename in enumerate(sorted(tifFiles)):
            basename = os.path.basename(filename)
            if "_" in basename:
                channel = os.path.splitext(basename)[0].split("_")[1]
                # round = os.path.basename (os.path.dirname (filename))
                round = os.path.splitext(basename)[0].split("_")[0]
            else:
                channel = os.path.splitext(basename)[0]
                # round = os.path.basename (os.path.dirname (filename))
                round = ""
            if channel not in channels:
                channels.append(channel)
            filePath = os.path.relpath(filename, path)
            filePath = filePath.replace("\\", "/")
            if filePath[0] != "/":
                filePath = "/" + filePath
            layer = {"name": filePath, "tileSource": filePath + ".dzi"}
            layerFilter = [
                {
                    "value": colors[channels.index(channel) % len(colors)],
                    "name": "Color",
                }
            ]
            layerFilters[fileIndex] = layerFilter
            layers.append(layer)
        jsonFile = {
            "markerFiles": csvFilesDesc,
            "CPFiles": [],
            "filters": ["Color"],
            "layers": layers,
            "layerFilters": layerFilters,
            "slideFilename": os.path.basename(path),
            "compositeMode": "lighter",
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
