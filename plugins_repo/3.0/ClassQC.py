import glob
import os
from io import BytesIO

import matplotlib
from flask import abort, make_response

matplotlib.use("Agg")

import base64
import logging
from urllib.parse import unquote

import matplotlib.pyplot as plt
import numpy as np
import pyvips

import tissuumaps.views as tv


class PILBytesIO(BytesIO):
    def fileno(self):
        """Classic PIL doesn't understand io.UnsupportedOperation."""
        raise AttributeError("Not supported")


class Plugin:
    def __init__(self, app):
        self.app = app

    def getTile(self, slide, bbox):
        tile = slide._osr.read_region((bbox[0], bbox[1]), 0, (bbox[2], bbox[3]))

        return np.array(tile.convert("L"))

    # def getConcat(self, tiles, rounds, channels):
    #     tilesArray = []
    #     for row, round in enumerate(rounds):
    #         channelsArray = []
    #         for col, channel in enumerate(channels):
    #             channelsArray.append(tiles[round][channel])
    #         tilesArray.append(channelsArray)
    #     return np.block(tilesArray)

    def getConcat(self, tiles, rounds, channels):
        tilesArray = []

        for row, channel in enumerate(channels):
            tilesArray.append([tiles["_"][channel]])
        #     print (tiles["_"][channel].shape)
        # print (tilesArray)
        return np.block(tilesArray)

    def getPlot(self, tiles, rounds, channels, bbox):
        singleWidth = tiles[rounds[0]][channels[0]].shape[0]
        singleHeight = tiles[rounds[0]][channels[0]].shape[1]
        im = self.getConcat(tiles, rounds, channels)  # .convert("L")
        figureRatio = (len(rounds) + 2) / len(channels)
        fig = plt.figure(
            figsize=(self.figureSize * figureRatio, self.figureSize), dpi=80
        )
        ax = fig.add_subplot(111)
        imcolor = plt.imshow(im, cmap=plt.get_cmap(self.cmap))

        plt.colorbar(imcolor, fraction=0.036, pad=0.05)

        # for xIndex in range(1):
        #     ax.axvline(x=singleWidth * xIndex - 0.5, color="red", linewidth=1)
        for yIndex in range(len(channels) + 1):
            ax.axhline(y=singleHeight * yIndex - 0.5, color="red", linewidth=1)

        # for marker in markers:
        #     try:
        #         x, y = [], []
        #         if "rounds" in marker.keys():
        #             markerRounds = marker["rounds"].split(";")
        #             markerRounds = [rounds[int(m)] for m in markerRounds]
        #         else:
        #             markerRounds = rounds
        #         if "channels" in marker.keys():
        #             markerchannels = marker["channels"].split(";")
        #             markerchannels = [channels[int(m)] for m in markerchannels]
        #         else:
        #             markerchannels = marker["letters"]

        #         offset = (
        #             marker["global_X_pos"] - bbox[0] - 0.5,
        #             marker["global_Y_pos"] - bbox[1] - 0.5,
        #         )
        #         for yIndex, (markerchannel, markerRound) in enumerate(
        #             zip(markerchannels, markerRounds)
        #         ):
        #             xIndex = channels.index(markerchannel)
        #             yIndex_ = rounds.index(markerRound)
        #             x.append(offset[0] + singleWidth * xIndex)
        #             y.append(offset[1] + singleWidth * yIndex_)
        #         ax.plot(
        #             x,
        #             y,
        #             "o-",
        #             label=markerchannels,
        #             color=marker["color"],
        #             markersize=5,
        #             marker="x",
        #         )
        #     except:
        #         import traceback

        #         logging.error(traceback.format_exc())
        #         pass

        # ax.set_xticks(
        #     [i * singleWidth + singleWidth / 2 - 0.5 for i, _ in enumerate(channels)]
        # )
        # ax.set_xticklabels([c.replace(".tif", "") for c in channels])
        ax.set_xticklabels("")
        ax.set_yticks(
            [i * singleHeight + singleHeight / 2 - 0.5 for i, _ in enumerate(channels)]
        )
        # ax.set_yticklabels(['DAPI', 'Opal 480', 'Opal 520', 'Opal 540', 'Opal 570', 'Opal 620', 'Opal 650', 'Opal 690', 'Opal 780', 'Autofluorescence'],
        #                    rotation=90, va='center')
        ax.set_yticklabels(channels, rotation=90, va="center")
        ax.tick_params(axis="both", which="both", length=0)

        buf = PILBytesIO()
        fig.savefig(buf, bbox_inches="tight")
        fig.clf()
        # plt.close()
        return buf

    def getMatrix(self, jsonParam):
        if not jsonParam:
            logging.error("No arguments, aborting.")
            abort(500)
        # print(jsonParam)
        bbox = jsonParam["bbox"]
        layers = jsonParam["layers"]
        path = jsonParam["path"]
        for layer in layers:
            if not "_" in layer["name"]:
                layer["name"] = layer["name"] + "_1"
        self.figureSize = jsonParam["figureSize"]
        if "cmap" in jsonParam.keys():
            self.cmap = jsonParam["cmap"]
        else:
            self.cmap = "Greys_r"
        tiles = {}
        channels = [layer["name"] for layer in layers]

        for layer in layers:
            globalpath = os.path.abspath(os.path.join(self.app.basedir, path))
            imagePath = layer["tileSource"].replace(".dzi", "")
            slide = tv._get_slide(globalpath + "/" + imagePath)

            channel = layer["name"]

            if "_" not in tiles.keys():
                tiles["_"] = {}
            tiles["_"][channel] = self.getTile(slide, bbox)
        # print(channels, tiles)
        plot = self.getPlot(tiles, ["_"], channels, bbox)
        img_str = base64.b64encode(plot.getvalue())
        resp = make_response(img_str)
        return resp

    def importFolder(self, jsonParam):
        if not jsonParam:
            logging.error("No arguments, aborting.")
            abort(500)
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
            layer = {"name": basename, "tileSource": filePath + ".dzi"}
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
