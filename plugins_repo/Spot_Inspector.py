import glob
import os
from io import BytesIO

import matplotlib
from flask import abort, make_response
from PIL import Image

matplotlib.use("Agg")

import base64
import logging
from urllib.parse import unquote

import matplotlib.pyplot as plt
import pyvips

import tissuumaps.views as tv

class PILBytesIO(BytesIO):
    def fileno(self):
        """Classic PIL doesn't understand io.UnsupportedOperation."""
        raise AttributeError("Not supported")


class Plugin:
    def __init__(self, app):
        self.app = app

    def getTile(self, image, bbox):
        tile = image.crop(bbox[0], bbox[1], bbox[2], bbox[3])
        return tile

    def getConcat(self, tiles, rounds, channels):
        tilesArray = []
        for row, round in enumerate(rounds):
            for col, channel in enumerate(channels):
                tilesArray.append(tiles[round][channel])
        return pyvips.Image.arrayjoin(tilesArray, across=len(channels))

    def getPlot(self, tiles, rounds, channels, markers, bbox):
        singleWidth = tiles[rounds[0]][channels[0]].width
        singleHeight = tiles[rounds[0]][channels[0]].height
        im = self.getConcat(tiles, rounds, channels)  # .convert("L")
        figureRatio = (len(channels) + 2) / len(rounds)
        fig = plt.figure(
            figsize=(self.figureSize * figureRatio, self.figureSize), dpi=80
        )
        ax = fig.add_subplot(111)
        imcolor = plt.imshow(im, cmap=plt.get_cmap(self.cmap))

        plt.colorbar(imcolor, fraction=0.036, pad=0.05)

        for xIndex in range(len(channels) + 1):
            ax.axvline(x=singleWidth * xIndex - 0.5, color="red", linewidth=1)
        for yIndex in range(len(rounds) + 1):
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

                offset = (
                    marker["global_X_pos"] - bbox[0] - 0.5,
                    marker["global_Y_pos"] - bbox[1] - 0.5,
                )
                for yIndex, (markerchannel, markerRound) in enumerate(
                    zip(markerchannels, markerRounds)
                ):
                    xIndex = channels.index(markerchannel)
                    yIndex_ = rounds.index(markerRound)
                    x.append(offset[0] + singleWidth * xIndex)
                    y.append(offset[1] + singleWidth * yIndex_)
                ax.plot(
                    x,
                    y,
                    "o-",
                    label=markerchannels,
                    color=marker["color"],
                    markersize=5,
                    marker="x",
                )
            except:
                import traceback

                logging.error(traceback.format_exc())
                pass

        ax.set_xticks(
            [i * singleWidth + singleWidth / 2 - 0.5 for i, _ in enumerate(channels)]
        )
        ax.set_xticklabels([c.replace(".tif", "") for c in channels])
        ax.set_yticks(
            [i * singleHeight + singleHeight / 2 - 0.5 for i, _ in enumerate(rounds)]
        )
        ax.set_yticklabels(rounds, rotation=90, va="center")
        ax.tick_params(axis="both", which="both", length=0)
        plt.tight_layout()

        buf = PILBytesIO()
        fig.savefig(buf)
        fig.clf()
        plt.close()
        return buf

    def getMatrix(self, jsonParam):
        if not jsonParam:
            logging.error("No arguments, aborting.")
            abort(500)
        print(jsonParam)
        bbox = jsonParam["bbox"]
        layers = jsonParam["layers"]
        path = jsonParam["path"]
        markers = jsonParam["markers"]

        self.figureSize = jsonParam["figureSize"]
        if "cmap" in jsonParam.keys():
            self.cmap = jsonParam["cmap"]
        else:
            self.cmap = "Greys_r"
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
            globalpath = os.path.abspath(os.path.join(self.app.basedir, path))
            image = pyvips.Image.new_from_file(
                globalpath + "/" + layer["tileSource"].replace(".dzi", ""),
                memory=False,
                access="sequential",
            )
            round, channel = layer["name"].split("_")
            if round not in tiles.keys():
                tiles[round] = {}
            tiles[round][channel] = self.getTile(image, bbox)

        plot = self.getPlot(tiles, rounds, channels, markers, bbox)
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
