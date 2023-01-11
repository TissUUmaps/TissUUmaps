try:
    import glob
    import os
    from io import BytesIO

    import matplotlib
    from flask import abort, make_response
    from PIL import Image

    matplotlib.use("Agg")

    import base64
    import logging
    import re
    from urllib.parse import unquote

    import matplotlib.pyplot as plt
    import numpy as np
    import pyvips

    import tissuumaps.views as tv
except:
    pass


class PILBytesIO(BytesIO):
    def fileno(self):
        """Classic PIL doesn't understand io.UnsupportedOperation."""
        raise AttributeError("Not supported")


class Plugin:
    def __init__(self, app):
        self.app = app

    def getTileRaw(self, image, bbox):
        tile = image.crop(bbox[0], bbox[1], bbox[2], bbox[3])
        return tile

    def getTile(self, path, bbox):
        path = path.replace(".dzi", "")
        if path[0] == "\\" or path[0] == "/":
            path = path[1:]
        if "__p" in path:
            path, page = path.split("__p")
            page = int(page)
        else:
            page = None
        slide = tv._get_slide(path)
        try:
            with slide.tileLock:
                if page is not None:
                    tile = slide.osr.read_region(
                        (bbox[0], bbox[1]), 0, (bbox[2], bbox[3]), page=page
                    )
                else:
                    tile = slide.osr.read_region(
                        (bbox[0], bbox[1]), 0, (bbox[2], bbox[3])
                    )
        except ValueError:
            # Invalid level or coordinates
            logging.error("ValueError, aborting.")
            abort(500, "ValueError, aborting.")
        return tile

    def getConcat(self, outputFields, tiles, use_raw):
        singleWidth = tiles[0]["tile"].width
        singleHeight = tiles[0]["tile"].height

        width = len(outputFields[1]) * singleWidth
        height = len(outputFields[0]) * singleHeight

        if use_raw:
            dst = Image.new("I", (width, height))
        else:
            dst = Image.new("RGB", (width, height))
        for tile in tiles:
            try:
                if use_raw:
                    tile["tile"] = Image.fromarray(tile["tile"].__array__())
                dst.paste(
                    tile["tile"],
                    box=(
                        tile["coord"][1] * singleWidth,
                        tile["coord"][0] * singleHeight,
                    ),
                )
            except:
                pass
        return dst

    def getPlot(self, outputFields, tiles, markers, bbox, use_raw, show_trace):
        plt.style.use(self.style)
        if self.style == "dark_background":
            grid_color = "white"
        else:
            grid_color = "red"
        singleWidth = tiles[0]["tile"].width
        singleHeight = tiles[0]["tile"].height
        im = self.getConcat(outputFields, tiles, use_raw)
        print(im, im.getextrema())

        figureRatio = (im.width + singleWidth / 2) / (im.height + singleHeight / 2)
        if figureRatio > 1:
            figureSize = (self.figureSize, self.figureSize / figureRatio)
        else:
            figureSize = (self.figureSize * figureRatio, self.figureSize)
        fig = plt.figure(dpi=80, figsize=figureSize)
        ax = fig.add_subplot(111)
        if use_raw and self.cmap is None:
            self.cmap = "Greys_r"
        if not use_raw and self.cmap is not None:
            im = im.convert("L")
        try:
            imcolor = plt.imshow(im.__array__(), cmap=plt.get_cmap(self.cmap))
        except:
            imcolor = plt.imshow(im, cmap=plt.get_cmap(self.cmap))
        if self.cmap is not None:
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size=0.2, pad=0.05)
            cax = fig.add_axes(
                [
                    ax.get_position().x1 + 0.1,
                    ax.get_position().y0,
                    0.03 / figureRatio**0.5,
                    ax.get_position().height,
                ]
            )
            plt.colorbar(imcolor, cax=cax)

        for xIndex in range(len(outputFields[1]) + 1):
            ax.axvline(x=singleWidth * xIndex - 0.5, color=grid_color, linewidth=1)
        for yIndex in range(len(outputFields[0]) + 1):
            ax.axhline(y=singleHeight * yIndex - 0.5, color=grid_color, linewidth=1)

        for marker in markers:
            try:
                x, y = [], []
                if self.marker_row in marker.keys():
                    markerRounds = marker[self.marker_row].split(";")
                    # markerRounds = [outputFields[0][int(m)] for m in markerRounds]
                else:
                    markerRounds = outputFields[0]
                if self.marker_col in marker.keys():
                    markerchannels = marker[self.marker_col].split(";")
                    # markerchannels = [outputFields[1][int(m)] for m in markerchannels]
                else:
                    markerchannels = marker["letters"]
                offset = (
                    marker["global_X_pos"] - bbox[0] - 0.5,
                    marker["global_Y_pos"] - bbox[1] - 0.5,
                )
                for yIndex, (markerchannel, markerRound) in enumerate(
                    zip(markerchannels, markerRounds)
                ):
                    xIndex = outputFields[1].index(markerchannel)
                    yIndex_ = outputFields[0].index(markerRound)
                    x.append(offset[0] + singleWidth * xIndex)
                    y.append(offset[1] + singleWidth * yIndex_)
                if show_trace:
                    line_ = "o-"
                    marker_ = "x"
                else:
                    line_ = "o"
                    marker_ = "o"
                ax.plot(
                    x,
                    y,
                    line_,
                    label=markerchannels,
                    color=marker["color"],
                    markersize=5,
                    marker=marker_,
                )
            except:
                import traceback

                logging.error(traceback.format_exc())
                pass

        ax.set_xticks(
            [
                i * singleWidth + singleWidth / 2 - 0.5
                for i, _ in enumerate(outputFields[1])
            ]
        )
        ax.set_xticklabels([c.replace(".tif", "") for c in outputFields[1]])
        ax.set_yticks(
            [
                i * singleHeight + singleHeight / 2 - 0.5
                for i, _ in enumerate(outputFields[0])
            ]
        )
        ax.set_yticklabels(outputFields[0], rotation=90, va="center")
        ax.tick_params(axis="both", which="both", length=0)
        ax.grid(False)
        ax.minorticks_off()
        plt.tight_layout()
        buf = PILBytesIO()

        fig.savefig(buf, bbox_inches="tight")
        fig.clf()
        plt.close()
        return buf

    def getMatrix(self, jsonParam):
        try:
            if not jsonParam:
                logging.error("No arguments, aborting.")
                abort(500, "No arguments, aborting.")

            bbox = jsonParam["bbox"]
            show_trace = jsonParam["show_trace"]
            layers = jsonParam["layers"]
            path = jsonParam["path"]
            markers = jsonParam["markers"]
            use_raw = jsonParam["use_raw"]
            self.marker_row = jsonParam["marker_row"]
            self.marker_col = jsonParam["marker_col"]
            self.figureSize = jsonParam["figureSize"]
            self.style = "dark_background"
            if "cmap" in jsonParam.keys():
                self.cmap = jsonParam["cmap"]
            else:
                self.cmap = None
            if self.cmap == "None":
                self.cmap = None
            layer_format = jsonParam["layer_format"]
            if "{row}" not in layer_format:
                layer_format = "{row}" + layer_format
            if "{col}" not in layer_format:
                layer_format = layer_format + "{col}"
            invert_row_col = layer_format.index("{row}") > layer_format.index("{col}")
            regexp_format = re.escape(layer_format)
            for fieldName in ["\{row\}", "\{col\}"]:
                regexp_format = regexp_format.replace(fieldName, "(.*)")
            # Get all raws and cols sorted
            outputFields = None
            kept_layers = []
            for layer in layers:
                tileCoord = re.match(regexp_format, layer["name"])
                if tileCoord is None:
                    continue
                kept_layers.append(layer)
                if invert_row_col:
                    tileCoord = list(reversed(tileCoord.groups()))
                else:
                    tileCoord = list(tileCoord.groups())
                if outputFields is None:
                    outputFields = [[x] for x in tileCoord]
                outputFields = [x + [y] for x, y in zip(outputFields, tileCoord)]
            layers = kept_layers

            # Remove duplicates:
            outputFields = [list(dict.fromkeys(fields)) for fields in outputFields]
            tiles = []
            for layer in sorted(
                layers, key=lambda x: list(re.match(regexp_format, x["name"]).groups())
            ):
                globalpath = os.path.abspath(os.path.join(self.app.basedir, path))
                # Get coordinate index of tile in output
                tileCoord = re.match(regexp_format, layer["name"])
                if invert_row_col:
                    tileCoord = list(reversed(tileCoord.groups()))
                else:
                    tileCoord = list(tileCoord.groups())

                tileCoord = [
                    output.index(x) for output, x in zip(outputFields, tileCoord)
                ]
                try:
                    if use_raw:
                        image = pyvips.Image.new_from_file(
                            globalpath + "/" + layer["tileSource"].replace(".dzi", ""),
                            memory=False,
                            access="random",
                        )
                        tile = self.getTileRaw(image, bbox)
                    else:
                        tile = self.getTile(path + "/" + layer["tileSource"], bbox)
                    tiles.append({"coord": tileCoord, "tile": tile})
                except:
                    pass
            plot = self.getPlot(outputFields, tiles, markers, bbox, use_raw, show_trace)

            img_str = base64.b64encode(plot.getvalue())
            resp = make_response(img_str)
            return resp
        except:
            import traceback

            resp = abort(500, traceback.format_exc())
            return resp

    def importFolder(self, jsonParam):
        if not jsonParam:
            logging.error("No arguments, aborting.")
            abort(500, "No arguments, aborting.")
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
