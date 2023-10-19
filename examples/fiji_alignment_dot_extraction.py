##
## Automatic alignment for staining and destaining
## Christophe Avenel - 2021
###

# Script parameters:
# @ String (visibility=MESSAGE, value="<html><b>The input folder should contain one subfolder per round, and one reference image in each subfolder.</b></html>") docmsg

# @ File (label="Select input folder", style="directory") inputFolder
# @ File (label="Select output folder", style="directory") outputFolder
# @ String (label="All reference images contain the string", value="DAPI", description="Images containing this string will be used as reference to align other images.") stringFilter
# @ Boolean (label="Use interpolation in transformation", value="True") useInterpolate
# @ Boolean (label="Use shrinking constraint", value="False") useShrinking

# @ String (visibility=MESSAGE, value="<html><b>Define spot detection parameters:</b></html>") docmsg2

# @ Float (label="Gaussian filter size", value="0.6") gaussianVal
# @ Float (label="Top hat filter size", value="4") tophatSize
# @ Float (label="Trackmate radius size", value="2") trackmateSize
# @ Float (label="Trackmate threshold", value="10") trackmateThreashold

# @ String (visibility=MESSAGE, value="<html><b>Define output parameters:</b></html>") docmsg3

# @ Integer (label="Signal quality threshold", value="30") qualityThreshold


import csv

# Python imports
import glob
import json
import os
import urllib
from shutil import copyfile

# ImageJ imports
from ij import IJ, ImagePlus, ImageStack

# Fiji plugins
from register_virtual_stack import Register_Virtual_Stack_MT, Transform_Virtual_Stack_MT

print("Automatic alignment for staining and destaining")
IJ.log("Automatic alignment for staining and destaining")


class alignImages:
    def __init__(
        self, inputFolder, outputFolder, stringFilter, useInterpolate, useShrinking
    ):
        # Initialize folders from script parameters
        self.inputFolder = inputFolder + "/"
        self.outputFolder = outputFolder + "/"

        self.inputDAPIFolder = self.outputFolder + "/DAPI/"
        self.inputFluoFolder = self.outputFolder + "/fluo/"
        self.outputDAPIFolder = self.outputFolder + "/DAPI_registered/"
        self.outputFluoFolder = self.outputFolder + "/fluo_registered/"

        # Create output folders
        for folder in [
            self.inputDAPIFolder,
            self.inputFluoFolder,
            self.outputDAPIFolder,
            self.outputFluoFolder,
        ]:
            if not os.path.isdir(folder):
                os.makedirs(folder)

        self.stringFilter = stringFilter
        self.useInterpolate = useInterpolate
        self.useShrinking = useShrinking

        # Get list of files
        self.getFiles(self.inputFolder)

        # Run alignement
        self.runVirtualAlignment()

    # Look for tif files in the inputFolder, and sepearate DAPI from fluo.
    def getFiles(self, inputFolder):
        # Find reference images
        DAPIFiles = glob.glob(inputFolder + "/*/*" + self.stringFilter + "*.tif")
        # Find other non-reference images
        self.imageFiles = []
        for DAPIFile in DAPIFiles:
            fluoFiles = [
                f
                for f in glob.glob(os.path.dirname(DAPIFile) + "/*.tif")
                if f != DAPIFile
            ]
            self.imageFiles.append((DAPIFile, fluoFiles))

    def runVirtualAlignment(self):
        IJ.log("Function runVirtualAlignment on " + self.inputFolder)
        refImage = None
        for DAPIFile, fluoFiles in self.imageFiles:
            copyfile(DAPIFile, self.inputDAPIFolder + os.path.basename(DAPIFile))
            for fluoFile in fluoFiles:
                copyfile(fluoFile, self.inputFluoFolder + os.path.basename(fluoFile))
            if refImage is None:
                refImage = os.path.basename(
                    self.inputDAPIFolder + os.path.basename(DAPIFile)
                )

        p = Register_Virtual_Stack_MT.Param()
        use_shrinking_constraint = self.useShrinking

        IJ.log("Registering DAPI")
        Register_Virtual_Stack_MT.exec(
            self.inputDAPIFolder,
            self.outputDAPIFolder,
            self.outputDAPIFolder,
            refImage,
            p,
            use_shrinking_constraint,
        )

        IJ.log("Copying DAPI transformation matrices to Fluo folder")
        for DAPIFile, fluoFiles in self.imageFiles:
            for fluoFile in fluoFiles:
                copyfile(
                    self.outputDAPIFolder
                    + os.path.basename(DAPIFile).replace(".tif", ".xml"),
                    self.outputFluoFolder
                    + os.path.basename(fluoFile).replace(".tif", ".xml"),
                )

        use_interpolate = self.useInterpolate
        IJ.log("Registering Fluo based on DAPI transformations")
        Transform_Virtual_Stack_MT.exec(
            self.inputFluoFolder,
            self.outputFluoFolder,
            self.outputFluoFolder,
            use_interpolate,
        )

        self.displayStack()

    def displayStack(self):
        IJ.log("Creating virtual stacks from files")
        FluoStack = None
        for DAPIFile, fluoFiles in self.imageFiles:
            for fluoFile in fluoFiles:
                fluoFile = self.outputFluoFolder + os.path.basename(fluoFile)
                FluoImp = IJ.openImage(fluoFile)
                if not FluoStack:
                    FluoStack = ImageStack(FluoImp.width, FluoImp.height)
                (minimum, maximum) = alignImages.getMinMax(FluoImp)
                FluoImp.setDisplayRange(minimum, maximum)
                IJ.run(FluoImp, "8-bit", "")

                FluoStack.addSlice(FluoImp.getProcessor())
        FluoStackImp = ImagePlus("Fluo stack", FluoStack)
        FluoStackImp.setTitle("Fluorescence alignment (8 bit)")
        FluoStackImp.show()
        IJ.run(FluoStackImp, "Make Composite", "display=Composite")
        IJ.resetMinAndMax(FluoStackImp)

    # function to get min and max values of pixels, while ignoring outliers.
    @staticmethod
    def getMinMax(imp, percMin=0.1, percMax=1.0):
        nPixels = imp.getWidth() * imp.getHeight()

        stats = imp.getStatistics()
        histo = stats.histogram16
        sum, minimum, maximum = 0, None, None
        for i, h in enumerate(histo):
            sum += h
            if minimum is None and sum > nPixels * percMin:
                minimum = i - 1
            if maximum is None and sum >= nPixels * percMax:
                maximum = i
                break
        return minimum, maximum


import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter
import fiji.plugin.trackmate.features.spot.SpotIntensityMultiCAnalyzerFactory as SpotIntensityMultiCAnalyzerFactory
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
from fiji.plugin.trackmate import Logger, Model, SelectionModel, Settings, TrackMate
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.providers import (
    EdgeAnalyzerProvider,
    SpotAnalyzerProvider,
    TrackAnalyzerProvider,
)
from fiji.plugin.trackmate.tracking import LAPUtils
from fiji.plugin.trackmate.tracking.sparselap import SparseLAPTrackerFactory
from ij.plugin import ImageCalculator


class extractDots:
    def __init__(
        self, imageFilename, gaussianVal, tophatSize, trackmateSize, trackmateThreashold
    ):
        self.imageFilename = os.path.basename(imageFilename)
        self.gaussianVal = gaussianVal
        self.tophatSize = tophatSize
        self.trackmateSize = trackmateSize
        self.trackmateThreashold = trackmateThreashold

        self.imp = IJ.openImage(imageFilename)
        self.preprocess()
        spots = self.trackmate()
        self.csvSpots = self.spotsToCSV(spots)

    def preprocess(self):
        IJ.run(self.imp, "Gaussian Blur...", "sigma=" + str(gaussianVal))
        IJ.run(self.imp, "8-bit", "")
        imp_tmp = self.imp.duplicate()
        IJ.run(
            self.imp,
            "Gray Morphology",
            "radius=" + str(tophatSize) + " type=circle operator=open",
        )
        ImageCalculator().run("Subtract", imp_tmp, self.imp)
        self.imp.changes = False
        self.imp.close()
        self.imp = imp_tmp

    def trackmate(self):
        calibration = self.imp.getCalibration()
        model = Model()
        model.setLogger(Logger.IJ_LOGGER)
        settings = Settings()
        settings.setFrom(self.imp)
        # Configure detector - We use the Strings for the keys
        settings.detectorFactory = LogDetectorFactory()
        settings.detectorSettings = {
            "DO_SUBPIXEL_LOCALIZATION": True,
            "RADIUS": calibration.getX(self.trackmateSize),
            "TARGET_CHANNEL": 1,
            "THRESHOLD": self.trackmateThreashold,
            "DO_MEDIAN_FILTERING": True,
        }

        # Configure spot filters - Classical filter on quality
        filter1 = FeatureFilter("QUALITY", 0.01, True)
        settings.addSpotFilter(filter1)
        settings.addSpotAnalyzerFactory(SpotIntensityMultiCAnalyzerFactory())

        settings.initialSpotFilterValue = 1

        # Configure tracker - We want to allow merges and fusions
        settings.trackerFactory = SparseLAPTrackerFactory()
        settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap()

        trackmate = TrackMate(model, settings)

        # --------
        # Process
        # --------

        ok = trackmate.checkInput()
        if not ok:
            print("NOT OK")

        ok = trackmate.process()
        if not ok:
            print("NOT OK")

        # ----------------
        # Display results
        # ----------------

        # selectionModel = SelectionModel(model)
        # displayer =  HyperStackDisplayer(model, selectionModel, self.imp)
        # displayer.render()
        # displayer.refresh()

        # Echo results with the logger we set at start:
        spots = model.getSpots()
        return spots.iterable(True)

    def floatToHexa(self, quality, qmin, qmax):
        hexa = int(255 - min(255, 255 * (float(quality) - qmin) / qmax))
        hexaString = "%02x" % hexa
        return "#ff" + hexaString + hexaString

    def spotsToCSV(self, spots):
        spotsDict = []
        calibration = self.imp.getCalibration()
        for spot in spots:
            q = spot.getFeature("QUALITY")  # Stored the ROI id.
            # Fetch spot features directly from spot.
            x = spot.getFeature("POSITION_X")
            y = spot.getFeature("POSITION_Y")
            q = spot.getFeature("QUALITY")
            mean = spot.getFeature("MEAN_INTENSITY")
            spotsDict.append(
                {
                    "x": calibration.getRawX(x) + 0.5,
                    "y": calibration.getRawY(y) + 0.5,
                    "q": q,
                    "mean": mean,
                    "name": self.imageFilename,
                }
            )
        qmin, qmax = (
            min([s["q"] for s in spotsDict]),
            max([s["q"] for s in spotsDict]) / 3.0,
        )
        for spot in spotsDict:
            spot["color"] = self.floatToHexa(spot["q"], qmin, qmax)
        return spotsDict

    def updateBoundingBox(self, boundingBox, transFile):
        corners = [
            (0, 0),
            (0, self.imp.getHeight()),
            (self.imp.getWidth(), self.imp.getHeight()),
            (self.imp.getWidth(), 0),
        ]
        transCoord = Transform_Virtual_Stack_MT.readCoordinateTransform(transFile)
        for x, y in corners:
            x, y = transCoord.apply([x, y])
            if x < boundingBox[0]:
                boundingBox[0] = x
            if x > boundingBox[1]:
                boundingBox[1] = x
            if y < boundingBox[2]:
                boundingBox[2] = y
            if y > boundingBox[3]:
                boundingBox[3] = y
        return boundingBox

    def transformSpots(self, spotsDict, boundingBox, transFile):
        newDict = [d.copy() for d in spotsDict]
        transCoord = Transform_Virtual_Stack_MT.readCoordinateTransform(transFile)
        for spot in newDict:
            spot["x"], spot["y"] = transCoord.apply([spot["x"], spot["y"]])
            spot["x"], spot["y"] = (
                spot["x"] - boundingBox[0],
                spot["y"] - boundingBox[2],
            )
        return newDict


class TissUUmaps:
    def __init__(self, layers, markerFile, outputFile):
        self.layers = [
            {"name": os.path.basename(layer), "tileSource": layer} for layer in layers
        ]
        self.markerFile = markerFile
        self.outputFile = outputFile
        jsonProj = self.getJsonProject()
        with open(self.outputFile, "w") as outFile:
            json.dump(jsonProj, outFile)

    def getJsonProject(self):
        return {
            "markerFiles": [
                {
                    "path": self.markerFile[1],
                    "title": "Download markers",
                    "comment": "Only markers with q>{q}".format(q=qualityThreshold),
                    "expectedCSV": {
                        "group": "name",
                        "name": "",
                        "X_col": "x",
                        "Y_col": "y",
                        "key": "letters",
                        "color": "color",
                    },
                    "autoLoad": True,
                }
            ],
            "filters": ["Contrast", "Color"],
            "compositeMode": "lighter",
            "layers": self.layers,
            "filename": "",
            "settings": [
                {
                    "module": "overlayUtils",
                    "function": "_linkMarkersToChannels",
                    "value": True,
                },
                {"module": "dataUtils", "function": "_autoLoadCSV", "value": True},
            ],
        }


ai = alignImages(
    unicode(inputFolder),
    unicode(outputFolder),
    stringFilter,
    useInterpolate,
    useShrinking,
)
images = glob.glob(unicode(outputFolder) + "/fluo/*.tif")
allSpotsDict = []
allSpotsTransDict = []
boundingBox = [0, 0, 0, 0]
eD = {}
for imageFilename in images:
    IJ.log("Extracting spots " + imageFilename)
    transFile = imageFilename.replace("fluo", "fluo_registered").replace(".tif", ".xml")
    eD[imageFilename] = extractDots(
        imageFilename, gaussianVal, tophatSize, trackmateSize, trackmateThreashold
    )
    allSpotsDict += eD[imageFilename].csvSpots
    boundingBox = eD[imageFilename].updateBoundingBox(boundingBox, transFile)
    print(boundingBox)

for imageFilename in images:
    IJ.log("Transforming spots " + imageFilename)
    transFile = imageFilename.replace("fluo", "fluo_registered").replace(".tif", ".xml")
    transDict = eD[imageFilename].transformSpots(
        eD[imageFilename].csvSpots, boundingBox, transFile
    )
    allSpotsTransDict += transDict
    del eD[imageFilename]

print(len(allSpotsDict), len(allSpotsTransDict))

with open(unicode(outputFolder) + "/markers.csv", "wb") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=allSpotsDict[0].keys())
    writer.writeheader()
    for data in allSpotsDict:
        writer.writerow(data)

with open(unicode(outputFolder) + "/markers_reg.csv", "wb") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=allSpotsTransDict[0].keys())
    writer.writeheader()
    for data in allSpotsTransDict:
        writer.writerow(data)

with open(
    unicode(outputFolder) + "/markers_{q}.csv".format(q=qualityThreshold), "wb"
) as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=allSpotsDict[0].keys())
    writer.writeheader()
    for data in allSpotsDict:
        if data["q"] > qualityThreshold:
            writer.writerow(data)

with open(
    unicode(outputFolder) + "/markers_reg_{q}.csv".format(q=qualityThreshold), "wb"
) as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=allSpotsTransDict[0].keys())
    writer.writeheader()
    for data in allSpotsTransDict:
        if data["q"] > qualityThreshold:
            writer.writerow(data)

images = glob.glob(unicode(outputFolder) + "/DAPI/*.tif") + glob.glob(
    unicode(outputFolder) + "/fluo/*.tif"
)
layers = [
    os.path.basename(os.path.dirname(i)) + "/" + os.path.basename(i) + ".dzi"
    for i in images
]
TissUUmaps(
    layers,
    ["markers.csv", "markers_{q}.csv".format(q=qualityThreshold)],
    unicode(outputFolder) + "/NonRegistered.tmap",
)
images = glob.glob(unicode(outputFolder) + "/DAPI_registered/*.tif") + glob.glob(
    unicode(outputFolder) + "/fluo_registered/*.tif"
)
layers = [
    os.path.basename(os.path.dirname(i)) + "/" + os.path.basename(i) + ".dzi"
    for i in images
]
TissUUmaps(
    layers,
    ["markers_reg.csv", "markers_reg_{q}.csv".format(q=qualityThreshold)],
    unicode(outputFolder) + "/Registered.tmap",
)

print("DONE")
