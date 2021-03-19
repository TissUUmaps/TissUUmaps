/**
 * @file overlayUtils.js Interface to ask information to OSD
 * @author Leslie Solorzano
 * @see {@link overlayUtils}
 */

/**
 * @namespace overlayUtils
 * @property {Bool}   overlayUtils._drawRegions - If false then disable the drawing of regions
 * @property {Object} overlayUtils._d3nodes - Main group or container of all d3 svg groups of overlays corresponding to the 3 main marker data groups
 * @property {Number} overlayUtils._percentageForSubsample - Take this percentage of each barcode when downsamplir for lower resolutions
 * @property {Number}  overlayUtils._zoomForSubsample - When the zoom is bigger than this, display all the checked genes 
 */
overlayUtils = {
    _drawRegions: false,
    _d3nodes: {},
    _percentageForSubsample: 0.25,
    _zoomForSubsample:5.15
}

/**
 * This method is used to add all layers from tmapp */
overlayUtils.addAllLayers = function() {
    tmapp.layers.forEach(function(layer, i) {
        overlayUtils.addLayer(layer.name, layer.tileSource, i);
    });
}

/**
 * This method is used to add a layer */
overlayUtils.addLayer = function(layerName, tileSource, i) {
    var op = tmapp["object_prefix"];
    var vname = op + "_viewer";

    tmapp[vname].addTiledImage({
        index: i + 1,
        tileSource: tmapp._url_suffix + tileSource,
        opacity: 1.0
    });
    HTMLElementUtils.addLayerSettings(layerName, i);
}


/** 
 * @param {Number} item Index of an OSD tile source
 * Set the opacity of a tile source */
overlayUtils.setItemOpacity = function(item, opacity) {
    var op = tmapp["object_prefix"];
    if (!tmapp[op + "_viewer"].world.getItemAt(item)) {
        setTimeout(function() {
            overlayUtils.setItemOpacity(item, opacity);
        }, 100);
        return;
    }
    tmapp[op + "_viewer"].world.getItemAt(item).setOpacity(opacity);
}

overlayUtils.areAllFullyLoaded = function () {
    var tiledImage;
    var op = tmapp["object_prefix"];
    var count = tmapp[op + "_viewer"].world.getItemCount();
    for (var i = 0; i < count; i++) {
      tiledImage = tmapp[op + "_viewer"].world.getItemAt(i);
      if (!tiledImage.getFullyLoaded() && tiledImage.getOpacity() != 0) {
        return false;
      }
    }
    return true;
  }

/** 
 * @param {String} layerName name of an existing d3 node
 * @param {Number} opacity desired opacity
 * Set the opacity of a tile source */
overlayUtils.setLayerOpacity= function(layerName,opacity){
    if(layerName in overlayUtils._d3nodes){
        var layer = overlayUtils._d3nodes[layerName];
        layer._groups[0][0].style.opacity=opacity;    
    }else{
        console.log("layer does not exist or is not a D3 node");
    }
}

/**
 * @param {String} colortype A string from [hex,hsl,rgb]
 * Get a random color in the desired format
 */
overlayUtils.randomColor = function (colortype) {
    if (!colortype) {
        colortype = "hex";
    }
    //I need random colors that are far away from the palette in the image
    //in this case Hematoxilyn and DAB so far away from brown and light blue
    //and avoid light colors because of the white  background 
    //in HSL color space this means L from 0.2 to 0.75
    //H [60,190],[220,360], S[0.3, 1.0]
    var rh1 = Math.floor(Math.random() * (190 - 60 + 1)) + 60;
    var rh2 = Math.floor(Math.random() * (360 - 220 + 1)) + 220;
    var H = 0.0;

    if (Math.random() > 0.5) { H = rh1; } else { H = rh2; }

    var L = Math.floor(Math.random() * (75 - 20 + 1)) + 20 + '%';
    var S = Math.floor(Math.random() * (100 - 40 + 1)) + 40 + '%';

    var hslstring = 'hsl(' + H.toString() + ',' + S.toString() + ',' + L.toString() + ')';

    var d3color = d3.hsl(hslstring);
    if (colortype == "hsl") return hslstring;
    if (colortype == "rgb") {
        return d3color.rgb().toString();
    }
    if (colortype == "hex") {
        var hex = function (value) {
            value = Math.max(0, Math.min(255, Math.round(value) || 0));
            return (value < 16 ? "0" : "") + value.toString(16);
        }
        var rgbcolor = d3color.rgb();
        return "#" + hex(rgbcolor.r) + hex(rgbcolor.g) + hex(rgbcolor.b);
    }
}

/**
 * Main function to update the view if there has been a reason for it. 
 * It computes all the elements that have to be drawn.
 */
overlayUtils.modifyDisplayIfAny = function () {
    //get four corners of view
    var op = tmapp["object_prefix"];
    var bounds = tmapp[op + "_viewer"].viewport.getBounds();
    var currentZoom = tmapp[op + "_viewer"].viewport.getZoom();

    var xmin, xmax, ymin, ymax;
    xmin = bounds.x; ymin = bounds.y;
    xmax = xmin + bounds.width; ymax = ymin + bounds.height;

    var imageWidth = OSDViewerUtils.getImageWidth();
    var imageHeight = OSDViewerUtils.getImageHeight();

    if (xmin < 0) { xmin = 0; }; if (xmax > 1.0) { xmax = 1.0; };
    if (ymin < 0) { ymin = 0; }; if (ymax > imageHeight / imageWidth) { ymax = imageHeight / imageWidth; };
    
    var total = imageWidth * imageHeight;

    //convert to global image coords
    xmin *= imageWidth; xmax *= imageWidth; ymin *= imageWidth; ymax *= imageWidth;

    var portion = (xmax - xmin) * (ymax - ymin);
    var percentage = portion / total;

    //get barcodes that are checked to draw
    for (var barcode in markerUtils._checkBoxes) {
        if (tmapp["hideSVGMarkers"]) continue;  // We are using WebGL instead for the drawing

        if (markerUtils._checkBoxes[barcode].checked) {
            var markersInViewportBounds = []
            if (percentage < overlayUtils._percentageForSubsample) {
                //console.log("percentage less than " + overlayUtils._percentageForSubsample);
                markersInViewportBounds = markerUtils.arrayOfMarkersInBox(
                    dataUtils[op + "_barcodeGarden"][barcode], xmin, ymin, xmax, ymax, { globalCoords: true }
                );

                //console.log(markersInViewportBounds.length);
                var drawThese = dataUtils.randomSamplesFromList(markerUtils.startCullingAt, markersInViewportBounds);

                //console.log(drawThese.length);
                markerUtils.drawAllFromList(drawThese);

            } else {
                //if the percentage of image I see is bigger than a threshold then use the predownsampled markers
                if (dataUtils._subsampledBarcodes[barcode]) {
                    markerUtils.drawAllFromList(dataUtils._subsampledBarcodes[barcode]);
                } else {
                    markerUtils.drawAllFromBarcode(barcode);
                }
            }
        } 
    }

    //if anything from cpdata exists
    var cpop="CP";
    if(CPDataUtils.hasOwnProperty(cpop+"_rawdata")){ //I need to put a button here to draw or remove
        //Zoom of 1 means all image is visible so low res. Bigger than 1 means zooming in.
        if (currentZoom > overlayUtils._zoomForSubsample) {            
            markerUtils.drawCPdata({searchInTree:true,xmin:xmin, xmax:xmax, ymin:ymin, ymax:ymax});
        } else {
            console.log("subsample");
            //I create the subsampled one already when I read de CP csv, in CPDataUtils[cpop + "_subsampled_data"]     
            markerUtils.drawCPdata({searchInTree:false});            
        }
    }
}

/**
 * Save the current SVG overlay to open in a vector graphics editor for a figure for example
 */
overlayUtils.saveSVG=function(){
    var svg = d3.select("svg");
    var svgData = svg._groups[0][0].outerHTML;
    var svgBlob = new Blob([svgData], {type:"image/svg+xml;charset=utf-8"});
    var svgUrl = URL.createObjectURL(svgBlob);
    var downloadLink = document.createElement("a");
    downloadLink.href = svgUrl;
    downloadLink.download = "currentview.svg";
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink); 
}
