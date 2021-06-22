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
    _zoomForSubsample:5.15,
    _layerOpacities:{},
    _linkMarkersToChannels:false
}

/**
 * This method is used to add all layers from tmapp */
overlayUtils.addAllLayers = function() {
    /* For backward compatibility with tmapp.fixed_file, but converted to a layer */
    if (tmapp.fixed_file && tmapp.fixed_file != "") {
        tmapp.layers.unshift({"name":tmapp.slideFilename, "tileSource":tmapp.fixed_file})
        overlayUtils.addLayer(tmapp.slideFilename, tmapp._url_suffix +  tmapp.fixed_file, -1)
        tmapp.fixed_file = "";
    }
    tmapp.layers.forEach(function(layer, i) {
        overlayUtils.addLayer(layer.name, layer.tileSource, i-1);
    });
    overlayUtils.addAllLayersSettings();
}

/**
 * This method is used to add all layer settings */
overlayUtils.addAllLayersSettings = function() {
    var settingsPanel = document.getElementById("image-overlay-panel");
    settingsPanel.innerHTML = "";
    tmapp.layers.forEach(function(layer, i) {
        overlayUtils.addLayerSettings(layer.name, layer.tileSource, i-1);
    });
    filterUtils.setRangesFromFilterItems();
}

/**
 * This method is used to add a layer */
overlayUtils.addLayerSettings = function(layerName, tileSource, layerIndex) {
    var settingsPanel = document.getElementById("image-overlay-panel");
    var layerTable = document.getElementById("image-overlay-tbody");
    if (!layerTable) {
        layerTable = document.createElement("table");
        layerTable.id = "image-overlay-table";
        layerTable.className += "table table-striped"
        layerTable.style.marginBottom = "0px";
        filterHeaders = "";
        for (filterIndex = 0; filterIndex < filterUtils._filtersUsed.length; filterIndex++) {
            filterHeaders += "<th style='text-align:center;'>" + filterUtils._filtersUsed[filterIndex] + "</th>";
        }
        layerTable.innerHTML = "<thead><th style='text-align:center;'>Name</th><th style='text-align:center;'>Visible</th><th style='text-align:center;'>Opacity</th>" + filterHeaders + "</thead><tbody id='image-overlay-tbody'></tbody>"
        settingsPanel.appendChild(layerTable);
    }
    layerTable = document.getElementById("image-overlay-tbody");
    var tr = document.createElement("tr");

    var visible = document.createElement("input");
    visible.type = "checkbox";
    if (layerIndex < 0)
        visible.checked = true; 
    visible.id = "visible-layer-" + (layerIndex + 1);
    visible.className = "visible-layers";
    visible.setAttribute("layer", (layerIndex + 1));
    var td_visible = document.createElement("td");
    td_visible.appendChild(visible);
    td_visible.style.textAlign = "center";
    td_visible.style.padding = "6px";
    td_visible.style.minWidth = "100px";

    var opacity = document.createElement("input");
    opacity.type = "range";
    opacity.setAttribute("min", "0");
    opacity.setAttribute("max", "1");
    opacity.setAttribute("step", "0.1");
    opacity.setAttribute("layer", (layerIndex + 1));
    opacity.id = "opacity-layer-" + (layerIndex + 1);
    var td_opacity = document.createElement("td");
    td_opacity.appendChild(opacity);
    td_opacity.style.textAlign = "center";
    td_opacity.style.padding = "6px";
    td_opacity.style.minWidth = "100px";
    tileSource = tileSource.replace(/\\/g, '\\\\');
    tr.innerHTML = "<td style='padding:6px;'>" + layerName + "</td>";
    tr.appendChild(td_visible);
    tr.appendChild(td_opacity);

    for (filterIndex = 0; filterIndex < filterUtils._filtersUsed.length; filterIndex++) {
        filterName = filterUtils._filtersUsed[filterIndex];
        filterParams = filterUtils.getFilterParams(filterName)
        filterParams.layer = layerIndex + 1;
        
        filterInput = filterUtils.createHTMLFilter(filterParams);
        var td_filterInput = document.createElement("td");
        td_filterInput.style.textAlign = "center";
        td_filterInput.style.padding = "6px";
        td_filterInput.style.minWidth = "100px";
        td_filterInput.appendChild(filterInput);
        
        tr.appendChild(td_filterInput);
    }
    layerTable.prepend(tr);

    visible.addEventListener("change", function(ev) {
        var layer = ev.srcElement.getAttribute("layer")
        var slider = document.querySelectorAll('[layer="' + layer + '"][type="range"]')[0];
        var checkbox = ev.srcElement;
        if (checkbox.checked) {
            overlayUtils._layerOpacities[layer] = slider.value;
        } else {
            overlayUtils._layerOpacities[layer] = 0;
        }
        overlayUtils.setItemOpacity(layer);
    });
    opacity.addEventListener("input", function(ev) {
        var layer = ev.srcElement.getAttribute("layer")
        var slider = ev.srcElement;
        var checkbox = document.querySelectorAll('[layer="' + layer + '"][type="checkbox"]')[0];
        if (checkbox.checked) {
            overlayUtils._layerOpacities[layer] = slider.value;
        } else {
            overlayUtils._layerOpacities[layer] = 0;
        }
        overlayUtils.setItemOpacity(layer);
    });
    overlayUtils.addLayerSlider();
}

/**
 * This method is used to add a layer */
 overlayUtils.addLayerSlider = function() {
    if (document.getElementById("channelRangeInput") == undefined) {
        var elt = document.createElement('div');
        elt.className = "channelRange"
        elt.id = "channelRangeDiv"
        elt.style.zIndex = "100";
        elt.style.paddingLeft = "5px";
        elt.style.paddingBottom = "2px";
        var span = document.createElement('div');
        span.innerHTML = "Channel 1"
        span.id = "channelValue"
        span.style.maxWidth="200px";
        span.style.overflow="hidden";
        var channelRange = document.createElement("input");
        channelRange.type = "range";
        channelRange.style.width = "200px";
        channelRange.id = "channelRangeInput";
        elt.appendChild(span);
        elt.appendChild(channelRange);
        changeFun = function(ev) {
            var slider = $(channelRange)[0];
            channel = slider.value;
            $(".visible-layers").prop("checked",true);$(".visible-layers").click();$("#visible-layer-"+(channel- -1)).click();
            channelName = tmapp.layers[channel- -1].name;
            channelName = channelName.split('.').slice(0, -1).join('.')
            document.getElementById("channelValue").innerHTML = "Channel " + (channel - -2) + ": " + channelName;
            if (overlayUtils._linkMarkersToChannels) {
                $("#ISS_table input[type=checkbox]").prop("checked",false);
                if (document.getElementById(channelName+"-checkbox-ISS")) {
                    $(document.getElementById(channelName +"-checkbox-ISS")).click();
                }
                else {
                    $("#AllMarkers-checkbox-ISS").click().click();
                }
            }
        };
        channelRange.addEventListener("input", changeFun);

        var mousewheelevt = (/Firefox/i.test(navigator.userAgent)) ? "DOMMouseScroll" : "mousewheel";
        $(elt).bind(mousewheelevt, moveSlider);
        function moveSlider(e){
            var zoomLevel = parseInt($(channelRange).val()); 
            console.log("Mouse wheel!", zoomLevel, zoomLevel+1, zoomLevel-1)
            console.log(e.originalEvent.wheelDelta);
            // detect positive or negative scrolling
            if ( e.originalEvent.wheelDelta < 0 ) {
                //scroll down
                $(channelRange).val(zoomLevel+1);
            } else {
                //scroll up
                $(channelRange).val(zoomLevel-1);
            }
            
            // trigger the change event
            changeFun(e.originalEvent);
            
            //prevent page fom scrolling
            return false;
        }
        tmapp['ISS_viewer'].addControl(elt,{anchor: OpenSeadragon.ControlAnchor.BOTTOM_LEFT});
    }
    channelRange = document.getElementById("channelRangeInput");
    var op = tmapp["object_prefix"];
    var nLayers = tmapp.layers.length;
    if (tmapp.fixed_file && tmapp.fixed_file != "") {
        nLayers += 1
    }
    channelRange.setAttribute("min", -1);
    channelRange.setAttribute("max", nLayers - 2);
    channelRange.setAttribute("step", "1");
    channelRange.setAttribute("value", "-1");
    if (nLayers <= 1) {
        document.getElementById("channelRangeDiv").style.display = "none";
    }
    else {
        document.getElementById("channelRangeDiv").style.display = "table";
    }
}

/**
 * This method is used to add a layer from select input */
overlayUtils.addLayerFromSelect = function() {
    var e = document.getElementById("layerSelect");
    var layerName = e.options[e.selectedIndex].text;
    var tileSource = e.options[e.selectedIndex].value;
    tmapp.layers.push({
        name: layerName,
        tileSource: tileSource
    });
    console.log("tileSource", tileSource);
    i = tmapp.layers.length - 1;
    overlayUtils.addLayer(layerName, tileSource, i);
    overlayUtils.addAllLayersSettings();
}

/**
 * This method is used to add a layer */
overlayUtils.addLayer = function(layerName, tileSource, i) {
    var op = tmapp["object_prefix"];
    var vname = op + "_viewer";
    var opacity = 1.0;
    if (i >= 0) {
        opacity = 0.0;
    }
    tmapp[vname].addTiledImage({
        index: i + 1,
        tileSource: tmapp._url_suffix + tileSource,
        opacity: opacity,
        success: function(i) {
            layer0X = tmapp[op + "_viewer"].world.getItemAt(0).getContentSize().x;
            layerNX = tmapp[op + "_viewer"].world.getItemAt(tmapp[op + "_viewer"].world.getItemCount()-1).getContentSize().x;
            tmapp[op + "_viewer"].world.getItemAt(tmapp[op + "_viewer"].world.getItemCount()-1).setWidth(layerNX/layer0X);
        } 
    });
}
 

/** 
 * @param {Number} item Index of an OSD tile source
 * Set the opacity of a tile source */
overlayUtils.setItemOpacity = function(item) {
    opacity = overlayUtils._layerOpacities[item];

    var op = tmapp["object_prefix"];
    if (!tmapp[op + "_viewer"].world.getItemAt(item)) {
        setTimeout(function() {
            overlayUtils.setItemOpacity(item);
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
