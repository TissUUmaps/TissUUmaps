/**
 * @namespace markerUtils
 * @classdesc Work with anything that has to do with markers, take options from the interface
  about markers, and create markers */
markerUtils = {
    /*  drawSymbol=function(overlay,type,x,y,size,barcode) */
    //type must be like d3.symbolVoss
    _drawPaths: true,
    _globalMarkerSize: 1,
    _startCullingAt: 9000,
    _checkBoxes: {},
    _d3Symbols: [d3.symbolCross, d3.symbolDiamond, d3.symbolSquare, d3.symbolTriangle, d3.symbolStar, d3.symbolWye],
    _d3SymbolStrings: ["Cross", "Diamond", "Square", "Triangle", "Star", "Wye"]

}

/** 
 * Draw a D3 symbol
 * @param {string} group d3 group where to put this marker at.
 * @param {sring} type [symbolCross, symbolDiamond, symbolSquare, symbolTriangle, symbolStar, symbolWye]
 * @param {Number} x x coordinate normalized by the width of the image
 * @param {Number} y y coordinate normalized by the width of the image... YES it is the width as well
 * @param {Number} size size relative to the normalized image width (width=1). Common size can be 0.005
 * @param {string} color css formated color string hsl, rgb, hex etc
 * @param {string} barcode string of the key either barcode or name ,exmaple AGGC or "GeneName"
 * @param {Number} globalx pixel location coord x
 * @param {Number} globaly pixel location coord y
 *   */
markerUtils.drawSymbol = function (group, type, x, y, size, color, barcode, globalx, globaly) {
    var marker = overlayUtils._d3nodes[group].append("path")
        .style("fill", color).attr('transform', 'translate(' + x + ',' + y + ')').attr('class', barcode)//+' '+overlay+'ov zl'+level)
        .attr("d", d3.symbol().size(size).type(type)).attr("x", x).attr("y", y);
    if (globalx && globaly) {
        marker.attr("globalx", globalx).attr("globaly", globaly);
    }

}

/** 
 * Draw a D3 rect, rects are drawn faster in the svg engines
 * @param {Number} x x coordinate normalized by the width of the image
 * @param {Number} y y coordinate normalized by the width of the image... YES it is the width as well
 * @param {Number} size size relative to the normalized image width (width=1). Common size can be 0.005
 * @param {string} color css formated color string hsl, rgb, hex etc
 * @param {string} barcode string of the key either barcode or name ,exmaple AGGC or "GeneName"*/
markerUtils.drawd3rect = function (x, y, size, color, barcode) {
    if (size === undefined || size === null || Number(size) <= 0) size = markerUtils._globalMarkerSize;

    var d3R = d3.select(tmapp[tmapp["object_prefix"] + "_svgov"].node());

    d3R.append("rect").style('fill', color).attr("class", barcode + ' ' + ovetmapp["object_prefix"] + 'ov')
        .attr("x", Number(x)).attr("width", Number(size)).attr("y", Number(y)).attr("height", Number(size));
}

/** 
 * Remove the svg group that contains a certain barcode
 * @param {string} barcode Barcode to erase
 * */
markerUtils.removeMarkerByBarcode = function (barcode) {
    var op = tmapp["object_prefix"];
    console.log("trying to delete markers " + ".Gr" + op + barcode);
    overlayUtils._d3nodes["Gr" + op + barcode].remove();
    overlayUtils._d3nodes["Gr" + op + barcode] = null;
}

markerUtils.drawCPdata= function(options){
    //pick up the property from the UI
    if(!CPDataUtils._drawCPdata){
        CPDataUtils.removeCPdata();
        return 0;
    }
    var cpop="CP";
    var CPProperty = document.getElementById(cpop+"_property_header");
    var propertyselector=CPProperty.value;
    var CPX = document.getElementById(cpop+"_X_header");
    var xselector=CPX.value;
    var CPY = document.getElementById(cpop+"_Y_header");
    var yselector=CPY.value;
    var CPLut = document.getElementById(cpop+"_colorscale");
    var interpFunction=CPLut.value;

    var x = function (d) {
        return d[xselector];
    };
    var y = function (d) {
        return d[yselector];
    };
    
    if(!CPDataUtils[cpop + "_tree"])
        console.log("CP tree does not exist");
    
    //make sure there is a group to draw into
    var svggroupname=cpop+"_prop_"+propertyselector+"_svgnode";

    console.log(svggroupname);

    if(!overlayUtils._d3nodes[svggroupname])
        overlayUtils._d3nodes[svggroupname]=overlayUtils._d3nodes[cpop+"_svgnode"].append("g").attr("id",svggroupname);

    var imageWidth=OSDViewerUtils.getImageWidth();

    // temporary values that will be acquired from the data later
    var minproperty=CPDataUtils.CP_rawdata_stats[propertyselector].min;
    var maxproperty=CPDataUtils.CP_rawdata_stats[propertyselector].max;
    //what do we draw?
    var data=[];

    var msize=CPDataUtils._markersize;
    var searchInTree=options.searchInTree || false;
    if(searchInTree || false){
        data=CPDataUtils.arrayOfElementsInBox(options.xmin, options.ymin, options.xmax, options.ymax, 
                {xselector:xselector,yselector:yselector});
        overlayUtils._d3nodes[svggroupname].attr("drawn")!="treedata"
    }else{
        msize *=1.5;
        data=CPDataUtils[cpop+"_subsampled_data"];
        overlayUtils._d3nodes[svggroupname].attr("drawn")!="subsampled"
    }
     

    overlayUtils._d3nodes[svggroupname].selectAll("*").remove();
    data.forEach(function(d){
        var property= d[propertyselector];
        var interpcolorat=(property-minproperty)/maxproperty;
        var color="#0000ff";
        if(interpFunction=="ownColorFromColumn"){
            var color=CPDataUtils._ownColorLut[property.toString()];
        }else{
            var color=d3[interpFunction](interpcolorat);
        }

        var x=Number(d[xselector])/imageWidth;
        var y=Number(d[yselector])/imageWidth;
        overlayUtils._d3nodes[svggroupname].append("rect")
        .style('fill', color ).attr("x", x-msize/2)
        .attr("width", msize)
        .attr("y", y-msize/2)
        .attr("height",msize);
    });
    

}

/** 
 * Draws all the markers from a non downsambled version of the barcode. Mostly for checking purposes
 * and is only invoked by code in a console. 
 * @param {string} barcode */
markerUtils.drawAllFromNonDownsampledBarcode = function (barcode) {
    var op = tmapp["object_prefix"];
    var d3nodeName = "Gr" + op + barcode;
    //create G group for these barcodes

    if (!overlayUtils._d3nodes[d3nodeName]) {
        //console.log("new " + d3nodeName);
        overlayUtils._d3nodes[d3nodeName] = overlayUtils._d3nodes[op + "_markers_svgnode"].append("g").attr("class", "Gr" + op + barcode);
    } else {
        overlayUtils._d3nodes[d3nodeName].selectAll("*").remove();
        overlayUtils._d3nodes[d3nodeName] = overlayUtils._d3nodes[op + "_markers_svgnode"].append("g").attr("class", "Gr" + op + barcode);
    }
    //imageWidth is the size of the image in the viewer
    var imageWidth = OSDViewerUtils.getImageWidth();

    var calculatedSize = null;
    if (document.getElementById(op + "_globalmarkersize_text")) {
        if (document.getElementById(op + "_globalmarkersize_text").value) {
            markerUtils._globalMarkerSize = Number(document.getElementById(op + "_globalmarkersize_text").value);
        } else {
            markerUtils._globalMarkerSize = 1;
        }
    }

    if (document.getElementById(barcode + "-size-" + op)) {
        calculatedSize = Number(document.getElementById(barcode + "-size-" + op).value) / imageWidth / 1000;
    }

    if ((calculatedSize === undefined || calculatedSize === null) || Number(calculatedSize) <= 0) {
        calculatedSize = markerUtils._globalMarkerSize / imageWidth / 1000;
    }
    var color = document.getElementById(barcode + "-color-" + op).value;

    var symbolIndex = document.getElementById(barcode + "-shape-" + op).value;

    dataUtils.findBarcodesInRawData(barcode).forEach(function (b) {
        markerUtils.drawSymbol(d3nodeName, markerUtils._d3Symbols[symbolIndex],
            (b.viewer_X_pos), (b.viewer_Y_pos),
            calculatedSize, color, b.letters,
            b.global_X_pos, b.global_Y_pos);
    });
}

/** 
 * Draws all the markers from a all non downsambled version of all data. Mostly for
 * visualization  purposes and is only invoked by code in a console. 
 * WILL TAKE A SUPER LONG TIME IF THE TOTAL AMOUNT OF BARCODES IS BIGGER THAN 50K
 *  */
markerUtils.drawAllFromNonDownsampled = function () {
    console.log("Be careful this might be slow");
    var op = tmapp["object_prefix"];
    //create G group for these barcodes

    dataUtils[op + "_data"].forEach(function (arr) {
        markerUtils.drawAllFromNonDownsampledBarcode(arr.key);
    });

}



/** 
 * Draws all the markers from a DOWNSAMPLED version of the barcode. Invoked when the OSDviewer 
 * covers a very big portion of the image meaning we are looking at a low resolution therefore
 * no need to draw all barcodes.
 * @param {string} barcode */
markerUtils.drawAllFromBarcode = function (barcode) {
    var op = tmapp["object_prefix"];
    var d3nodeName = "Gr" + op + barcode;
    //create G group for these barcodes

    if (!overlayUtils._d3nodes[d3nodeName]) {
        //console.log("new " + d3nodeName);
        overlayUtils._d3nodes[d3nodeName] = overlayUtils._d3nodes[op + "_markers_svgnode"].append("g").attr("class", "Gr" + op + barcode);
    } else {
        overlayUtils._d3nodes[d3nodeName].selectAll("*").remove();
        overlayUtils._d3nodes[d3nodeName] = overlayUtils._d3nodes[op + "_markers_svgnode"].append("g").attr("class", "Gr" + op + barcode);
    }
    //imageWidth is the size of the image in the viewer
    var imageWidth = OSDViewerUtils.getImageWidth();
    var calculatedSize = null;
    if (document.getElementById(op + "_globalmarkersize_text")) {
        if (document.getElementById(op + "_globalmarkersize_text").value) {
            markerUtils._globalMarkerSize = Number(document.getElementById(op + "_globalmarkersize_text").value);
        } else {
            markerUtils._globalMarkerSize = 1;
        }
    }

    if (document.getElementById(barcode + "-size-" + op)) {
        calculatedSize = Number(document.getElementById(barcode + "-size-" + op).value) / imageWidth / 1000;
    }

    if ((calculatedSize === undefined || calculatedSize === null) || Number(calculatedSize) <= 0) {
        calculatedSize = markerUtils._globalMarkerSize / imageWidth / 1000;
    }

    calculatedSize = Number(calculatedSize);
    var color = document.getElementById(barcode + "-color-" + op).value;

    var symbolIndex = document.getElementById(barcode + "-shape-" + op).value;

    dataUtils._subsampledBarcodes[barcode].forEach(function (b) {
        markerUtils.drawSymbol(d3nodeName, markerUtils._d3Symbols[symbolIndex],
            (b.viewer_X_pos), (b.viewer_Y_pos),
            calculatedSize, color, b.letters,
            b.global_X_pos, b.global_Y_pos);
    });
}

/** 
 * Draws all the markers from a precomputed LIST of a barcode. Invoked when the OSDviewer 
 * covers a very big portion of the image meaning we are looking at a low resolution therefore
 * no need to draw all barcodes.
 * @param {Array} list */
markerUtils.drawAllFromList = function (list) {
    //assuming the list is of only one barcode
    if (!list.length) {
        return;
    }

    //find if we are using gene name or barcode as key
    key=undefined;
    if(dataUtils._nameAndLetters.drawGeneName && !dataUtils._nameAndLetters.drawGeneLetters){
        //the key is GENE NAME  now
        key=list[0].gene_name;
    }else if(dataUtils._nameAndLetters.drawGeneLetters){
        //if Barcode (gene letters) is on then we rather use this
        key = list[0].letters;
    }

    if(key==undefined){
        console.log("No key to find elements in list")
    }

    //var barcode = list[0].letters; //this should be key now
    var op = tmapp["object_prefix"];
    var d3nodeName = "Gr" + op + key;

    console.log(d3nodeName + " from list");

    if (!overlayUtils._d3nodes[d3nodeName]) {
        overlayUtils._d3nodes[d3nodeName] = overlayUtils._d3nodes[op + "_markers_svgnode"].append("g").attr("class", "Gr" + op + key);
    } else {
        overlayUtils._d3nodes[d3nodeName].selectAll("*").remove();
        //overlayUtils._d3nodes[d3nodeName] = overlayUtils._d3nodes[op + "_markers_svgnode"].append("g").attr("class", "Gr" + op + key);
    }

    //imageWidth is the size of the image in the viewer
    var imageWidth = OSDViewerUtils.getImageWidth();

    var calculatedSize = Number(document.getElementById(key + "-size-" + op).value) / imageWidth / 1000;

    if (document.getElementById(op + "_globalmarkersize_text")) {
        if (document.getElementById(op + "_globalmarkersize_text").value) {
            markerUtils._globalMarkerSize = Number(document.getElementById(op + "_globalmarkersize_text").value);
        } else {
            markerUtils._globalMarkerSize = 1;
        }
    }

    if ((calculatedSize === undefined || calculatedSize === null) || Number(calculatedSize) <= 0) {
        calculatedSize = markerUtils._globalMarkerSize / imageWidth / 1000;
    }

    calculatedSize = Number(calculatedSize);

    var color = document.getElementById(key + "-color-" + op).value;

    var symbolIndex = document.getElementById(key + "-shape-" + op).value;

    for (var i in list) {
        markerUtils.drawSymbol(d3nodeName, markerUtils._d3Symbols[symbolIndex],
            (list[i].viewer_X_pos), (list[i].viewer_Y_pos),
            calculatedSize, color, key,
            list[i].global_X_pos, list[i].global_Y_pos);
    }

}

/** 
 * Checkbox to know if barcode from it should be drawn or removed 
 * @param {htmlnode} barcodeBox Checkbox to know if barcode from it should be drawn or removed  */
markerUtils.markerBoxToggle = function (barcodeBox) {
    if (barcodeBox.is(':checked')) {
        //console.log(barcodeBox[0].attributes.barcode.value, "checked");
        //get the correct overlay, fixed or moving
        //markerUtils.drawAllFromBarcode(barcodeBox[0].attributes.barcode.value);
        markerUtils.drawBarcodeByView(barcodeBox[0].attributes.barcode.value);
    } else {
        markerUtils.removeMarkerByBarcode(barcodeBox[0].attributes.barcode.value);
        console.log("not checked");
    }
}

/** 
 * Invokes all the HTML necessary to create the interface for a barcode and add the listener so that
 * clicking the box invokes the drawing of the marker or erase. 
 * Selects a color based on the barcode letters so that the color are different versions of 
 * the four corners of Ycbcr. Chooses a random shape
 * @param {Object} barObject coming from dataUtils[op+_data] which looks like this {key: "AGGGC", values: Array(1234)} 
 * @param {Object} options containing inforamtion on what to omit if necessary. for instance if
 * options.drawGeneName is there and it is true it will draw the column "name" and omit it otherwise
 * @returns {htmlnode} The nicely formated row for our markerUi table
 * */
markerUtils.markerUI = function (barObject,options) {
    var op = tmapp["object_prefix"];
    var row = HTMLElementUtils.createElement({ type: "tr", id: barObject.key + "-tr" });

    //var tdkey = HTMLElementUtils.createElement({ type: "td", innerText: barObject.key });
    //row.appendChild(tdkey);

    if(options.drawGeneLetters){
        var barcodeLetters=barObject.values[0].letters;
        var lettersrow = HTMLElementUtils.createElement({ type: "td", innerText: barcodeLetters,
            extraAttributes: { "title": barcodeLetters, "data-title":barcodeLetters } });
        row.appendChild(lettersrow);
    }

    if(options.drawGeneName){
        var gn=barObject.values[0].gene_name;
        var name = HTMLElementUtils.createElement({ type: "td", innerText: gn,
            extraAttributes: { "title": gn, "data-title":gn } });
        row.appendChild(name);
    }

    var amount = HTMLElementUtils.createElement({ type: "td", innerText: '(' + barObject.values.length + ')' });
    row.appendChild(amount);

    var check = HTMLElementUtils.createElement({ type: "td" });
    var checkinput = HTMLElementUtils.inputTypeCheckbox({
        id: barObject.key + "-checkbox-" + op,
        extraAttributes: { barcode: barObject.key },
        eventListeners: { click: function () { markerUtils.markerBoxToggle($(this)); } }
    });
    markerUtils._checkBoxes[barObject.key] = checkinput;

    check.appendChild(checkinput);
    row.appendChild(check);

    var thecolor="#5fb5f6"
    if(options.randomColorForMarker){
        thecolor=overlayUtils.randomColor("hex");
    }else{
        thecolor=HTMLElementUtils.barcodeHTMLColor(barObject.key);
    }
    var color = HTMLElementUtils.createElement({ type: "td" });
    var colorinput = HTMLElementUtils.inputTypeColor({ id: barObject.key + "-color-" + op, extraAttributes: { value: thecolor } })
    color.appendChild(colorinput);
    row.appendChild(color);

    var shape = HTMLElementUtils.createElement({ type: "td" });
    var shapeParams = { random: true, id: barObject.key + "-shape-" + op, "options": markerUtils._d3SymbolStrings };
    var shapeinput = HTMLElementUtils.selectTypeDropDown(shapeParams);
    if (shapeParams.random) { var rnd = Math.floor(Math.random() * (markerUtils._d3SymbolStrings.length)) + 0; shapeinput.selectedIndex = rnd; }
    shape.appendChild(shapeinput);
    row.appendChild(shape);

    var size = HTMLElementUtils.createElement({ type: "td" });
    var sizeinput = HTMLElementUtils.inputTypeText({ id: barObject.key + "-size-" + op, "class": "form-control" });
    size.appendChild(sizeinput);
    row.appendChild(size);

    return row;
}

/** Print the table filled with all the interactions to turn on and off the barcodes 
 * choose color, shape and size
 */
markerUtils.printBarcodeUIs = function (options) {
    //get object prefix to refer to it by code
    var op = tmapp["object_prefix"];
    //overlayUtils._d3nodes[op]=d3.select( tmapp[op+"_svgov"].node());
    //chekc if gene_name exists    
    var headers = ["Count", "Show", "Color", "Shape", "Size"];
    dataUtils.sortDataAndDownsample();
    //this is causing weird behaviour sometims it creates the name column sometimes no
    var example = dataUtils[op + "_data"][0].values[0];
    //so instead of using an example check for the keys and find if a gene_name exists
    //or maybe if it is selected in the interface adn a name is expected

    var options=dataUtils._nameAndLetters;

    if(dataUtils._nameAndLetters.drawGeneName){
        options.drawGeneName=true;
        headers.unshift("Gene");
    }
    if(dataUtils._nameAndLetters.drawGeneLetters){
        options.drawGeneLetters=true;
        headers.unshift("Spage2vec cluster");
    }

    var container = document.getElementById(op + "_markers"); container.innerHTML = "";
    var tbl = document.createElement("table");
    tbl.setAttribute("class", "table table-striped");
    tbl.setAttribute("id", op + "_table");
    tbl.setAttribute("style","word-break: break-all;");

    var colg=document.createElement ("colgroup");
    if(headers.length == 6 ){
        colg.innerHTML='<col width="20%"><col width="20%"><col width="12%"><col width="12%"><col width="15%"><col width="17%">';
        tbl.appendChild(colg);
    }
    else if(headers.length == 7 ){
        colg.innerHTML='<col width="16%"><col width="16%"><col width="15%"><col width="11%"><col width="11%"><col width="15%"><col width="16%">';
        tbl.appendChild(colg);
    } else if(headers.length>1 && headers.length<6){
        var quotient = Math.floor(100/headers.length);
        var str='<col width="'+quotient+'%">'.repeat(headers.length);
        colg.innerHTML=str;
        tbl.appendChild(colg);
    }

    var tblHead = document.createElement("thead");
    var tblBody = document.createElement("tbody");
    headers.forEach(function (header) {
        var th = document.createElement("th");
        th.appendChild(document.createTextNode(header));
        tblHead.appendChild(th);
    });
    tbl.appendChild(tblHead);

    dataUtils[op + "_data"].forEach(function (barcode) {
        var row = markerUtils.markerUI(barcode,options);
        tblBody.appendChild(row);
    });
    tbl.appendChild(tblBody);
    container.appendChild(tbl);

}
/** In the marlers interface, hide all the rows that do not contain the search string 
 *  specified in the interface in the textarea
*/
markerUtils.hideRowsThatDontContain = function () {
    var op = tmapp["object_prefix"];
    var contains = function (row, searchFor) {
        var v = row.textContent.toLowerCase();
        var v2 = searchFor;
        if (v2) {
            v2 = v2.toLowerCase();
        }
        return v.indexOf(v2) > -1;
    };

    var aneedle = document.getElementById(op + "_search").value;
    var rows = document.getElementById(op + "_table").rows;

    //make it so that the needle can be a list separated by comma, no spaces

    console.log(aneedle);

    var needles=[];

    if (aneedle.indexOf(',') > -1) { 
        needles=aneedle.split(',');
    }else{
        needles.push(aneedle)
    }

    
    for (var i = 0; i < rows.length; i++) {
        var show=false;
        needles.forEach(function(needle){
            if (contains(rows[i], needle)) {
                show=true;
            }
        });
        if (!show) {
            rows[i].setAttribute("style", "display:none;");
        } else { rows[i].setAttribute("style", ""); }
    }
}
/** Show all rows from the markers UI again */
markerUtils.showAllRows = function () {
    var op = tmapp["object_prefix"];
    var rows = document.getElementById(op + "_table").rows;

    for (var i = 0; i < rows.length; i++) {
        rows[i].setAttribute("style", "");
    }
}
/** Draw all downsampled barcodes at the same time */
markerUtils.drawAllMarkers = function () {
    var op = tmapp["object_prefix"];
    document.getElementById(op + '_drawall_btn').setAttribute("class", "btn btn-primary");

    if (document.getElementById("_globalmarkersize_text")) {
        if (document.getElementById("_globalmarkersize_text").value) {
            markerUtils._globalMarkerSize = Number(document.getElementById("_globalmarkersize_text").value);
        } else {
            markerUtils._globalMarkerSize = 1;
        }
    }

    Object.keys(dataUtils[op + "_barcodeGarden"]).forEach(function (b) {
        markerUtils.drawAllFromBarcode(b);
        document.getElementById(b + "-checkbox-" + op).checked = true;
    });

}
/** 
 * Checkbox to know if barcode from it should be drawn or removed 
 * @param {d3.quadtree} quadtree Barcode tree from the garden. dataUtils[op+_barcodeGarden] 
 * @param {Number} x0 leftmost coordinate  
 * @param {Number} y0 topmost coordinate
 * @param {Number} x3 rightmost coordinate
 * @param {Number} y3 bottommost coordinate
 * @param {Object} options Object containing options for this method. Namely globalCoords to state if the given coordinates are global or normalized by the OSD image width
 * */
markerUtils.arrayOfMarkersInBox = function (quadtree, x0, y0, x3, y3, options) {
    var op = tmapp["object_prefix"];
    //choose if the coordinates are in global image coordinates or in viewer corrdinates where 0 to 1 and  1 is image width.
    if (options.globalCoords) {
        //var pointInRegion=regionUtils.globalPointInRegion;
        var xselector = "global_X_pos";
        var yselector = "global_Y_pos";
    } else {
        //var pointInRegion=regionUtils.viewerPointInRegion;
        var xselector = "viewer_X_pos";
        var yselector = "viewer_Y_pos";
        //console.log("viewer coords");
    }

    //var imageWidth = OSDViewerUtils.getImageWidth();
    var pointsInside = [];
    //var countsInside = 0;
    //var d3nodeName = "Gr" + op + quadtree.treeName;
    //var counts=regionUtils._regions[regionid].counts[d.letters];
    quadtree.visit(function (node, x1, y1, x2, y2) {
        if (!node.length) {
            do {
                var d = node.data;
                //console.log(d);
                d.scanned = true;
                //console.log(d[xselector],d[yselector]);
                var selected = (d[xselector] >= x0) && (d[xselector] < x3) && (d[yselector] >= y0) && (d[yselector] < y3);
                if (selected) {
                    //console.log(d);
                    pointsInside.push(d);
                }
            } while (node = node.next);
        }
        return x1 >= x3 || y1 >= y3 || x2 < x0 || y2 < y0;
    });
    //if(pointsInside.length>0){
    return pointsInside;
    //}else {return null;}
}
/** Take the SVG group that contains the marker groups and empty it */
markerUtils.removeAllMarkers = function () {
    var op = tmapp["object_prefix"];
    document.getElementById(op + '_drawall_btn').setAttribute("class", "btn btn-secondary")
    overlayUtils._d3nodes[op + "_markers_svgnode"].selectAll("*").remove();
    for (var c in markerUtils._checkBoxes) {
        markerUtils._checkBoxes[c].checked = false;
    }
}
/** some desc */
markerUtils.drawAllToggle = function () {
    var op = tmapp["object_prefix"];
    var button = document.getElementById(op + '_drawall_btn');
    if (button) {
        if (button.classList.contains("btn-secondary")) markerUtils.drawAllMarkers();
        else if (button.classList.contains("btn-primary")) markerUtils.removeAllMarkers();
        else markerUtils.drawAllMarkers();
    }

}
/** some desc */
markerUtils.drawBarcodeByView = function (barcode) {
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
    if (ymin < 0) { ymin = 0; }; if (ymax > 1.0) { ymax = 1.0; };

    //console.log(xmin, ymin, xmax, ymax);

    //console.log((xmax - xmin) * (ymax - ymin));

    var total = imageWidth * imageHeight;

    xmin *= imageWidth; xmax *= imageWidth; ymin *= imageWidth; ymax *= imageWidth;

    var portion = (xmax - xmin) * (ymax - ymin);
    var percentage = portion / total;
    
    var markersInViewportBounds = []
    if (percentage < overlayUtils._percentageForSubsample) {
        //console.log("percentage less than " + overlayUtils._percentageForSubsample);
        markersInViewportBounds = markerUtils.arrayOfMarkersInBox(
            dataUtils[op + "_barcodeGarden"][barcode], xmin, ymin, xmax, ymax, { globalCoords: true }
        );

        //console.log(markersInViewportBounds.length);
        var drawThese = dataUtils.randomSamplesFromList(markerUtils._startCullingAt, markersInViewportBounds);

        //console.log("drawing=" + drawThese.length);
        markerUtils.drawAllFromList(drawThese);

    } else {

        //console.log("percentage bigger than " + overlayUtils._percentageForSubsample);
        //if the percentage of image I see is bigger than a threshold then use the predownsampled markers
        if (dataUtils._subsampledBarcodes[barcode]) {
            markerUtils.drawAllFromList(dataUtils._subsampledBarcodes[barcode]);
        } else {
            markerUtils.drawAllFromBarcode(barcode);
        }
    }//markerUtils.drawAllFromList(dataUtils.subsampledBarcodes[barcode]);

}
