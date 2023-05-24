/**
 * @namespace regionUtils
 * @classdesc Region utilities, everything to do with 
 * regions or their calculations goes here  
 * @property {Bool}     regionUtils._isNewRegion - if _isNewRegion is true then a new region will start 
 * @property {Bool}     regionUtils._currentlyDrawing - Boolean to specify if a region is currently being drawn
 * @property {Number}   regionUtils._currentRegionId - Keep then number of drawn regions and also let them be the id, 
 * @property {Object[]} regionUtils._currentPoints - Array of points for the current region, 
 * @property {String}   regionUtils._colorInactiveHandle - String for a color "#cccccc", 
 * @property {String}   regionUtils._colorActiveHandle - Color of the point in the region, 
 * @property {Number}   regionUtils._scaleHandle - Scale of the point in regions, 
 * @property {Number}   regionUtils._polygonStrokeWidth - Width of the stroke of the polygon, 
 * @property {Number}   regionUtils._handleRadius - Radius of the point of the region, 
 * @property {Number}   regionUtils._epsilonDistance - Distance at which a click from the first point will consider to close the region, 
 * @property {Object}   regionUtils._regions - Object that contains the regions in the viewer, 
 * @property {String}   regionUtils._drawingclass - String that accompanies the classes of the polygons in the interface"drawPoly", 
*/
regionUtils = {
    _isNewRegion: true,
    _currentlyDrawing: false,
    _currentRegionId: 0,
    _currentPoints: null,
    _colorInactiveHandle: "#cccccc",
    _colorActiveHandle: "#ffff00",
    _scaleHandle: 0.0025,
    _polygonStrokeWidth: 0.0015,
    _handleRadius: 0.1,
    _epsilonDistance: 0.004,
    _regions: {},
    _drawingclass: "drawPoly",
    _maxRegionsInMenu: 200
}

/** 
 *  Reset the drawing of the regions */
regionUtils.resetManager = function () {
    var drawingclass = regionUtils._drawingclass;
    d3.select("." + drawingclass).remove();
    regionUtils._isNewRegion = true;
    regionUtils._currentPoints = null;
}
/** 
 *  When a region is being drawn, this function takes care of the creation of the region */
regionUtils.manager = function (event) {
    //console.log(event);
    var drawingclass = regionUtils._drawingclass;
    //if we come here is because overlayUtils.drawRegions mode is on
    // No matter what we have to get the normal coordinates so
    //I am going to have to do a hack to get the right OSD viewer
    //I will go two parents up to get the DOM id which will tell me the name
    //and then I will look for it in tmapp... this is horrible, but will work

    /*var eventSource=event.eventSource;//this is a mouse tracker not a viewer
    var OSDsvg=d3.select(eventSource.element).select("svg").select("g");
    var stringOSDVname=eventSource.element.parentElement.parentElement.id;
    var overlay=stringOSDVname.substr(0,stringOSDVname.indexOf('_'));*/
    //console.log(overlay);
    var OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];
    var normCoords = OSDviewer.viewport.pointFromPixel(event.position);
    //var canvas=tmapp[tmapp["object_prefix"]+"_svgov"].node();
    var canvas = overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
    //console.log(normCoords);
    var regionobj;
    //console.log(d3.select(event.originalEvent.target).attr("is-handle"));
    var strokeWstr = regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom();

    if (regionUtils._isNewRegion) {
        //if this region is new then there should be no points, create a new array of points
        regionUtils._currentPoints = [];
        //it is not a new region anymore
        regionUtils._isNewRegion = false;
        //give a new id
        regionUtils._currentRegionId += 1;
        var idregion = regionUtils._currentRegionId;
        //this is out first point for this region
        var startPoint = [normCoords.x, normCoords.y];
        regionUtils._currentPoints.push(startPoint);
        //create a group to store region
        regionobj = d3.select(canvas).append('g').attr('class', drawingclass);
        regionobj.append('circle').attr('r', 10* regionUtils._handleRadius / tmapp["ISS_viewer"].viewport.getZoom()).attr('fill', regionUtils._colorActiveHandle).attr('stroke', '#ff0000')
            .attr('stroke-width', strokeWstr).attr('class', 'region' + idregion).attr('id', 'handle-0-region' + idregion)
            .attr('transform', 'translate(' + (startPoint[0].toString()) + ',' + (startPoint[1].toString()) + ') scale(' + regionUtils._scaleHandle + ')')
            .attr('is-handle', 'true').style({ cursor: 'pointer' });

    } else {
        var idregion = regionUtils._currentRegionId;
        var nextpoint = [normCoords.x, normCoords.y];
        var count = regionUtils._currentPoints.length - 1;

        //check if the distance is smaller than epsilonDistance if so, CLOSE POLYGON

        if (regionUtils.distance(nextpoint, regionUtils._currentPoints[0]) < 2* regionUtils._epsilonDistance / tmapp["ISS_viewer"].viewport.getZoom() && count >= 2) {
            regionUtils.closePolygon();
            return;
        }

        regionUtils._currentPoints.push(nextpoint);
        regionobj = d3.select("." + drawingclass);

        regionobj.append('circle')
            .attr('r', 10* regionUtils._handleRadius / tmapp["ISS_viewer"].viewport.getZoom()).attr('fill', regionUtils._colorActiveHandle).attr('stroke', '#ff0000')
            .attr('stroke-width', strokeWstr).attr('class', 'region' + idregion).attr('id', 'handle-' + count.toString() + '-region' + idregion)
            .attr('transform', 'translate(' + (nextpoint[0].toString()) + ',' + (nextpoint[1].toString()) + ') scale(' + regionUtils._scaleHandle + ')')
            .attr('is-handle', 'true').style({ cursor: 'pointer' });

        regionobj.select('polyline').remove();
        var polyline = regionobj.append('polyline').attr('points', regionUtils._currentPoints)
            .style('fill', 'none')
            .attr('stroke-width', strokeWstr)
            .attr('stroke', '#ff0000').attr('class', "region" + idregion);


    }

}
/** 
 *  Close a polygon, adding a region to the viewer and an interface to it in the side panel */
regionUtils.closePolygon = function () {
    var canvas = overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
    var drawingclass = regionUtils._drawingclass;
    var regionid = 'region' + regionUtils._currentRegionId.toString();
    d3.select("." + drawingclass).remove();
    regionsobj = d3.select(canvas);

    var hexcolor = overlayUtils.randomColor("hex");    

    regionUtils._isNewRegion = true;
    regionUtils.addRegion([[regionUtils._currentPoints]], regionid, hexcolor);
    regionUtils._currentPoints = null;
    var strokeWstr = regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom();
    regionsobj.append('path').attr("d", regionUtils.pointsToPath(regionUtils._regions[regionid].points)).attr("id", regionid + "_poly")
        .attr("class", "regionpoly").attr("polycolor", hexcolor).attr('stroke-width', strokeWstr)
        .style("stroke", hexcolor).style("fill", "none")
        .append('title').text(regionid).attr("id","path-title-" + regionid);
    regionUtils.updateAllRegionClassUI();
    $(document.getElementById("regionClass-")).collapse("show");

}

/** 
 * @param {Object} JSON formatted region to convert to GeoJSON
 * @summary This is only for backward compatibility */
 regionUtils.oldRegions2GeoJSON = function (regionsObjects) {
    try {
        // Checking if json is in old format
        if (Object.values(regionsObjects)[0].globalPoints) {
            return regionUtils.regions2GeoJSON(regionsObjects)
        }
        else {
            return regionsObjects;
        }
    } catch (error) {
        return regionsObjects;
    }
 }

/** 
 * @param {Object} GeoJSON formatted region to import
 * @summary When regions are imported, create all objects for it from a region object */
 regionUtils.regions2GeoJSON = function (regionsObjects) {
    function HexToRGB(hex) {
        var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return [ parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16) ];
    }
    function oldCoord2GeoJSONCoord(coordinates) {
        // Check for older JSON format with only one list of coordinates
        if (coordinates[0].x) {
            return [[coordinates.map(function(x) {
                return [x.x, x.y];
            })]];
        }
        return coordinates.map (function(coordinateList, i) {
            return coordinateList.map (function(coordinateList_i, index) {
                return coordinateList_i.map(function(x) {
                    return [x.x, x.y];
                });
                
            });
        })
    }
    geoJSONObjects = {
        "type": "FeatureCollection",
        "features": Object.values(regionsObjects).map (function(Region, i) {
            return {
                "type": "Feature",
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": oldCoord2GeoJSONCoord(Region.globalPoints)
                },
                "properties": {
                    "name": Region.regionName,
                    "classification": {
                        "name": Region.regionClass
                    },
                    "color": HexToRGB(Region.polycolor),
                    "isLocked": false
                }
            }
        })
    }
    return geoJSONObjects;
 }

/** 
 * @param {Object} GeoJSON formatted region to import
 * @summary When regions are imported, create all objects for it from a region object */
regionUtils.geoJSON2regions = function (geoJSONObjects) {
    // Helper functions for converting colors to hexadecimal
    var viewer = tmapp[tmapp["object_prefix"] + "_viewer"]
    if (!viewer.world || !viewer.world.getItemAt(0)) {
        setTimeout(function() {
            regionUtils.geoJSON2regions(geoJSONObjects);
        }, 100);
        return;
    }
    function rgbToHex(rgb) {
        return "#" + ((1 << 24) + (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]).toString(16).slice(1);
    }
    function decimalToHex(number) {
        if (number < 0){ number = 0xFFFFFFFF + number + 1; }
        return "#" + number.toString(16).toUpperCase().substring(2, 8);
    }
    var canvas = overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
    geoJSONObjects = regionUtils.oldRegions2GeoJSON(geoJSONObjects);
    if (!Array.isArray(geoJSONObjects)) {
        geoJSONObjects = [geoJSONObjects];
    }
    geoJSONObjects.forEach(function(geoJSONObj, geoJSONObjIndex) {
        if (geoJSONObj.type == "FeatureCollection") {
            return regionUtils.geoJSON2regions(geoJSONObj.features);
        }
        if (geoJSONObj.type == "GeometryCollection") {
            return regionUtils.geoJSON2regions(geoJSONObj.geometries);
        }
        /*if (geoJSONObj.type != "Feature") {
            return;
        }*/
        if (geoJSONObj.geometry === undefined)
            geometry = geoJSONObj
        else
            geometry = geoJSONObj.geometry
        var geometryType = geometry.type;
        var coordinates;
        if (geometryType=="Polygon") {
            coordinates = [geometry.coordinates];
        }
        else if (geometryType=="MultiPolygon") {
            coordinates = geometry.coordinates;
        }
        else {
            coordinates = [];
        }
        var geoJSONObjClass = "";
        var hexColor = "#ff0000";
        if (!geoJSONObj.properties)
            geoJSONObj.properties = {};
        if (geoJSONObj.properties.color) {
            hexColor = rgbToHex(geoJSONObj.properties.color)
        }
        if (geoJSONObj.properties.name) {
            regionName = geoJSONObj.properties.name;
        }
        else {
            regionName = "Region_" + (geoJSONObjIndex - -1);
        }
        if (geoJSONObj.properties.object_type) {
            geoJSONObjClass = geoJSONObj.properties.object_type;
        }
        if (geoJSONObj.properties.classification) {
            geoJSONObjClass = geoJSONObj.properties.classification.name;
            if (geoJSONObj.properties.classification.colorRGB) {
                hexColor = decimalToHex(geoJSONObj.properties.classification.colorRGB);
            }
        }
        coordinates = coordinates.map (function(coordinateList, i) {
            return coordinateList.map (function(coordinateList_i, index) {
                coordinateList_i = coordinateList_i.map(function(x) {
                    xPoint = new OpenSeadragon.Point(x[0], x[1]);
                    xPixel = viewer.world.getItemAt(0).imageToViewportCoordinates(xPoint);
                    return [xPixel.x.toFixed(5), xPixel.y.toFixed(5)];
                });
                return coordinateList_i.filter(function(value, index, Arr) {
                    return index % 1 == 0;
                });
            });
        })
        var regionId = "Region_geoJSON_" + geoJSONObjIndex;
        if (regionId in regionUtils._regions) {
            regionId += "_" + (Math.random() + 1).toString(36).substring(7);
        }
        regionUtils.addRegion(coordinates, regionId, hexColor, geoJSONObjClass);
        regionUtils._regions[regionId].regionName = regionName;
        regionobj = d3.select(canvas).append('g').attr('class', "mydrawingclass");
        var strokeWstr = regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom();
        regionobj.append('path').attr("d", regionUtils.pointsToPath(regionUtils._regions[regionId].points)).attr("id", regionId + "_poly")
            .attr("class", "regionpoly").attr("polycolor", hexColor).attr('stroke-width', strokeWstr)
            .style("stroke", hexColor).style("fill", "none")
            .append('title').text(regionName).attr("id","path-title-" + regionId);
        
        if (document.getElementById(regionId + "_class_ta")) {
            document.getElementById(regionId + "_class_ta").value = geoJSONObjClass;
            document.getElementById(regionId + "_name_ta").value = regionName;
            regionUtils.changeRegion(regionId);
        }
    });
}

/** 
 * @param {List} points List of list of list of points representing a path
 * @summary Given points' coordinates, returns a path string */
regionUtils.pointsToPath = function (points) {
    var path = "";
    points.forEach(function (subregions) {
        subregions.forEach(function (polygons) {
            var first = true
            polygons.forEach(function (point) {
                if (first) {path += "M";first = false;}
                else {path += "L"}
                path += point.x + " " + point.y;
            });
            path += "Z"
        });
    });
    return path;
}

/** 
 * @param {Number[]} p1 Array with x and y coords
 * @param {Number[]} p2 Array with x and y coords
 * @summary Distance between two points represented as arrays [x1,y1] and [x2,y2] */
regionUtils.distance = function (p1, p2) {
    return Math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
}
/** 
 *  @param {Number[]} points Array of 2D points in normalized coordinates
 *  @summary Create a region object and store it in the regionUtils._regions container */
regionUtils.addRegion = function (points, regionid, color, regionClass) {
    if (!regionClass) regionClass = "";
    var op = tmapp["object_prefix"];
    var viewer = tmapp[tmapp["object_prefix"] + "_viewer"]
    //var imageWidth = OSDViewerUtils.getImageWidth();
    var region = { "id": regionid, "points": [], "globalPoints": [], "regionName": regionid, "regionClass": regionClass, "barcodeHistogram": [] };
    region.len = points.length;
    var _xmin = parseFloat(points[0][0][0][0]), 
        _xmax = parseFloat(points[0][0][0][0]),
        _ymin = parseFloat(points[0][0][0][1]),
        _ymax = parseFloat(points[0][0][0][1]);
    var objectPointsArray = [];
    for (var i = 0; i < region.len; i++) {
        subregion = [];
        globalSubregion = [];
        for (var j = 0; j < points[i].length; j++) {
            polygon = [];
            globalPolygon = [];
            for (var k = 0; k < points[i][j].length; k++) {
                let x = parseFloat(points[i][j][k][0]);
                let y = parseFloat(points[i][j][k][1]);
                
                if (x > _xmax) _xmax = x;
                if (x < _xmin) _xmin = x;
                if (y > _ymax) _ymax = y;
                if (y < _ymin) _ymin = y;
                polygon.push({ "x": x, "y": y });
                let tiledImage = viewer.world.getItemAt(0);
                let imageCoord = tiledImage.viewportToImageCoordinates(
                    x, y, true
                );
                globalPolygon.push({ "x": imageCoord.x, "y": imageCoord.y });
            }
            subregion.push(polygon);
            globalSubregion.push(globalPolygon);
        }
        region.points.push(subregion);
        region.globalPoints.push(globalSubregion);
    }
    region._xmin = _xmin, region._xmax = _xmax, region._ymin = _ymin, region._ymax = _ymax;
    let tiledImage = viewer.world.getItemAt(0);
    let _min_imageCoord = tiledImage.viewportToImageCoordinates(
        _xmin,
        _ymin
    );
    let _max_imageCoord = tiledImage.viewportToImageCoordinates(
        _xmax,
        _ymax
    );
    region._gxmin = _min_imageCoord.x, region._gxmax = _max_imageCoord.x, region._gymin = _min_imageCoord.y, region._gymax = _max_imageCoord.y;
    region.polycolor = color;

    regionUtils._regions[regionid] = region;
    regionUtils._regions[regionid].associatedPoints=[];
    regionUtils.regionUI(regionid);
}
/** 
 *  @param {String} regionid Region identifier to be searched in regionUtils._regions
 *  @summary Create the whole UI for a region in the side panel */
regionUtils.regionUI = function (regionid) {

    var op = tmapp["object_prefix"];
    regionClass = regionUtils._regions[regionid].regionClass;
    if (regionClass) {
        regionUtils.addRegionClassUI (regionClass)
        regionClassID = HTMLElementUtils.stringToId(regionClass);
        var regionsPanel = document.getElementById("markers-regions-panel-" + regionClassID);
        numRegions = Object.values(regionUtils._regions).filter(x => x.regionClass==regionClass).length
        if (numRegions > regionUtils._maxRegionsInMenu) {
            spanEl = document.getElementById("regionGroupWarning-" + regionClassID)
            if (spanEl) spanEl.innerHTML = "<i class='bi bi-exclamation-triangle'></i> Max "+regionUtils._maxRegionsInMenu+" regions displayed below";
            return;
        }
    }
    else {
        regionUtils.addRegionClassUI (null)
        regionClassID = "";
        var regionsPanel = document.getElementById("markers-regions-panel-");
        numRegions = Object.values(regionUtils._regions).filter(x => x.regionClass==regionClass || x.regionClass=="undefined").length
        if (numRegions > regionUtils._maxRegionsInMenu) {
            spanEl = document.getElementById("regionGroupWarning-" + regionClassID)
            if (spanEl) spanEl.innerHTML = "<i class='bi bi-exclamation-triangle'></i> Max "+regionUtils._maxRegionsInMenu+" regions displayed below";
            return;
        }
    }
    var trPanel = HTMLElementUtils.createElement({
        kind: "tr",
        extraAttributes: {
            class: "regiontr",
            id: op + regionid + "_tr"
        }
    });
    regionsPanel.appendChild(trPanel);
    
    // Get Class name and Region name
    if (regionUtils._regions[regionid].regionClass) {
        rClass = regionUtils._regions[regionid].regionClass;
        //regionclasstext.value = rClass;
    }
    else {
        rClass = "";
    }
    if (regionUtils._regions[regionid].regionName) {
        rName = regionUtils._regions[regionid].regionName;
        //if (regionUtils._regions[regionid].regionName != regionid)
        //    regionnametext.value = rName;
    } else {
        rName = regionid;
    }
    var tdPanel = HTMLElementUtils.createElement({
        kind: "td",
    });
    var checkinput = HTMLElementUtils.inputTypeCheckbox({
        id: regionid + "_fill_ta",
        class: "form-check-input",
        value: regionUtils._regions[regionid].filled,
        eventListeners: { click: function () {
            regionUtils._regions[regionid].filled = this.checked;
            regionUtils.fillRegion(regionid, regionUtils._regions[regionid].filled);
        }}
    });
    tdPanel.appendChild(checkinput);
    trPanel.appendChild(tdPanel);
    
    var tdPanel = HTMLElementUtils.createElement({
        kind: "td",
        id: op + regionid + "_name",
    });
    var regionnametext = HTMLElementUtils.inputTypeText({
        id: regionid + "_name_ta",
        extraAttributes: {
            size:9,
            placeholder: "name",
            value: rName,
            class: "col mx-1 input-sm form-control form-control-sm"
        }
    });
    regionnametext.addEventListener('change', function () {
        regionUtils.changeRegion(regionid);
    });
    tdPanel.appendChild(regionnametext);
    trPanel.appendChild(tdPanel);
    var tdPanel = HTMLElementUtils.createElement({
        kind: "td",
    });
    var regionclasstext = HTMLElementUtils.inputTypeText({
        id: regionid + "_class_ta",
        extraAttributes: {
            size: 9,
            placeholder: "class",
            value: rClass,
            class: "col mx-1 input-sm form-control form-control-sm"
        }
    });
    regionclasstext.addEventListener('change', function () {
        regionUtils.changeRegion(regionid);
    });
    tdPanel.appendChild(regionclasstext);
    trPanel.appendChild(tdPanel);

    var regioncolorinput = HTMLElementUtils.inputTypeColor({
        id: regionid + "_color_input",
        extraAttributes: {
            class: "mx-1 form-control form-control-sm form-control-color-sm"
        }
    });
    regioncolorinput.addEventListener('change', function () {
        regionUtils.changeRegion(regionid);
    });
    if (document.getElementById(regionid + "_poly")) {
        var regionpoly = document.getElementById(regionid + "_poly");
        regioncolorinput.setAttribute("value", regionpoly.getAttribute("polycolor"));
    } else if (regionUtils._regions[regionid].polycolor) {
        regioncolorinput.setAttribute("value", regionUtils._regions[regionid].polycolor);
    }
    var tdPanel = HTMLElementUtils.createElement({
        kind: "td",
    });
    tdPanel.appendChild(regioncolorinput);
    trPanel.appendChild(tdPanel);

    trPanel.appendChild(tdPanel);
    var tdPanel = HTMLElementUtils.createElement({
        kind: "td"
    });
    var regionsdeletebutton = HTMLElementUtils.createButton({
        id: regionid + "_delete_btn",
        innerText: "<i class='bi bi-trash'></i>",
        extraAttributes: {
            parentRegion: regionid,
            class: "col btn btn-sm btn-primary form-control-sm mx-1"
        }
    });
    regionsdeletebutton.addEventListener('click', function () {
        regionUtils.deleteRegion(regionid);
    });
    tdPanel.appendChild(regionsdeletebutton);
    trPanel.appendChild(tdPanel);
    
    var trPanelHist = HTMLElementUtils.createElement({
        kind: "tr",
        extraAttributes: {
            id: op + regionid + "_tr_hist"
        }
    });
    trPanelHist.style.display="none";
    regionsPanel.appendChild(trPanelHist);
    var row = HTMLElementUtils.createElement({
        kind: "td",
        extraAttributes: {
            class: "region-histogram my-1",
            colspan: "52"
        }
    });
    trPanelHist.appendChild(row);
}

/**
 * @param {*} x X coordinate of the point to check
 * @param {*} y Y coordinate of the point to check
 * @param {*} path SVG path
 * @param {*} tmpPoint Temporary point to check if in path. This is only for speed.
 */
 regionUtils.globalPointInPath=function(x,y,path,tmpPoint) {
    tmpPoint.x = x;
    tmpPoint.y = y;
    return path.isPointInFill(tmpPoint);
};

/** 
 *  @param {Object} quadtree d3.quadtree where the points are stored
 *  @param {Number} x0 X coordinate of one point in a bounding box
 *  @param {Number} y0 Y coordinate of one point in a bounding box
 *  @param {Number} x3 X coordinate of diagonal point in a bounding box
 *  @param {Number} y3 Y coordinate of diagonal point in a bounding box
 *  @param {Object} options Tell the function 
 *  @summary Search for points inside a particular region */
 regionUtils.searchTreeForPointsInBbox = function (quadtree, x0, y0, x3, y3, options) {    
    if (options.globalCoords) {
        var xselector = options.xselector;
        var yselector = options.yselector;
    }else{
        throw {name : "NotImplementedError", message : "ViewerPointInPath not yet implemented."}; 
    }
    var pointsInside=[];
    quadtree.visit(function (node, x1, y1, x2, y2) {
        if (!node.length) {
            const markerData = dataUtils.data[options.dataset]["_processeddata"];
            const columns = dataUtils.data[options.dataset]["_csv_header"];
            for (const d of node.data) {
                const x = markerData[xselector][d];
                const y = markerData[yselector][d];
                if (x >= x0 && x < x3 && y >= y0 && y < y3) {
                    // Note: expanding each point into a full object will be
                    // very inefficient memory-wise for large datasets, so
                    // should return points as array of indices instead (TODO)
                    let p = {};
                    for (const key of columns) {
                        p[key] = markerData[key][d];
                    }
                    pointsInside.push(p);
                }
            }
        }
        return x1 >= x3 || y1 >= y3 || x2 < x0 || y2 < y0;
    });
    return pointsInside;
 }
/** 
 *  @param {Object} quadtree d3.quadtree where the points are stored
 *  @param {Number} x0 X coordinate of one point in a bounding box
 *  @param {Number} y0 Y coordinate of one point in a bounding box
 *  @param {Number} x3 X coordinate of diagonal point in a bounding box
 *  @param {Number} y3 Y coordinate of diagonal point in a bounding box
 *  @param {Object} options Tell the function 
 *  @summary Search for points inside a particular region */
regionUtils.searchTreeForPointsInRegion = function (quadtree, x0, y0, x3, y3, regionid, options) {    
    if (options.globalCoords) {
        var pointInPath = regionUtils.globalPointInPath;
        var xselector = options.xselector;
        var yselector = options.yselector;
    }else{
        throw {name : "NotImplementedError", message : "ViewerPointInPath not yet implemented."}; 
    }

    var op = tmapp["object_prefix"];
    var viewer = tmapp[op + "_viewer"]
    var countsInsideRegion = 0;
    var pointsInside=[];
    regionPath=document.getElementById(regionid + "_poly");
    var svgovname = tmapp["object_prefix"] + "_svgov";
    var svg = tmapp[svgovname]._svg;
    tmpPoint = svg.createSVGPoint();
    pointInBbox = regionUtils.searchTreeForPointsInBbox(quadtree, x0, y0, x3, y3, options);
    for (d of pointInBbox) {
        let tiledImage = viewer.world.getItemAt(0);
        let x = d[xselector];
        let y = d[yselector];
        viewport_coord = tiledImage.imageToViewportCoordinates(x,y)
        if (pointInPath(viewport_coord.x, viewport_coord.y, regionPath, tmpPoint)) {
            countsInsideRegion += 1;
            pointsInside.push(d);
        }
    }
    if (countsInsideRegion) {
        regionUtils._regions[regionid].barcodeHistogram.push({ "key": quadtree.treeID, "name": quadtree.treeName, "count": countsInsideRegion });
    }
    return pointsInside;
}

/** Fill all regions  */
regionUtils.fillAllRegions=function(){
    var allFilled = Object.values(regionUtils._regions).map(function(e) { return e.filled; }).includes(false);
    for(var regionid in regionUtils._regions){
        if (regionUtils._regions.hasOwnProperty(regionid)) {
            regionUtils.fillRegion(regionid, allFilled);
            if(document.getElementById(regionid + "_fill_ta"))
                document.getElementById(regionid + "_fill_ta").checked = allFilled;
        }
    }
}

/** 
 * @param {String} regionid String id of region to fill
 * @summary Given a region id, fill this region in the interface */
regionUtils.fillRegion = function (regionid, value) {
    if (value === undefined) {
        // we toggle
        if(regionUtils._regions[regionid].filled === 'undefined'){
            value = true;
        }
        else {
            value = !regionUtils._regions[regionid].filled;
        }
    }
    regionUtils._regions[regionid].filled=value;
    var newregioncolor = regionUtils._regions[regionid].polycolor;
    var d3color = d3.rgb(newregioncolor);
    var newStyle="";
    if(regionUtils._regions[regionid].filled){
        newStyle = "stroke: " + d3color.rgb().toString()+";";
        d3color.opacity=0.5;
        newStyle +="fill: "+d3color.rgb().toString()+";";
    }else{
        newStyle = "stroke: " + d3color.rgb().toString() + "; fill: none;";
    }
    document.getElementById(regionid + "_poly").setAttribute("style", newStyle);

}
/** 
 * @param {String} regionid String id of region to delete
 * @summary Given a region id, deletes this region in the interface */
regionUtils.deleteRegion = function (regionid) {
    var regionPoly = document.getElementById(regionid + "_poly")
    regionPoly.parentElement.removeChild(regionPoly);
    delete regionUtils._regions[regionid];
    var op = tmapp["object_prefix"];
    var rPanel = document.getElementById(op + regionid + "_tr");
    if (rPanel) {
        rPanel.parentElement.removeChild(rPanel);
        var rPanelHist = document.getElementById(op + regionid + "_tr_hist");
        rPanelHist.parentElement.removeChild(rPanelHist);
    }
    regionUtils.updateAllRegionClassUI();
}
/** 
 * @param {String} regionid String id of region to delete
 * @summary Given a region id, deletes this region in the interface */
regionUtils.deleteAllRegions = function () {
    var canvas = overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
    regionsobj = d3.select(canvas);
    regionsobj.selectAll("*").remove();

    var regionsPanel = document.getElementById("markers-regions-panel");
    regionsPanel.innerText = "";
    var regionsPanel = document.getElementById("regionAccordions");
    regionsPanel.innerText = "";
    regionUtils._regions = {};
}
regionUtils.updateAllRegionClassUI = function (regionClass) {
    // get all region classes
    var allRegionClasses = Object.values(regionUtils._regions).map(function(e) { return e.regionClass; })
    // get only unique values
    var singleRegionClasses = allRegionClasses.filter((v, i, a) => a.indexOf(v) === i);
    singleRegionClasses.forEach(function (regionClass) {
        regionClassID = HTMLElementUtils.stringToId(regionClass);
        numRegions = allRegionClasses.filter(x => x==regionClass).length
        spanEl = document.getElementById("numRegions-" + regionClassID)
        if (spanEl) {
            spanEl.innerText = numRegions;
            spanElS = document.getElementById("numRegionsS-" + regionClassID)
            if (numRegions > 1) spanElS.innerText = "s"; else spanElS.innerText = ""; 
        }
    })
    Array.from(document.getElementsByClassName("region-accordion")).forEach(function(accordionItem) {
        if (Array.from(accordionItem.getElementsByClassName("regiontr")).length == 0) {
            accordionItem.remove();
        }
    });
}
/** 
 *  @param {String} regionClass Region class
 *  @summary Add accordion for a new region class */
regionUtils.addRegionClassUI = function (regionClass) {
    if (regionClass == null) regionClass = "";
    var op = tmapp["object_prefix"];
    var regionClassID = HTMLElementUtils.stringToId(regionClass);
    var accordion_item = document.getElementById("regionClassItem-" + regionClassID);
    if (!accordion_item) {
        var regionAccordions = document.getElementById("regionAccordions");
        var accordion_item = HTMLElementUtils.createElement({
            kind: "div",
            extraAttributes: {
                class: "accordion-item region-accordion",
                id: "regionClassItem-" + regionClassID
            }
        });
        regionAccordions.appendChild(accordion_item);
        var accordion_header = HTMLElementUtils.createElement({
            kind: "h2",
            extraAttributes: {
                class: "accordion-header",
                id: "regionClassHeading-" + regionClassID
            }
        });
        accordion_item.appendChild(accordion_header);
        if (!regionClass) regionClassName = "Unclassified"; else regionClassName = regionClass;
        var accordion_header_button = HTMLElementUtils.createElement({
            kind: "button",
            innerHTML: "<i class='bi bi-pentagon'></i>&nbsp;" + regionClassName + " (<span id='numRegions-" + regionClassID + "'>1</span>&nbsp;region<span id='numRegionsS-" + regionClassID + "'></span>)&nbsp;<span class='text-warning' id='regionGroupWarning-" + regionClassID + "'></span>",
            extraAttributes: {
                "type": "button",
                "class": "accordion-button collapsed",
                "id": "regionClassHeading-" + regionClassID,
                "data-bs-toggle": "collapse",
                "data-bs-target": "#" + "regionClass-" + regionClassID,
                "aria-expanded": "true",
                "aria-controls": "collapseOne"
            }
        });
        accordion_header.appendChild(accordion_header_button);
        
        var accordion_content = HTMLElementUtils.createElement({
            kind: "div",
            extraAttributes: {
                class: "accordion-collapse collapse px-2",
                id: "regionClass-" + regionClassID,
                "aria-labelledby":"headingOne",
                "data-bs-parent":"#regionAccordions"
            }
        });
        accordion_item.appendChild(accordion_content);
        var buttonRow = HTMLElementUtils.createElement({
            kind: "div",
            extraAttributes: {
                class: "row my-1 mx-2"
            }
        });
        accordion_content.appendChild(buttonRow);
        
        var regionTable = HTMLElementUtils.createElement({
            kind: "table",
            extraAttributes: {
                class: "table regions_table",
                id: "markers-regions-table-" + regionClassID
            }
        });
        accordion_content.appendChild(regionTable);
        var colg=document.createElement ("colgroup");
        colg.innerHTML='<col width="5%"><col width="38%"><col width="37%"><col width="10%"><col width="10%">';
        regionTable.appendChild(colg);
        var tblHead = document.createElement("thead");
        var tblHeadTr = document.createElement("tr");
        tblHead.appendChild(tblHeadTr);
        tblHeadTr.appendChild(HTMLElementUtils.createElement({kind:"th",innerText:"Fill"}));
        tblHeadTr.appendChild(HTMLElementUtils.createElement({kind:"th",innerText:"Name"}));
        tblHeadTr.appendChild(HTMLElementUtils.createElement({kind:"th",innerText:"Class"}));
        tblHeadTr.appendChild(HTMLElementUtils.createElement({kind:"th",innerText:"Color"}));
        tblHeadTr.appendChild(HTMLElementUtils.createElement({kind:"th",innerText:"Delete"}));
        regionTable.appendChild(tblHead);
        var regionTbody = HTMLElementUtils.createElement({
            kind: "tbody",
            id: "markers-regions-panel-" + regionClassID
        });
        regionTable.appendChild(regionTbody);
            
        var trPanel = HTMLElementUtils.createElement({
            kind: "tr"
        });
        regionTbody.appendChild(trPanel);
        
        var tdPanel = HTMLElementUtils.createElement({
            kind: "td",
        });
        var checkinput = HTMLElementUtils.inputTypeCheckbox({
            class: "form-check-input",
            id: regionClassID + "_group_fill_ta",
            value: false,
            eventListeners: { click: function () {
                var newFill = this.checked;
                groupRegions = Object.values(regionUtils._regions).filter(
                    x => x.regionClass==regionClass
                ).forEach(function (region) {
                    region.filled = newFill;
                    if (document.getElementById(region.id + "_fill_ta"))
                        document.getElementById(region.id + "_fill_ta").checked = newFill;
                    regionUtils.fillRegion(region.id, newFill);
                });
            }}
        });
        tdPanel.appendChild(checkinput);
        trPanel.appendChild(tdPanel);
        
        var tdPanel = HTMLElementUtils.createElement({
            kind: "td",
            innerHTML: "<label style='cursor:pointer' for='"+regionClassID+"_group_fill_ta'>All</label>"
        });
        trPanel.appendChild(tdPanel);
        var tdPanel = HTMLElementUtils.createElement({
            kind: "td",
        });
        if (regionClass) rClass = regionClass; else rClass = "";
        var regionclasstext = HTMLElementUtils.inputTypeText({
            extraAttributes: {
                size: 9,
                placeholder: "class",
                value: rClass,
                class: "col mx-1 input-sm form-control form-control-sm"
            }
        });
        regionclasstext.addEventListener('change', function () {
            var newClass = this.value;
            groupRegions = Object.values(regionUtils._regions).filter(
                x => x.regionClass==regionClass
            );
            for (region of groupRegions) {
                if (document.getElementById(region.id + "_class_ta"))
                    document.getElementById(region.id + "_class_ta").value = newClass;
                regionUtils.changeRegion(region.id);
                region.regionClass = newClass;
            };
            regionUtils.updateAllRegionClassUI();
        });
        tdPanel.appendChild(regionclasstext);
        trPanel.appendChild(tdPanel);
    
        var regioncolorinput = HTMLElementUtils.inputTypeColor({
            extraAttributes: {
                class: "mx-1 form-control form-control-sm form-control-color-sm"
            }
        });
        regioncolorinput.addEventListener('change', function () {
            var newColor = this.value;
            groupRegions = Object.values(regionUtils._regions).filter(
                x => x.regionClass==regionClass
            )
            for (region of groupRegions) {
                region.polycolor = newColor;
                if (document.getElementById(region.id + "_color_input"))
                    document.getElementById(region.id + "_color_input").value = newColor;
                regionUtils.changeRegion(region.id);
            };
        });
        var tdPanel = HTMLElementUtils.createElement({
            kind: "td",
        });
        tdPanel.appendChild(regioncolorinput);
        trPanel.appendChild(tdPanel);
    
        trPanel.appendChild(tdPanel);
        var tdPanel = HTMLElementUtils.createElement({
            kind: "td"
        });
        var regionsdeletebutton = HTMLElementUtils.createButton({
            innerText: "<i class='bi bi-trash'></i>",
            extraAttributes: {
                class: "col btn btn-sm btn-primary form-control-sm mx-1"
            }
        });
        regionsdeletebutton.addEventListener('click', function () {
            interfaceUtils.confirm('Are you sure you want to delete the whole '+regionClass+' group?')
            .then(function(_confirm){
                if (_confirm) {
                    groupRegions = Object.values(regionUtils._regions).filter(
                        x => x.regionClass==regionClass
                    ).forEach(function (region) {
                        regionUtils.deleteRegion(region.id);
                    });
                }
            });
        });
        tdPanel.appendChild(regionsdeletebutton);
        trPanel.appendChild(tdPanel);

        var regionanalyzebutton = HTMLElementUtils.createButton({
            id: regionClassID + "_analyze_btn",
            innerText: "Analyze group",
            extraAttributes: {
                parentRegion: regionClassID,
                class: "col btn btn-primary btn-sm form-control mx-1"
            }
        });
        
        regionanalyzebutton.addEventListener('click', function () {
            if (Object.keys(dataUtils.data).length == 0) {
                interfaceUtils.alert("Load markers first");
                return;
            }
            Object.values(regionUtils._regions).filter(
                x => x.regionClass==regionClass
            ).forEach(function(region){
                regionUtils.analyzeRegion(region.id);
            });
        });
        buttonRow.appendChild(regionanalyzebutton);

    }
}

/** 
 *  @param {String} regionid Region identifier
 *  @summary Change the region properties like color, class name or region name */
regionUtils.changeRegion = function (regionid) {
    if (document.getElementById(regionid + "_name_ta")) {
        var op = tmapp["object_prefix"];
        var rPanel = document.getElementById(op + regionid + "_tr");
        var rPanel_hist = document.getElementById(op + regionid + "_tr_hist");
        if (regionUtils._regions[regionid].regionClass != document.getElementById(regionid + "_class_ta").value) {
            if (document.getElementById(regionid + "_class_ta").value) {
                regionUtils._regions[regionid].regionClass = document.getElementById(regionid + "_class_ta").value;
                classID = HTMLElementUtils.stringToId(regionUtils._regions[regionid].regionClass);
                regionUtils.addRegionClassUI (regionUtils._regions[regionid].regionClass)
                $(rPanel).detach().appendTo('#markers-regions-panel-' + classID)
                $(rPanel_hist).detach().appendTo('#markers-regions-panel-' + classID)
            } else {
                regionUtils._regions[regionid].regionClass = null;
                regionUtils.addRegionClassUI (null)
                classID = HTMLElementUtils.stringToId(regionUtils._regions[regionid].regionClass);
                $(rPanel).detach().appendTo('#markers-regions-panel-')
                $(rPanel_hist).detach().appendTo('#markers-regions-panel-')
            }
            regionUtils.updateAllRegionClassUI();
        }
        if (document.getElementById(regionid + "_name_ta").value) {
            regionUtils._regions[regionid].regionName = document.getElementById(regionid + "_name_ta").value;
        } else {
            regionUtils._regions[regionid].regionName = regionid;
        }
        var newregioncolor = document.getElementById(regionid + "_color_input").value;
        regionUtils._regions[regionid].polycolor = newregioncolor;
    }
    regionUtils.updateRegionDraw(regionid);
}

/** 
 *  @param {String} regionid Region identifier
 *  @summary Change the region properties like color, class name or region name */
 regionUtils.updateRegionDraw = function (regionid) {
    var newregioncolor = regionUtils._regions[regionid].polycolor;
    var d3color = d3.rgb(newregioncolor);
    var newStyle = "stroke: " + d3color.rgb().toString() + "; fill: none;";
    document.getElementById(regionid + "_poly").setAttribute("style", newStyle);
    if (regionUtils._regions[regionid].filled === undefined)
        regionUtils._regions[regionid].filled = false;
    regionUtils.fillRegion(regionid, regionUtils._regions[regionid].filled);
    if (regionUtils._regions[regionid].regionName) {rName = regionUtils._regions[regionid].regionName;}
    else {rName = regionid;}
    document.getElementById("path-title-" + regionid).innerHTML = rName;
 }

/** 
 *  regionUtils */
regionUtils.analyzeRegion = function (regionid) {
    var op = tmapp["object_prefix"];

    function compare(a, b) {
        if (a.count > b.count)
            return -1;
        if (a.count < b.count)
            return 1;
        return 0;
    }

    function clone(obj) {
        if (null == obj || "object" != typeof obj) return obj;
        var copy = obj.constructor();
        for (var attr in obj) {
            if (obj.hasOwnProperty(attr)) copy[attr] = obj[attr];
        }
        return copy;
    }

    regionUtils._regions[regionid].associatedPoints=[];
    regionUtils._regions[regionid].barcodeHistogram=[];
    allDatasets = Object.keys(dataUtils.data);
    for (var uid of allDatasets) {
        var allkeys=Object.keys(dataUtils.data[uid]["_groupgarden"]);

        var datapath = dataUtils.data[uid]["_csv_path"];
        if (datapath.includes(".csv") || datapath.includes(".CSV")) {
            // Strip everything except the filename, to save a bit of memory
            // and reduce the filesize when exporting to CSV
            datapath = dataUtils.data[uid]["_csv_path"].split("/").pop();
        } else if (datapath.includes("base64")) {
            // If the file is encoded in the path as a Base64 string, use
            // the name of the marker tab as identifier in the output CSV
            datapath = dataUtils.data[uid]["_name"];
        }

        for (var codeIndex in allkeys) {
            var code = allkeys[codeIndex];

            var pointsInside=regionUtils.searchTreeForPointsInRegion(dataUtils.data[uid]["_groupgarden"][code],
                regionUtils._regions[regionid]._gxmin,regionUtils._regions[regionid]._gymin,
                regionUtils._regions[regionid]._gxmax,regionUtils._regions[regionid]._gymax,
                regionid, {
                    "globalCoords":true,
                    "xselector":dataUtils.data[uid]["_X"],
                    "yselector":dataUtils.data[uid]["_Y"],
                    "dataset":uid
                });
            if(pointsInside.length>0){
                pointsInside.forEach(function(p){
                    var pin=clone(p);
                    pin.regionid=regionid;
                    pin.dataset=datapath
                    regionUtils._regions[regionid].associatedPoints.push(pin)
                });
            }
        }
    }
    regionUtils._regions[regionid].barcodeHistogram.sort(compare);

    var rPanel = document.getElementById(op + regionid + "_tr_hist");
    if (rPanel) {
        var rpanelbody = rPanel.getElementsByClassName("region-histogram")[0];
        histodiv = document.getElementById(regionid + "_histogram");
        if (histodiv) {
            histodiv.parentNode.removeChild(histodiv);
        }

        var div = HTMLElementUtils.createElement({ kind: "div", id: regionid + "_histogram" });
        var histogram = regionUtils._regions[regionid].barcodeHistogram;
        var table = div.appendChild(HTMLElementUtils.createElement({
            kind: "table",
            extraAttributes: {
                class: "table table-striped",
                style: "overflow-y: auto;"
            }
        }));
        thead = HTMLElementUtils.createElement({kind: "thead"});
        thead.innerHTML = `<tr>
        <th scope="col">Key</th>
        <th scope="col">Name</th>
        <th scope="col">Count</th>
        </tr>`;
        tbody = HTMLElementUtils.createElement({kind: "tbody"});
        table.appendChild(thead);
        table.appendChild(tbody);

        for (var i in histogram) {
            var innerHTML = "";
            innerHTML += "<td>" + histogram[i].key + "</td>";
            innerHTML += "<td>" + histogram[i].name + "</td>";
            innerHTML += "<td>" + histogram[i].count + "</td>";
            tbody.appendChild(HTMLElementUtils.createElement({
                kind: "tr",
                "innerHTML": innerHTML
            }));
        }
        rpanelbody.appendChild(div);
        $(rPanel).show();
    }
}
/** 
 *  regionUtils */
regionUtils.regionsOnOff = function () {
    overlayUtils._drawRegions = !overlayUtils._drawRegions;
    var op = tmapp["object_prefix"];
    let regionIcon = document.getElementById(op + '_drawregions_icon');
    if (overlayUtils._drawRegions) {
        regionIcon.classList.remove("bi-circle");
        regionIcon.classList.add("bi-check-circle");
    } else {
        regionUtils.resetManager();
        regionIcon.classList.remove("bi-check-circle");
        regionIcon.classList.add("bi-circle");
    }
}
/** 
 *  regionUtils */
regionUtils.exportRegionsToJSON = function () {
    regionUtils.regionsToJSON();
}
/** 
 *  regionUtils */
regionUtils.importRegionsFromJSON = function () {
    regionUtils.deleteAllRegions();
    regionUtils.JSONToRegions();
}

regionUtils.pointsInRegionsToCSV=function(){
    var alldata=[]
    for (r in regionUtils._regions){
        var regionPoints=regionUtils._regions[r].associatedPoints;
        regionUtils._regions[r].associatedPoints.forEach(function(p){
            p.regionName=regionUtils._regions[r].regionName
            p.regionClass=regionUtils._regions[r].regionClass
            alldata.push(p);
        });
    }

    var csvRows=[];
    var headers=alldata.reduce(function(arr, o) {
        return Object.keys(o).reduce(function(a, k) {
          if (a.indexOf(k) == -1) a.push(k);
          return a;
        }, arr)
      }, []);
    csvRows.push(headers.join(','));
    
    for(var row of alldata){
        var values=[];
        headers.forEach(function(header){
            const value = row[header];
            if (isNaN(value) && typeof(value) == "string" &&
                (value.includes(",") || value.includes("\""))) {
                // Make sure that commas and quotation marks are properly escaped
                let escaped = value;
                if (escaped.includes(",") || escaped.includes("\""))
                    escaped = "\"" + escaped.replaceAll("\"", "\"\"") + "\"";
                values.push(escaped);
            } else {
                values.push(value);
            }
        });
        csvRows.push(values.join(","));
    }
    var theblobdata=csvRows.join('\n');

    regionUtils.downloadPointsInRegionsCSV(theblobdata);

}

regionUtils.downloadPointsInRegionsCSV=function(data){
    var blob = new Blob([data],{kind:"text/csv"});
    var url=window.URL.createObjectURL(blob);
    var a=document.createElement("a");
    a.setAttribute("hidden","");
    a.setAttribute("href",url);
    a.setAttribute("download","pointsinregions.csv");
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}


regionUtils.regionsToJSON= function(){
    if (window.Blob) {
        var op=tmapp["object_prefix"];
        var jsonse = JSON.stringify(regionUtils.regions2GeoJSON(regionUtils._regions));
        var blob = new Blob([jsonse], {kind: "application/json"});
        var url  = URL.createObjectURL(blob);
        var a=document.createElement("a");// document.getElementById("invisibleRegionJSON");
        if(document.getElementById(op+"_region_file_name")){
            var name=document.getElementById(op+"_region_file_name").value;
        }else{
            var name="regions.json";
        }
        a.href        = url;
        a.download    = name;
        a.textContent = "Download backup.json";
        a.click();
          // Great success! The Blob API is supported.
    } else {
        interfaceUtils.alert('The File APIs are not fully supported in this browser.');
    }        
}

regionUtils.JSONToRegions= function(filepath){
    if(filepath!==undefined){
        const queryString = window.location.search;
        const urlParams = new URLSearchParams(queryString);
        const path = urlParams.get('path')
        if (path != null) {
            filepath = path + "/" + filepath;
        }
        fetch(filepath)
        .then((response) => {
            return response.json();
        })
        .then((regionsobj) => {
            regionUtils.JSONValToRegions(regionsobj);
        });
    }
    else if(window.File && window.FileReader && window.FileList && window.Blob) {
        var op=tmapp["object_prefix"];
        var text=document.getElementById(op+"_region_files_import");
        var file=text.files[0];
        var reader = new FileReader();
        reader.onload=function(event) {
            // The file's text will be printed here
            regionUtils.JSONValToRegions(JSON.parse(event.target.result));
        };
        reader.readAsText(file);
    } else {
        interfaceUtils.alert('The File APIs are not fully supported in this browser.');
    }
}

regionUtils.JSONValToRegions= function(jsonVal){
    // The file's text will be printed here
    var regions=jsonVal;
    regionUtils.geoJSON2regions(regions);
    regionUtils.updateAllRegionClassUI();
    $('[data-bs-target="#markers-regions-project-gui"]').tab('show');
}
