/**
 * @namespace regionUtils
 * @classdesc Region utilities, everything to do with 
 * regions or their calculations goes here  
 * @property {String}   regionUtils._regionMode - Can be null, "points", "rectangles", "freehand", "brush", "selection"
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
 * @property {Object[]} regionUtils._edgeLists - Data structure used for rendering regions with WebGL
 * @property {Object[]} regionUtils._regionToColorLUT - LUT for storing color and visibility per object ID
 * @property {Object{}} regionUtils._regionIDToIndex - Mapping between region ID (string) and object ID (index)
 * @property {Object{}} regionUtils._regionIndexToID - Mapping between object ID (index) and region ID (string)
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
    _edgeLists: [],
    _regionToColorLUT: [],
    _regionIDToIndex: {},
    _regionIndexToID: {}
}

/** 
 *  Reset the drawing of the regions */
regionUtils.resetManager = function () {
    var drawingclass = regionUtils._drawingclass;
    d3.select("." + drawingclass).remove();
    drawingclass = "_brushRegion";
    d3.select("." + drawingclass).remove();
    
    regionUtils._isNewRegion = true;
    regionUtils._currentPoints = null;
}
/**
 * 
 * @param {OpenSeadragon.point} coordinates
 * @returns {Number} index of the layer that contains the coordinates
 */
regionUtils.getLayerFromCoord = function (coordinates) {
    var op = tmapp["object_prefix"];
    var viewer = tmapp[op + "_viewer"];
    for (var i = viewer.world.getItemCount()-1; i >= 0; i--) {
        let tiledImage = viewer.world.getItemAt(i);
        let imageCoord = tiledImage.viewportToImageCoordinates(
            coordinates.x, coordinates.y, true
        );
        if (imageCoord.x > 0 && imageCoord.y > 0 &&
            imageCoord.x < tiledImage.getContentSize().x && imageCoord.y < tiledImage.getContentSize().y) {
                return i;
        }
    }
    return 0;
}
/**
 * Get viewport coordinates from image coordinates for a given layer
 * @param {Object} globalPoints 
 * @param {Number} layerIndex
 * @returns viewportPoints
 */
regionUtils.globalPointsToViewportPoints = function (globalPoints, layerIndex) {
    var op = tmapp["object_prefix"];
    var viewer = tmapp[op + "_viewer"];
    var viewportPoints = [];
    for (var i = 0; i < globalPoints.length; i++) {
        var subregion = [];
        for (var j = 0; j < globalPoints[i].length; j++) {
            var polygon = [];
            for (var k = 0; k < globalPoints[i][j].length; k++) {
                let x = globalPoints[i][j][k].x;
                let y = globalPoints[i][j][k].y;
                let tiledImage = viewer.world.getItemAt(layerIndex);
                let imageCoord = tiledImage.imageToViewportCoordinates(
                    x, y, true
                );
                polygon.push({ "x": imageCoord.x, "y": imageCoord.y });
            }
            subregion.push(polygon);
        }
        viewportPoints.push(subregion);
    }
    return viewportPoints;
}

/**
 * Get viewport coordinates from image coordinates for a given layer
 * @param {Object} globalPoints 
 * @param {Number} layerIndex
 * @returns viewportPoints
 */
regionUtils.viewportPointsToGlobalPoints = function (viewportPoints, layerIndex) {
    var op = tmapp["object_prefix"];
    var viewer = tmapp[op + "_viewer"];
    var globalPoints = [];
    for (var i = 0; i < viewportPoints.length; i++) {
        var subregion = [];
        for (var j = 0; j < viewportPoints[i].length; j++) {
            var polygon = [];
            for (var k = 0; k < viewportPoints[i][j].length; k++) {
                let x = viewportPoints[i][j][k].x;
                let y = viewportPoints[i][j][k].y;
                let tiledImage = viewer.world.getItemAt(layerIndex);
                let imageCoord = tiledImage.viewportToImageCoordinates(
                    x, y, true
                );
                polygon.push({ "x": imageCoord.x, "y": imageCoord.y });
            }
            subregion.push(polygon);
        }
        globalPoints.push(subregion);
    }
    return globalPoints;
}

/** 
 *  When a region is being drawn, this function takes care of the creation of the region */
regionUtils.manager = function (event) {
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
    var OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];
    var normCoords = OSDviewer.viewport.pointFromPixel(event.position);
    //var canvas=tmapp[tmapp["object_prefix"]+"_svgov"].node();
    var canvas = overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
    var regionobj;
    var strokeWstr = regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom();

    if (regionUtils._isNewRegion) {
        //if this region is new then there should be no points, create a new array of points
        regionUtils._currentPoints = [];
        //it is not a new region anymore
        regionUtils._isNewRegion = false;
        //give a new id
        regionUtils._currentRegionId += 1;
        //set corresponding layer index
        regionUtils._currentLayerIndex = regionUtils.getLayerFromCoord(normCoords);
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
    
    regionUtils._isNewRegion = true;
    regionUtils._currentPoints.push(regionUtils._currentPoints[0]);
    regionUtils.addRegion([[regionUtils._currentPoints]], regionid, null, "", regionUtils._currentLayerIndex);
    regionUtils._currentPoints = null;

    regionUtils.updateAllRegionClassUI();
    regionUtils.highlightRegion(regionid);
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
regionUtils.geoJSON2regions = async function (geoJSONObjects) {
    // Helper functions for converting colors to hexadecimal
    var viewer = tmapp[tmapp["object_prefix"] + "_viewer"]
    await overlayUtils.waitLayersReady();
    function rgbToHex(rgb) {
        return "#" + ((1 << 24) + (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]).toString(16).slice(1);
    }
    function decimalToHex(number) {
        if (number < 0){ number = 0xFFFFFFFF + number + 1; }
        return "#" + number.toString(16).toUpperCase().substring(2, 8);
    }
    geoJSONObjects = regionUtils.oldRegions2GeoJSON(geoJSONObjects);
    if (!Array.isArray(geoJSONObjects)) {
        geoJSONObjects = [geoJSONObjects];
    }
    //geoJSONObjects = geoJSONObjects.slice(0,3000);
    // Temporary hides the table for chrome issue with slowliness
    document.querySelector("#regionAccordions").classList.add("d-none");
    console.log(geoJSONObjects.length + " regions to import");
    for (let geoJSONObjIndex in geoJSONObjects) {
        let geoJSONObj = geoJSONObjects[geoJSONObjIndex];
        if (geoJSONObj.type == "FeatureCollection") {
            return await regionUtils.geoJSON2regions(geoJSONObj.features);
        }
        if (geoJSONObj.type == "GeometryCollection") {
            return await regionUtils.geoJSON2regions(geoJSONObj.geometries);
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
        //TODO: collectionIndex from modal if multiple layers
        regionUtils.addRegion(coordinates, regionId, hexColor, geoJSONObjClass, 0);
        regionUtils._regions[regionId].regionName = regionName;
        if (document.getElementById(regionId + "_class_ta")) {
            document.getElementById(regionId + "_class_ta").value = geoJSONObjClass;
            document.getElementById(regionId + "_name_ta").value = regionName;
            regionUtils.changeRegion(regionId);
        }
    };
    glUtils.updateRegionDataTextures();
    glUtils.updateRegionLUTTextures();
    glUtils.draw();
    document.querySelector("#regionAccordions").classList.remove("d-none");
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
regionUtils.addRegion = function (points, regionid, color, regionClass, collectionIndex) {
    if (collectionIndex == undefined) collectionIndex = 0;
    if (!regionClass) regionClass = "";
    const regionClassID = HTMLElementUtils.stringToId("region_" + regionClass);
    if (!color) {
        const color_picker = document.getElementById(`regionUI_${regionClassID}_color`);
        if (color_picker) {
            color = color_picker.value;
        } else {
            color = "#ff0000";
        }
    }
    var op = tmapp["object_prefix"];
    var viewer = tmapp[tmapp["object_prefix"] + "_viewer"]
    //var imageWidth = OSDViewerUtils.getImageWidth();
    var region = { 
        "id": regionid, 
        "globalPoints": [], 
        "regionName": regionid, 
        "regionClass": regionClass, 
        "barcodeHistogram": [],
        "visibility": true,
        "collectionIndex": collectionIndex
    };
    region.len = points.length;
    for (var i = 0; i < region.len; i++) {
        let globalSubregion = [];
        for (var j = 0; j < points[i].length; j++) {
            let globalPolygon = [];
            for (var k = 0; k < points[i][j].length; k++) {
                let x = parseFloat(points[i][j][k][0]);
                let y = parseFloat(points[i][j][k][1]);
                
                let tiledImage = viewer.world.getItemAt(collectionIndex);
                let imageCoord = tiledImage.viewportToImageCoordinates(
                    x, y, true
                );
                globalPolygon.push({ "x": imageCoord.x, "y": imageCoord.y });
            }
            globalSubregion.push(globalPolygon);
        }
        region.globalPoints.push(globalSubregion);
    }
    region.polycolor = color;

    regionUtils.updateBbox(region);
    regionUtils._regions[regionid] = region;
    regionUtils._regions[regionid].associatedPoints=[];
}

/**
 * Update bounding box region._gxmin, region._gxmax, region._gymin, region._gymax
 * from region.globalPoints
 */
regionUtils.updateBbox=function(region) {
    region._gxmin = Infinity;
    region._gxmax = -Infinity;
    region._gymin = Infinity;
    region._gymax = -Infinity;
    for (var i = 0; i < region.globalPoints.length; i++) {
        for (var j = 0; j < region.globalPoints[i].length; j++) {
            for (var k = 0; k < region.globalPoints[i][j].length; k++) {
                var x = region.globalPoints[i][j][k].x;
                var y = region.globalPoints[i][j][k].y;
                region._gxmin = Math.min(region._gxmin, x);
                region._gxmax = Math.max(region._gxmax, x);
                region._gymin = Math.min(region._gymin, y);
                region._gymax = Math.max(region._gymax, y);
            }
        }
    }
}

/**
 * @deprecated Kept for backward compatibility
 * @param {*} x X coordinate of the point to check
 * @param {*} y Y coordinate of the point to check
 * @param {*} path SVG path
 * @param {*} tmpPoint Temporary point to check if in path. This is only for speed.
 */
regionUtils.globalPointInPath=function(x,y,path,tmpPoint) {
    tmpPoint.x = x;
    tmpPoint.y = y;
    return path.isPointInFill(tmpPoint);
}

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
                const x = markerData[xselector][d] * options.coordFactor;
                const y = markerData[yselector][d] * options.coordFactor;
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
        var xselector = options.xselector;
        var yselector = options.yselector;
    }else{
        throw {name : "NotImplementedError", message : "ViewerPointInPath not yet implemented."}; 
    }

    // FIXME: For now, regions will always have the first image as parent
    const image = tmapp["ISS_viewer"].world.getItemAt(0);
    const imageWidth = image ? image.getContentSize().x : 1;
    const imageHeight = image ? image.getContentSize().y : 1;
    const imageBounds = [0, 0, imageWidth, imageHeight];

    // Note: searchTreeForPointsInBbox() currently returns a list of points
    // in array-of-structs format. This will make the memory usage explode for
    // large markersets (or for markers with many attributes), so it would be
    // better to just return a list of point indices instead.
    const pointInBbox = regionUtils.searchTreeForPointsInBbox(quadtree, x0, y0, x3, y3, options);

    let countsInsideRegion = 0;
    let pointsInside = [];
    for (d of pointInBbox) {
        const x = d[xselector] * options.coordFactor;
        const y = d[yselector] * options.coordFactor;
        if (regionUtils._pointInRegion(x, y, regionid, imageBounds)) {
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
    glUtils._regionFillRule = glUtils._regionFillRule == "never" ? "nonzero" : "never";
    
    let regionIcon = document.getElementById('region_fill_button');
    if (glUtils._regionFillRule != "never") {
        regionIcon.classList.remove("btn-light");
        regionIcon.classList.add("btn-primary");
    } else {
        regionIcon.classList.remove("btn-primary");
        regionIcon.classList.add("btn-light");
    }
    glUtils.draw();    
}

/** Zoom to a set of regions */
regionUtils.zoomToRegions=function(regions){
    console.assert(regions.length > 0, "No regions to zoom to")
    // Get bounding box of all regions by looking at region._gxmin, region._gxmax, region._gymin, region._gymax
    let x0 = Infinity;
    let x3 = -Infinity;
    let y0 = Infinity;
    let y3 = -Infinity;
    for (const region of regions) {
        const image = tmapp["ISS_viewer"].world.getItemAt(region.collectionIndex);
        const viewportRect = image.imageToViewportRectangle(
            region._gxmin, region._gymin, region._gxmax - region._gxmin, region._gymax - region._gymin
        );
        x0 = Math.min(x0, viewportRect.x);
        y0 = Math.min(y0, viewportRect.y);
        x3 = Math.max(x3, viewportRect.x + viewportRect.width);
        y3 = Math.max(y3, viewportRect.y + viewportRect.height);
    }
    // Zoom to bounding box
    // Convert image coordinates to viewport coordinates
    const OSDRect = new OpenSeadragon.Rect(
        x0, y0, x3 - x0, y3 - y0
    );
    tmapp["ISS_viewer"].viewport.fitBounds(OSDRect);
}

/** 
 * @param {String} regionid String id of region to delete
 * @summary Given a region id, split region.globalPoints in separate regions */
regionUtils.splitRegion = function (regionid) {
    const region = regionUtils._regions[regionid];
    const globalPoints = region.globalPoints;
    const globalPointsLength = globalPoints.length;
    for (let i = 0; i < globalPointsLength; i++) {
        const newRegionId = regionid + "_" + i;
        regionUtils.addRegion(
            regionUtils.objectToArrayPoints(
                regionUtils.globalPointsToViewportPoints(
                    [globalPoints[i]], 
                    region.collectionIndex
                )
            ),
            newRegionId,
            region.polycolor,
            region.regionClass,
            region.collectionIndex
        );
    }
    regionUtils.deleteRegion(regionid);
}

/** 
 * @param {String} regionid String id of region to delete
 * @summary Given a region id, fill holes in region.globalPoints */
regionUtils.fillHolesRegion = function (regionid) {
    const region = regionUtils._regions[regionid];
    for (let i = 0; i < region.globalPoints.length; i++) {
        region.globalPoints[i] = [region.globalPoints[i][0]];
    }
    regionUtils.deSelectRegion(regionid);
    regionUtils.selectRegion(region);
}

/** 
 * @param {String} regionid String id of region to delete
 * @summary Given a region id, deletes this region in the interface */
regionUtils.deleteRegion = function (regionid, skipUpdateAllRegionClassUI) {
    delete regionUtils._regions[regionid];
    var op = tmapp["object_prefix"];
    var rPanel = document.getElementById(op + regionid + "_tr");
    if (rPanel) {
        rPanel.parentElement.removeChild(rPanel);
        var rPanelHist = document.getElementById(op + regionid + "_tr_hist");
        rPanelHist.parentElement.removeChild(rPanelHist);
    }
    regionUtils.deSelectRegion(regionid); 
    if (!skipUpdateAllRegionClassUI) regionUtils.updateAllRegionClassUI();
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
    regionUtils.updateAllRegionClassUI();
}
regionUtils.updateAllRegionClassUI = function () {
    setTimeout(()=>{
        // get the collapse status of all elements ".collapse_button_regionClass"
        // and save in a list of uncollapsed element ids}
        let uncollapsedElements = [];
        let collapseButtons = document.getElementsByClassName("collapse_button_regionClass");
        for (let i = 0; i < collapseButtons.length; i++) {
            if (collapseButtons[i].getAttribute("aria-expanded") == "true") {
                uncollapsedElements.push(collapseButtons[i].getAttribute("data-bs-target"));
            }
        }
        let regionUI = interfaceUtils._rGenUIFuncs.createTable();
        menuui=interfaceUtils.getElementById("markers-regions-panel");
        menuui.innerText="";

        menuui.appendChild(regionUI);
        // uncollapse all elements in uncollapsedElements:
        for (let i = 0; i < uncollapsedElements.length; i++) {
            // set style transition to none:
            $(uncollapsedElements[i]).css("transition", "none");
            $(uncollapsedElements[i]).collapse("show");
            // put back transition to default:
            $(uncollapsedElements[i]).css("transition", "");
        }
        menuui.classList.remove("d-none")
    },10);
    glUtils.updateRegionDataTextures();
    glUtils.updateRegionLUTTextures();
    glUtils.draw();
}

/** 
 *  @param {String} regionid Region identifier
 *  @summary Change the region properties like color, class name or region name */
regionUtils.changeRegion = function (regionid) {
    const escapedRegionID = HTMLElementUtils.stringToId(regionid)
    if (document.getElementById(regionid + "_name_ta")) {
        var op = tmapp["object_prefix"];
        var rPanel = document.getElementById(op + escapedRegionID + "_tr");
        var rPanel_hist = document.getElementById(op + escapedRegionID + "_tr_hist");
        if (regionUtils._regions[regionid].regionClass != document.getElementById(escapedRegionID + "_class_ta").value) {
            if (document.getElementById(escapedRegionID + "_class_ta").value) {
                regionUtils._regions[regionid].regionClass = document.getElementById(escapedRegionID + "_class_ta").value;
                //classID = HTMLElementUtils.stringToId(regionUtils._regions[regionid].regionClass);
                //regionUtils.addRegionClassUI (regionUtils._regions[regionid].regionClass)
                //$(rPanel).detach().appendTo('#markers-regions-panel-' + classID)
                //$(rPanel_hist).detach().appendTo('#markers-regions-panel-' + classID)
            } else {
                regionUtils._regions[regionid].regionClass = null;
                //regionUtils.addRegionClassUI (null)
                //classID = HTMLElementUtils.stringToId(regionUtils._regions[regionid].regionClass);
                //$(rPanel).detach().appendTo('#markers-regions-panel-')
                //$(rPanel_hist).detach().appendTo('#markers-regions-panel-')
            }
            regionUtils.updateAllRegionClassUI();
        }
        if (document.getElementById(escapedRegionID + "_name_ta").value) {
            regionUtils._regions[regionid].regionName = document.getElementById(escapedRegionID + "_name_ta").value;
        } else {
            regionUtils._regions[regionid].regionName = regionid;
        }
        var newregioncolor = document.getElementById(escapedRegionID + "_color_input").value;
        regionUtils._regions[regionid].polycolor = newregioncolor;
    }
}

/** 
 *  TODO */
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
                    "dataset":uid,
                    "coordFactor":dataUtils.data[uid]["_coord_factor"]
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
}
/** 
 *  regionUtils */
regionUtils.setMode = function (mode) {
    let selectButtonIcon = document.getElementById(
        "region_selection_button"
    );
    let mainButtonIcon = document.getElementById(
        "region_drawing_button"
    );
    let mainButtonIconDropdown = document.getElementById(
        "region_drawing_button_dropdown"
    );
    // Toggle off other region modes
    selectButtonIcon.classList.add("btn-light");
    selectButtonIcon.classList.remove("btn-primary");
    mainButtonIcon.classList.add("btn-light");
    mainButtonIcon.classList.remove("btn-primary");
    mainButtonIconDropdown.classList.add("btn-light");
    mainButtonIconDropdown.classList.remove("btn-primary");
    
    mode = (mode == regionUtils._regionMode) ? null : mode;

    regionUtils._regionMode = mode;
    if (mode == "select") {
        selectButtonIcon.classList.remove("btn-light");
        selectButtonIcon.classList.add("btn-primary");
    }
    else if (mode != null) {
        mainButtonIcon.classList.remove("btn-light");
        mainButtonIcon.classList.add("btn-primary");
        mainButtonIconDropdown.classList.remove("btn-light");
        mainButtonIconDropdown.classList.add("btn-primary");
    }
    if (mode == "points" || mode == "select" || mode == null) {
        // Reset cursor
        regionUtils.setViewerCursor("auto")
    }
    else if (mode == "brush") {
        // Set region drawing cursor
        regionUtils.setViewerCursor("none")
    }
    else if (mode == "free" || mode == "rectangle" || mode == "ellipse") {
        // Set region drawing cursor 
        regionUtils.setViewerCursor("crosshair")
    }
    regionUtils.resetManager();
}

regionUtils.freeHandManager = function (event) {
    function onCanvasRelease(){
        // Get OSDViewer
        const OSDViewer = tmapp[tmapp["object_prefix"] + "_viewer"];
        // Remove mouse dragging handler
        OSDViewer.removeHandler(
          "canvas-drag",
          createRegionFromCanvasDrag
        );
        // Remove release handler
        OSDViewer.removeHandler(
          "canvas-release",
          onCanvasRelease
        );
        // If there is only one point, there is no region to be drawn, 
        // reset and stop here
        if (!regionUtils._currentPoints) { 
          regionUtils.resetManager(); 
          return;
        }
        if (regionUtils._currentPoints.length < 2) { 
          regionUtils.resetManager(); 
          return;
        }
        // Close the region if initial point and final point are close enough
        if (
          regionUtils.distance(
            regionUtils._currentPoints[regionUtils._currentPoints.length - 1],
            regionUtils._currentPoints[0]
          ) <
          (10 * regionUtils._epsilonDistance) /
            tmapp["ISS_viewer"].viewport.getZoom()
        ) {
          regionUtils.closePolygon();
          return;
        }
        // If initial point and final point are not close enough, reset
        regionUtils.resetManager();
    }
    
      function createRegionFromCanvasDrag(event) {
        const drawingclass = regionUtils._drawingclass;
        // Get OSDViewer
        const OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];
        const canvas =
          overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
        // Block viewer panning
        event.preventDefaultAction = true;
        // Get region's next point coordinates from event position 
        const normCoords = OSDviewer.viewport.pointFromPixel(event.position);
        // Get stroke width depending on currently applied zoom to image
        const strokeWstr =
          regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom();
        let regionobj;
        if (regionUtils._isNewRegion) {
          regionUtils._currentPoints = [];
          regionUtils._isNewRegion = false;
          regionUtils._currentRegionId += 1;
          regionUtils._currentLayerIndex = regionUtils.getLayerFromCoord(normCoords);
          
          const idregion = regionUtils._currentRegionId;
          const startPoint = [normCoords.x, normCoords.y];
          regionUtils._currentPoints.push(startPoint);
          // Create a group to store region
          regionobj = d3.select(canvas).append("g").attr("class", drawingclass);
          // Draw a circle in the position of the first point of the region
          regionobj
            .append("circle")
            .attr(
              "r",
              (10 * regionUtils._handleRadius) /
                tmapp["ISS_viewer"].viewport.getZoom()
            )
            .attr("fill", regionUtils._colorActiveHandle)
            .attr("stroke", "#ff0000")
            .attr("stroke-width", strokeWstr)
            .attr("class", "region" + idregion)
            .attr("id", "handle-0-region" + idregion)
            .attr(
              "transform",
              "translate(" +
                startPoint[0].toString() +
                "," +
                startPoint[1].toString() +
                ") scale(" +
                regionUtils._scaleHandle +
                ")"
            )
            .attr("is-handle", "true")
            .style({ cursor: "pointer" });
            return 
        } 
        const idregion = regionUtils._currentRegionId;
        const nextpoint = [normCoords.x, normCoords.y];
        regionUtils._currentPoints.push(nextpoint);
        regionobj = d3.select("." + drawingclass);
        regionobj.select("polyline").remove();
        regionobj
          .append("polyline")
          .attr("points", regionUtils._currentPoints)
          .style("fill", "none")
          .attr("stroke-width", strokeWstr)
          .attr("stroke", "#ff0000")
          .attr("class", "region" + idregion);
        
    };  
    // Get OSDViewer
    const OSDViewer = tmapp[tmapp["object_prefix"] + "_viewer"];
    // Add region creation handler while mouse is pressed.
    // Capture the drag events to get the mouse position as the
    // left button.
    // Build the region based on the position of those events.
    OSDViewer.addHandler("canvas-drag", createRegionFromCanvasDrag);
    // Finish region drawing when mouse is released
    OSDViewer.addHandler("canvas-release", onCanvasRelease);
};
regionUtils.rectangleManager = function (event) {
    var last_rectangle = [];
    function onCanvasRelease(){
        // Get OSDViewer
        const OSDViewer = tmapp[tmapp["object_prefix"] + "_viewer"];
        // Remove mouse dragging handler
        OSDViewer.removeHandler(
          "canvas-drag",
          createRegionFromCanvasDrag
        );
        // Remove release handler
        OSDViewer.removeHandler(
          "canvas-release",
          onCanvasRelease
        );
        // If x1 == x2 or y1 == y2, there is no region to be drawn
        // reset and stop here
        if (last_rectangle[0] == last_rectangle[2] ||
            last_rectangle[1] == last_rectangle[3]) {
          regionUtils.resetManager(); 
          return;
        }
        // Close the region if initial point and final point are close enough
        regionUtils.closePolygon();
        // If initial point and final point are not close enough, reset
        regionUtils.resetManager();
    }
    
      function createRegionFromCanvasDrag(event) {
        const drawingclass = regionUtils._drawingclass;
        // Get OSDViewer
        const OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];
        const canvas =
          overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
        // Block viewer panning
        event.preventDefaultAction = true;
        // Get region's next point coordinates from event position 
        const normCoords = OSDviewer.viewport.pointFromPixel(event.position);
        // Get stroke width depending on currently applied zoom to image
        const strokeWstr =
          regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom();
        let regionobj;
        if (regionUtils._isNewRegion) {
          last_rectangle = [normCoords.x, normCoords.y, normCoords.x, normCoords.y];
          regionUtils._isNewRegion = false;
          regionUtils._currentRegionId += 1;
          regionUtils._currentLayerIndex = regionUtils.getLayerFromCoord(normCoords);
          regionobj = d3.select(canvas).append("g").attr("class", drawingclass);
        } 
        const idregion = regionUtils._currentRegionId;
        const nextpoint = [normCoords.x, normCoords.y];
        last_rectangle[2] = normCoords.x;
        last_rectangle[3] = normCoords.y;
        let boundingBox = last_rectangle;
        if (event.shift) {
            const width = Math.abs(boundingBox[0] - boundingBox[2]);
            const height = Math.abs(boundingBox[1] - boundingBox[3]);
            const maxSize = Math.max(width, height);
            boundingBox = [
                last_rectangle[0],
                last_rectangle[1],
                last_rectangle[0] + maxSize,
                last_rectangle[1] + maxSize
            ];
        }
        if (event.originalEvent.ctrlKey) {
            const width = Math.abs(boundingBox[0] - boundingBox[2]);
            const height = Math.abs(boundingBox[1] - boundingBox[3]);
            boundingBox = [
                last_rectangle[0] - width, 
                last_rectangle[1] - height,
                last_rectangle[0] + width,
                last_rectangle[1] + height
            ]
        }
        regionUtils._currentPoints = [
            [boundingBox[0], boundingBox[1]],
            [boundingBox[0], boundingBox[3]],
            [boundingBox[2], boundingBox[3]],
            [boundingBox[2], boundingBox[1]],
            [boundingBox[0], boundingBox[1]]
        ]
        
        regionobj = d3.select("." + drawingclass);
        regionobj.select("polyline").remove();
        regionobj
          .append("polyline")
          .attr("points", regionUtils._currentPoints)
          .style("fill", "none")
          .attr("stroke-width", strokeWstr)
          .attr("stroke", "#ff0000")
          .attr("class", "region" + idregion);
    };  
    // Get OSDViewer
    const OSDViewer = tmapp[tmapp["object_prefix"] + "_viewer"];
    // Add region creation handler while mouse is pressed.
    // Capture the drag events to get the mouse position as the
    // left button.
    // Build the region based on the position of those events.
    OSDViewer.addHandler("canvas-drag", createRegionFromCanvasDrag);
    // Finish region drawing when mouse is released
    OSDViewer.addHandler("canvas-release", onCanvasRelease);
};
regionUtils.ellipseManager = function (event) {
    var last_ellipse = [];
    function onCanvasRelease(){
        // Get OSDViewer
        const OSDViewer = tmapp[tmapp["object_prefix"] + "_viewer"];
        // Remove mouse dragging handler
        OSDViewer.removeHandler(
          "canvas-drag",
          createRegionFromCanvasDrag
        );
        // Remove release handler
        OSDViewer.removeHandler(
          "canvas-release",
          onCanvasRelease
        );
        // If x1 == x2 or y1 == y2, there is no region to be drawn
        // reset and stop here
        if (last_ellipse[0] == last_ellipse[2] ||
            last_ellipse[1] == last_ellipse[3]) {
          regionUtils.resetManager(); 
          return;
        }
        // Close the region if initial point and final point are close enough
        regionUtils.closePolygon();
        // If initial point and final point are not close enough, reset
        regionUtils.resetManager();
    }
    function generateEllipseCoordinates(last_ellipse, nb_points) {
        const [x1, y1, x2, y2] = last_ellipse;
        const centerX = (x1 + x2) / 2;
        const centerY = (y1 + y2) / 2;
        const radiusX = Math.abs(x2 - x1) / 2;
        const radiusY = Math.abs(y2 - y1) / 2;
      
        const coordinates = [];
      
        for (let i = 0; i <= nb_points; i++) {
          const angle = (i / nb_points) * 2 * Math.PI;
          const x = centerX + radiusX * Math.cos(angle);
          const y = centerY + radiusY * Math.sin(angle);
          coordinates.push([x, y]);
        }
      
        return coordinates;
    }
    
      function createRegionFromCanvasDrag(event) {
        const drawingclass = regionUtils._drawingclass;
        // Get OSDViewer
        const OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];
        const canvas =
          overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
        // Block viewer panning
        event.preventDefaultAction = true;
        // Get region's next point coordinates from event position 
        const normCoords = OSDviewer.viewport.pointFromPixel(event.position);
        // Get stroke width depending on currently applied zoom to image
        const strokeWstr =
          regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom();
        let regionobj;
        if (regionUtils._isNewRegion) {
          last_ellipse = [normCoords.x, normCoords.y, normCoords.x, normCoords.y];
          regionUtils._isNewRegion = false;
          regionUtils._currentRegionId += 1;
          regionUtils._currentLayerIndex = regionUtils.getLayerFromCoord(normCoords);
          regionobj = d3.select(canvas).append("g").attr("class", drawingclass);
        } 
        const idregion = regionUtils._currentRegionId;
        const nextpoint = [normCoords.x, normCoords.y];
        last_ellipse[2] = normCoords.x;
        last_ellipse[3] = normCoords.y;
        let boundingBox = last_ellipse;
        if (event.shift) {
            const width = Math.abs(boundingBox[0] - boundingBox[2]);
            const height = Math.abs(boundingBox[1] - boundingBox[3]);
            const maxSize = Math.max(width, height);
            boundingBox = [
                last_ellipse[0],
                last_ellipse[1],
                last_ellipse[0] + maxSize,
                last_ellipse[1] + maxSize
            ];
        }
        if (event.originalEvent.ctrlKey) {
            const width = Math.abs(boundingBox[0] - boundingBox[2]);
            const height = Math.abs(boundingBox[1] - boundingBox[3]);
            boundingBox = [
                last_ellipse[0] - width, 
                last_ellipse[1] - height,
                last_ellipse[0] + width,
                last_ellipse[1] + height
            ]
        }
        regionUtils._currentPoints = generateEllipseCoordinates(boundingBox, 50);
        
        regionobj = d3.select("." + drawingclass);
        regionobj.select("polyline").remove();
        regionobj
          .append("polyline")
          .attr("points", regionUtils._currentPoints)
          .style("fill", "none")
          .attr("stroke-width", strokeWstr)
          .attr("stroke", "#ff0000")
          .attr("class", "region" + idregion);
    };  
    // Get OSDViewer
    const OSDViewer = tmapp[tmapp["object_prefix"] + "_viewer"];
    // Add region creation handler while mouse is pressed.
    // Capture the drag events to get the mouse position as the
    // left button.
    // Build the region based on the position of those events.
    OSDViewer.addHandler("canvas-drag", createRegionFromCanvasDrag);
    // Finish region drawing when mouse is released
    OSDViewer.addHandler("canvas-release", onCanvasRelease);
};
regionUtils.brushHover = function (event) {
    // Get OSDViewer
    const OSDViewer = tmapp[tmapp["object_prefix"] + "_viewer"];
    const canvas =
      overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
    const drawingclass = "_brushRegion";
    const normCoords = OSDViewer.viewport.pointFromPixel(event.position);
    regionobj = d3.select("." + drawingclass);
    regionobj.remove();

    regionobj = d3.select(canvas).append("g").attr("class", drawingclass);
    // Draw a circle in the position of the first point of the region
    regionobj
    .append("circle")
    .attr(
        "r",
        (0.2 * regionUtils._handleRadius) /
        tmapp["ISS_viewer"].viewport.getZoom()
    )
    .attr("fill", "#ff000088")
    .attr("stroke", "#ff0000")
    .attr("stroke-width", 0)
    .attr("class", "regionBrush")
    .attr("id", "regionBrush")
    .attr(
        "transform",
        "translate(" +
        normCoords.x.toString() +
        "," +
        normCoords.y.toString() +
        ") scale(" +
        "1" +
        ")"
    )
    const regions = Object.values(regionUtils._selectedRegions);
    if (regions.length == 1) {
        // add a plus sign in a tspan next to the circle cursor
        regionobj
        .append("text")
        .attr(
            "transform",
            "translate(" +
            normCoords.x.toString() +
            "," +
            normCoords.y.toString() +
            ") scale(" +
            "1" +
            ")"
        )
        .attr("fill", "#000000")
        .attr("stroke", "#000000")
        .attr("stroke-width", 0)
        .attr("class", "regionBrush")
        .attr("id", "regionBrush")
        .attr("text-anchor", "middle")
        .attr("alignment-baseline", "middle")
        .attr("font-size", (0.02 /
            tmapp["ISS_viewer"].viewport.getZoom()).toString() + "px")
        .attr("font-weight", "bold")
        .text((event.originalEvent.shiftKey)?"-":"+");     
    }
    return 
}
regionUtils.brushManager = function (event) {
    function onCanvasRelease(){
        // Get OSDViewer
        const OSDViewer = tmapp[tmapp["object_prefix"] + "_viewer"];
        // Remove mouse dragging handler
        OSDViewer.removeHandler(
          "canvas-drag",
          createRegionFromCanvasDrag
        );
        // Remove release handler
        OSDViewer.removeHandler(
          "canvas-release",
          onCanvasRelease
        );
        // If there is only one point, there is no region to be drawn, 
        // reset and stop here
        if (!regionUtils._currentPoints) { 
          regionUtils.resetManager(); 
          return;
        }
        if (regionUtils._currentPoints.length < 1) { 
          regionUtils.resetManager(); 
          return;
        }
        // Close the region if initial point and final point are close enough
        var canvas = overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
        var drawingclass = regionUtils._drawingclass;
        var regionid = (regionUtils._editedRegion)?regionUtils._editedRegion.id:'region' + regionUtils._currentRegionId.toString();
        var regionclass = (regionUtils._editedRegion)?regionUtils._editedRegion.regionClass:'';
        d3.select("." + drawingclass).remove();
        regionsobj = d3.select(canvas);

        regionUtils._isNewRegion = true;
        regionUtils.addRegion(regionUtils._currentPoints, regionid, null, regionclass, regionUtils._currentLayerIndex);
        regionUtils._currentPoints = null;
    
        regionUtils.updateAllRegionClassUI();
        $(document.getElementById("regionClass-")).collapse("show");
    
        // If initial point and final point are not close enough, reset
        regionUtils.resetManager();
        if (regionUtils._editedRegion) {
            regionUtils.selectRegion(regionUtils._regions[regionid])
        }
        regionUtils.highlightRegion(regionid);
    }
    function getBrushShape(x1,y1,x2,y2,brushSize){
        // Get coordinates of the perimeter around two circles of radius brushSize
        // and centers in x1,y1 and x2,y2, merged with the rectangle joining the
        // two circles
        
        // First we get coordinates of circle 1, circle 2 and rectangle:
        // Circle 1
        
        const brushCircle1 = d3.range(0, 360, 20).map(function (t) {
            return {
                X:x1 + brushSize * Math.cos((t * Math.PI) / 180),
                Y:y1 + brushSize * Math.sin((t * Math.PI) / 180),
            };
        });
        // Circle 2
        const brushCircle2 = d3.range(0, 360, 20).map(function (t) {
            return {
                X:x2 + brushSize * Math.cos((t * Math.PI) / 180),
                Y:y2 + brushSize * Math.sin((t * Math.PI) / 180),
            };
        });
        // Rectangle of width 2*brushSize and length the distance between the two points
        // rotated to join perfectly the two circles
        const brushRectangle = [
            {
                X:x1 + brushSize * Math.cos(Math.atan2(y2 - y1, x2 - x1) - Math.PI / 2),
                Y:y1 + brushSize * Math.sin(Math.atan2(y2 - y1, x2 - x1) - Math.PI / 2),
            },
            {
                X:x1 + brushSize * Math.cos(Math.atan2(y2 - y1, x2 - x1) + Math.PI / 2),
                Y:y1 + brushSize * Math.sin(Math.atan2(y2 - y1, x2 - x1) + Math.PI / 2),
            },
            {
                X:x2 + brushSize * Math.cos(Math.atan2(y2 - y1, x2 - x1) + Math.PI / 2),
                Y:y2 + brushSize * Math.sin(Math.atan2(y2 - y1, x2 - x1) + Math.PI / 2),
            },
            {
                X:x2 + brushSize * Math.cos(Math.atan2(y2 - y1, x2 - x1) - Math.PI / 2),
                Y:y2 + brushSize * Math.sin(Math.atan2(y2 - y1, x2 - x1) - Math.PI / 2),
            }
        ];
        // Merge all of them
        const mergedPoints = regionUtils.clipperPolygons(
            [[[brushCircle1]],
            [[brushCircle2]],
            [[brushRectangle]]],
            "union"
          );
        return mergedPoints;
      }

      function createRegionFromCanvasDrag(event) {
        
        const drawingclass = regionUtils._drawingclass;
        // Get OSDViewer
        const OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];
        const canvas =
          overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
        // Block viewer panning
        event.preventDefaultAction = true;
        // Get region's next point coordinates from event position 
        const normCoords = OSDviewer.viewport.pointFromPixel(event.position);
        if (regionUtils._lastPoints) {
            if (regionUtils.distance([normCoords.x, normCoords.y], [regionUtils._lastPoints.x, regionUtils._lastPoints.y]) < regionUtils._epsilonDistance / tmapp["ISS_viewer"].viewport.getZoom()) {
                return;
            }
        }

        // Get stroke width depending on currently applied zoom to image
        const strokeWstr =
            regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom();
        let regionobj;
        const regions = Object.values(regionUtils._selectedRegions);
        
        if (regionUtils._isNewRegion) {
            if (regions.length == 1) {
                regionUtils._currentPoints = 
                regionUtils.objectToArrayPoints(
                    regionUtils.globalPointsToViewportPoints(
                        regions[0].globalPoints, 
                        regions[0].collectionIndex
                    )
                );
                regionUtils._isNewRegion = false;
                regionUtils._currentLayerIndex = regions[0].collectionIndex;
                regionUtils._editedRegion = regions[0];
                regionUtils.deleteRegion(regions[0].id, true);
            } else {
                regionUtils._currentPoints = [];
                regionUtils._isNewRegion = false;
                regionUtils._currentRegionId += 1;
                regionUtils._editedRegion = null;
                regionUtils._currentLayerIndex = regionUtils.getLayerFromCoord(normCoords);
            }
            regionUtils._lastPoints = normCoords
        } 
        const idregion = regionUtils._currentRegionId;
        const operation = (!event.shift) ? "union" : "difference";
        const previousPolygon = regionUtils.regionToUpperCase(
            regionUtils.arrayToObjectPoints(regionUtils._currentPoints)
        );
        const brushShape = getBrushShape(
            regionUtils._lastPoints.x,
            regionUtils._lastPoints.y,
            normCoords.x,
            normCoords.y,
            (0.2 * regionUtils._handleRadius) /
                tmapp["ISS_viewer"].viewport.getZoom()
        );
        if (previousPolygon.length == 0) {
            regionUtils._currentPoints = regionUtils.objectToArrayPoints(
                regionUtils.regionToLowerCase(brushShape)
            )
        }
        else {
            regionUtils._currentPoints = regionUtils.objectToArrayPoints(
                regionUtils.regionToLowerCase(
                    regionUtils.clipperPolygons(
                        [
                            previousPolygon,
                            brushShape
                        ], operation
                    )
                )
            );
        }
        regionUtils._lastPoints = normCoords;
        regionobj = d3.select("." + drawingclass);
        regionobj.remove();
        regionobj = d3.select(canvas).append("g").attr("class", drawingclass);
        regionobj
        .append("path")
        .attr("d", regionUtils.pointsToPath(regionUtils.arrayToObjectPoints(regionUtils._currentPoints)))
        .style("fill", "none")
        .attr("stroke-width", strokeWstr)
        .attr("stroke", "#ff0000")
        .attr("class", "region" + idregion);
    };
    // Get OSDViewer
    const OSDViewer = tmapp[tmapp["object_prefix"] + "_viewer"];
    // Add region creation handler while mouse is pressed.
    // Capture the drag events to get the mouse position as the
    // left button.
    // Build the region based on the position of those events.
    OSDViewer.addHandler("canvas-drag", createRegionFromCanvasDrag);
    // Finish region drawing when mouse is released
    OSDViewer.addHandler("canvas-release", onCanvasRelease);
    createRegionFromCanvasDrag(event);
};

/**
 * 
 * @param {string} cursorType 
 * @summary Set the OSD Viewer cursor type 
 */
regionUtils.setViewerCursor = function(cursorType){
    // Get OSDViewer HTML element
    const OSDViewerElement = tmapp[tmapp["object_prefix"] + "_viewer"].element
    // Set the cursor type 
    OSDViewerElement.style.cursor = cursorType
}

/** 
 *  regionUtils */
regionUtils.exportRegionsToJSON = function () {
    regionUtils.regionsToJSON();
}
/** 
 *  regionUtils */
regionUtils.importRegionsFromJSON = function () {
    regionUtils.JSONToRegions();
}

regionUtils.pointsInRegionsToCSV= async function(){
    /* we loop through all regions in regionUtils._regions and compute the points in each region
    using regionUtils.analyzeRegion.
    */
    let analyzeAll = await interfaceUtils.confirm(
        "Do you want to run the analysis on all regions? This may take a while.<br/><br/> \
        If not, the exported file will only contain previously analyzed regions.",
        "Analyze all regions?"
    )
    if (analyzeAll) { 
        for (let r in regionUtils._regions){
            regionUtils.analyzeRegion(r);
        }
    }
    var alldata=[]
    for (let r in regionUtils._regions){
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

regionUtils.JSONValToRegions= async function(jsonVal){
    // The file's text will be printed here
    var regions=jsonVal;
    await regionUtils.geoJSON2regions(regions);
    regionUtils.updateAllRegionClassUI();
    $('[data-bs-target="#markers-regions-project-gui"]').tab('show');
}


// Build data structure for rendering region objects. The basic idea is to
// divide the image region into scanlines, and bin edges from polygons into
// those scanlines. Edges within scanlines will be ordered by object IDs.
regionUtils._generateEdgeListsForDrawing = function(imageBounds, numScanlines = 512) {
    console.assert(imageBounds.length == 4);
    const scanlineHeight = imageBounds[3] / numScanlines;

    regionUtils._edgeLists = [];
    for (let i = 0; i < numScanlines; ++i) {
        regionUtils._edgeLists[i] = [[0, 0, 0, 0, 0, 0, 0, 0], 0];
    }

    regionUtils._regionIDToIndex = {};
    regionUtils._regionIndexToID = {};

    let objectID = 0;
    for (let regionID of Object.keys(regionUtils._regions)) {
        const region = regionUtils._regions[regionID];
        regionUtils._regionIDToIndex[regionID] = objectID;  // Update mapping
        regionUtils._regionIndexToID[objectID] = regionID;  // ...

        for (let subregion of region.globalPoints) {
            for (let points of subregion) {
                const numPoints = points.length;
                if (numPoints <= 1) continue;

                // Compute axis-aligned bounding box for this path
                let xMin = 99999, xMax = -99999, yMin = 99999, yMax = -99999;
                for (let i = 0; i < numPoints; ++i) {
                    const v = points[i];
                    xMin = Math.min(xMin, v.x);
                    xMax = Math.max(xMax, v.x);
                    yMin = Math.min(yMin, v.y);
                    yMax = Math.max(yMax, v.y);
                }

                // Rasterise bounding box into bitmask used for updating occupancy mask
                // that will be stored in the first texel of each scanline during rendering
                let mask = [0, 0, 0, 0];
                const lsb = Math.max(0, Math.min(63, Math.floor(xMin * (64.0 / imageBounds[2]))));
                const msb = Math.max(0, Math.min(63, Math.floor(xMax * (64.0 / imageBounds[2]))));
                for (let bitIndex = lsb; bitIndex <= msb; ++bitIndex) {
                    mask[(bitIndex >> 4)] |= (1 << (bitIndex & 15));
                }

                // Create header elements for storing information about the number
                // of edges for path in overlapping scanlines. We also want to store
                // some other information such as bounding box and parent object ID.
                const lower = Math.max(Math.floor(yMin / scanlineHeight), 0);
                const upper = Math.min(Math.floor(yMax / scanlineHeight), numScanlines - 1);
                for (let i = lower; i <= upper; ++i) {
                    const headerOffset = regionUtils._edgeLists[i][0].length;
                    regionUtils._edgeLists[i][0].push(xMin, xMax, objectID + 1, 0);
                    regionUtils._edgeLists[i][1] = headerOffset;
                    for (let j = 0; j < 4; ++j) {
                        regionUtils._edgeLists[i][0][j] |= mask[j];  // Update occupancy mask
                    }
                }

                // Create elements for storing vertex pairs for edges in overlapping scanlines
                for (let i = 0; i < numPoints; ++i) {
                    const v0 = points[(i + 0) % numPoints];
                    const v1 = points[(i + 1) % numPoints];
                    if (v0.x == v1.x && v0.y == v1.y) { continue; }

                    const lower = Math.max(Math.floor(Math.min(v0.y, v1.y) / scanlineHeight), 0);
                    const upper = Math.min(Math.floor(Math.max(v0.y, v1.y) / scanlineHeight), numScanlines - 1);
                    for (let j = lower; j <= upper; ++j) {
                        const headerOffset = regionUtils._edgeLists[j][1];
                        regionUtils._edgeLists[j][0].push(v0.x, v0.y, v1.x, v1.y);
                        regionUtils._edgeLists[j][0][headerOffset + 3] += 1;  // Update edge counter
                    }
                }
            }
        }
        objectID += 1;
    }
}


// Split each individual edge list around a pivot point into two new lists
regionUtils._splitEdgeLists = function() {
    const numScanlines = regionUtils._edgeLists.length;

    regionUtils._edgeListsSplit = [];
    for (let i = 0; i < numScanlines; ++i) {
        regionUtils._edgeListsSplit[i] = [[], []];
    }

    for (let i = 0; i < numScanlines; ++i) {
        const edgeList = regionUtils._edgeLists[i][0];
        const numItems = edgeList.length / 4;

        // Find pivot point (mean center of all bounding boxes) for left-right split
        let accum = [0.0, 0.0];
        for (let j = 2; j < numItems; ++j) {
            const xMin = edgeList[j * 4 + 0];
            const xMax = edgeList[j * 4 + 1];
            const edgeCount = edgeList[j * 4 + 3];

            accum[0] += (xMin + xMax) * 0.5; accum[1] += 1;
            j += edgeCount;  // Position pointer before next bounding box
        }
        const pivot = accum[0] / Math.max(1, accum[1]);

        // Copy occupancy mask, and also store the pivot point
        for (let n = 0; n < 2; ++n) {
            regionUtils._edgeListsSplit[i][n].push(...edgeList.slice(0, 4));
            regionUtils._edgeListsSplit[i][n].push(pivot, 0, 0, 0);
        }

        // Do left-right split of edge data
        for (let j = 2; j < numItems; ++j) {
            const xMin = edgeList[j * 4 + 0];
            const xMax = edgeList[j * 4 + 1];
            const edgeCount = edgeList[j * 4 + 3];

            if (xMin < pivot) {
                regionUtils._edgeListsSplit[i][0].push(
                    ...edgeList.slice(j * 4, (j + edgeCount + 1) * 4));
            }
            if (xMax > pivot) {
                regionUtils._edgeListsSplit[i][1].push(
                    ...edgeList.slice(j * 4, (j + edgeCount + 1) * 4));
            }
            j += edgeCount;  // Position pointer before next bounding box
        }
    }
}


// Add cluster information to edge lists (WIP)
regionUtils._addClustersToEdgeLists = function(imageBounds) {
    // STUB
}


// Check if point is inside or outside a region, by computing the winding
// number for paths in a scanline of the edge list data structure
regionUtils._pointInRegion = function(px, py, regionID, imageBounds) {
    console.assert(imageBounds.length == 4);
    const numScanlines = regionUtils._edgeLists.length;
    const scanlineHeight = imageBounds[3] / numScanlines;
    const scanline = Math.floor(py / scanlineHeight);

    const objectID = regionUtils._regionIDToIndex[regionID];

    let isInside = false;
    if (scanline >= 0 && scanline < numScanlines) {
        const edgeList = regionUtils._edgeLists[scanline][0];
        const numItems = edgeList.length / 4;

        // Traverse edge list until we find first path for object ID
        let offset = 2;  // Offset starts at two because of occupancy mask
        while (offset < numItems && (edgeList[offset * 4 + 2] - 1) != objectID) {
            const count = edgeList[offset * 4 + 3];
            offset += count + 1;
        }

        // Compute winding number from all edges with stored for the object ID
        let windingNumber = 0;
        while (offset < numItems && (edgeList[offset * 4 + 2] - 1) == objectID) {
            const count = edgeList[offset * 4 + 3];
            for (let i = 0; i < count; ++i) {
                const x0 = edgeList[(offset + 1 + i) * 4 + 0];
                const y0 = edgeList[(offset + 1 + i) * 4 + 1];
                const x1 = edgeList[(offset + 1 + i) * 4 + 2];
                const y1 = edgeList[(offset + 1 + i) * 4 + 3];

                if (Math.min(y0, y1) <= py && py < Math.max(y0, y1)) {
                    const t = (py - y0) / (y1 - y0 + 1e-5);
                    const x = x0 + (x1 - x0) * t;
                    const weight = Math.sign(y1 - y0);
                    windingNumber += ((x - px) > 0.0 ? weight : 0);
                }
            }
            offset += count + 1;  // Position pointer at next path
        }

        // Apply non-zero fill rule for inside test
        isInside = windingNumber != 0;
    }
    return isInside;
}


// Find region under point. Returns a key to the regionUtils_regions dict if a
// region is found, otherwise null. If multiple regions overlap at the point,
// the key of the last one in the draw order shall be returned.
regionUtils._findRegionByPoint = function(px, py, imageBounds) {
    console.assert(imageBounds.length == 4);
    const numScanlines = regionUtils._edgeLists.length;
    const scanlineHeight = imageBounds[3] / numScanlines;
    const scanline = Math.floor(py / scanlineHeight);

    if (scanline < 0 || scanline >= numScanlines) return null;  // Outside image
    const edgeList = regionUtils._edgeLists[scanline][0];
    const numItems = edgeList.length / 4;

    let offset = 2;  // Offset starts at two because of occupancy mask
    let objectID = offset < numItems ? (edgeList[offset * 4 + 2] - 1) : -1;

    let foundRegion = -1;
    while (offset < numItems) {
        console.assert(regionUtils._regionToColorLUT.length > (objectID * 4));
        const visible = regionUtils._regionToColorLUT[objectID * 4 + 3];

        // (TODO Add bounding box test to check if object can be skipped)

        // Compute winding number from all edges stored for the object ID
        let windingNumber = 0;
        while (offset < numItems && (edgeList[offset * 4 + 2] - 1) == objectID) {
            const count = edgeList[offset * 4 + 3];
            for (let i = 0; i < count; ++i) {
                const x0 = edgeList[(offset + 1 + i) * 4 + 0];
                const y0 = edgeList[(offset + 1 + i) * 4 + 1];
                const x1 = edgeList[(offset + 1 + i) * 4 + 2];
                const y1 = edgeList[(offset + 1 + i) * 4 + 3];

                if (Math.min(y0, y1) <= py && py < Math.max(y0, y1)) {
                    const t = (py - y0) / (y1 - y0 + 1e-5);
                    const x = x0 + (x1 - x0) * t;
                    const weight = Math.sign(y1 - y0);
                    windingNumber += ((x - px) > 0.0 ? weight : 0);
                }
            }
            offset += count + 1;  // Position pointer at next path
        }

        // Apply non-zero fill rule for inside test
        const isInside = windingNumber != 0;
        if (isInside && visible) { foundRegion = objectID; }

        objectID = offset < numItems ? (edgeList[offset * 4 + 2] - 1) : -1;
    }
    return foundRegion >= 0 ? regionUtils._regionIndexToID[foundRegion] : null;
}


// Build lookup table used during render time for mapping object IDs generated
// by regionUtils._generateEdgeListsForDrawing() to color and visibility
regionUtils._generateRegionToColorLUT = function() {
    regionUtils._regionToColorLUT = [];

    let objectID = 0;
    for (let region of Object.values(regionUtils._regions)) {
        const hexColor = region.polycolor;
        const r = Number("0x" + hexColor.substring(1,3));
        const g = Number("0x" + hexColor.substring(3,5));
        const b = Number("0x" + hexColor.substring(5,7));
        const visibility = region.visibility ? 255 : 0;
        regionUtils._regionToColorLUT.push(r, g, b, visibility);
        objectID += 1;
    }
    console.assert(regionUtils._regionToColorLUT.length == (objectID * 4));
}
