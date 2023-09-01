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
 * @property {Object[]} regionUtils._edgeLists - Data structure used for rendering regions with WebGL
 * @property {Object[]} regionUtils._regionToColorLUT - LUT for storing color and visibility per object ID
 * @property {Object{}} regionUtils._regionIDToIndex - Mapping between region ID (string) and object ID (index)
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
    _maxRegionsInMenu: 200,
    _edgeLists: [],
    _regionToColorLUT: [],
    _regionIDToIndex: {}
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

    var hexcolor = "#FF0000"; //overlayUtils.randomColor("hex");    

    regionUtils._isNewRegion = true;
    regionUtils._currentPoints.push(regionUtils._currentPoints[0]);
    regionUtils.addRegion([[regionUtils._currentPoints]], regionid, hexcolor);
    regionUtils._currentPoints = null;

    regionUtils.updateAllRegionClassUI();
    if(overlayUtils._regionOperations){
        regionUtils.addRegionOperationsRow(regionid)
    }
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
        regionUtils.addRegion(coordinates, regionId, hexColor, geoJSONObjClass);
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
regionUtils.addRegion = function (points, regionid, color, regionClass) {
    if (!regionClass) regionClass = "";
    var op = tmapp["object_prefix"];
    var viewer = tmapp[tmapp["object_prefix"] + "_viewer"]
    //var imageWidth = OSDViewerUtils.getImageWidth();
    var region = { 
        "id": regionid, 
        "points": [], 
        "globalPoints": [], 
        "regionName": regionid, 
        "regionClass": regionClass, 
        "barcodeHistogram": [],
        "visibility": true
    };
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
        const x = d[xselector];
        const y = d[yselector];
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
    
    let regionIcon = document.getElementById('ISS_fillregions_icon');
    if (glUtils._regionFillRule != "never") {
        regionIcon.classList.remove("bi-circle");
        regionIcon.classList.add("bi-check-circle");
    } else {
        regionIcon.classList.remove("bi-check-circle");
        regionIcon.classList.add("bi-circle");
    }
    glUtils.draw();    
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
    if(!overlayUtils._regionOperations) return; 
    regionUtils.deleteRegionOperationRows(regionid);  
    if(!regionUtils._selectedRegions[regionid]) return; 
    const regionClass = regionUtils._selectedRegions[regionid].regionClass;
    delete regionUtils._selectedRegions[regionid];
    const remainingClassRegions = Object.values(regionUtils._regions).filter((region) => region.regionClass === regionClass);
    if(remainingClassRegions.length === 0){
        regionUtils.deleteRegionOperationsAccordion(regionClass);
    }
    regionUtils.updateAllRegionClassUI();
    //if(!skipUpdateAllRegionClassUI) {
    //    regionUtils.updateAllRegionClassUI();
    //}
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
regionUtils.updateAllRegionClassUI = function () {
    setTimeout(()=>{
        let regionUI = interfaceUtils._rGenUIFuncs.createTable();
        menuui=interfaceUtils.getElementById("markers-regions-panel");
        menuui.classList.remove("d-none")
        menuui.innerText="";

        menuui.appendChild(regionUI);
    },10);
    glUtils.updateRegionDataTextures();
    glUtils.updateRegionLUTTextures();
    glUtils.draw();
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
        if (document.getElementById(regionid + "_name_ta").value) {
            regionUtils._regions[regionid].regionName = document.getElementById(regionid + "_name_ta").value;
        } else {
            regionUtils._regions[regionid].regionName = regionid;
        }
        var newregioncolor = document.getElementById(regionid + "_color_input").value;
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
    // Toggle off other region modes
    if (overlayUtils._freeHandDrawRegions) {
        regionUtils.freeHandRegionsOnOff();
    }
    overlayUtils._drawRegions = !overlayUtils._drawRegions;
    var op = tmapp["object_prefix"];
    let regionIcon = document.getElementById(op + '_drawregions_icon');
    if (overlayUtils._drawRegions) {
        regionIcon.classList.remove("bi-circle");
        regionIcon.classList.add("bi-check-circle");
        // Set region drawing cursor and show hint
        regionUtils.setViewerCursor("crosshair")
        regionUtils.showHint("Click to draw regions")
    } else {
        regionUtils.resetManager();
        regionIcon.classList.remove("bi-check-circle");
        regionIcon.classList.add("bi-circle");
         // Reset cursor and hide hint
        regionUtils.setViewerCursor("auto")
        regionUtils.hideHint();
    }
}

regionUtils.freeHandRegionsOnOff = function () {
    // Toggle off other region modes
    if (overlayUtils._drawRegions) {
        regionUtils.regionsOnOff();
    }
    overlayUtils._freeHandDrawRegions = !overlayUtils._freeHandDrawRegions;
    const op = tmapp["object_prefix"];
    let freeHandButtonIcon = document.getElementById(
        op + "_draw_regions_free_hand_icon"
    );
    if (overlayUtils._freeHandDrawRegions) {
        freeHandButtonIcon.classList.remove("bi-circle");
        freeHandButtonIcon.classList.add("bi-check-circle");
        // Set region drawing cursor and show hint
        regionUtils.setViewerCursor("crosshair");
        regionUtils.showHint("Drag the mouse to draw regions");
    } else {
        regionUtils.resetManager();
        freeHandButtonIcon.classList.remove("bi-check-circle");
        freeHandButtonIcon.classList.add("bi-circle");
        // Reset cursor and hide hint
        regionUtils.setViewerCursor("auto");
        regionUtils.hideHint();
    }
};

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
        if (regionUtils._currentPoints < 1) { 
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
 * 
 * @param {string} message 
 * @summary Show a hint in the regions tab  
 */
regionUtils.showHint = function(message){
    // Get region buttons container
    const regionsButtonsContainer = document.getElementById("regionButtons")
    // Check if banner is already visible, if not, create it 
    let hintBanner = document.getElementById("regionHintBanner")
    if(!hintBanner) {
        hintBanner = document.createElement("div")
        hintBanner.setAttribute("id", "regionHintBanner")
    } 
    // Set banner styles
    hintBanner.innerText = message
    hintBanner.style.width = "100%"
    hintBanner.style.textAlign = "center"
    hintBanner.style.background = "rgba(239,239,240, 1)"
    hintBanner.style.padding = "8px 0 8px 0"
    hintBanner.style.margin = "8px 0 8px 0"
    hintBanner.style.color = "green"
    // Add banner to region buttons container
    regionsButtonsContainer.append(hintBanner)
}

/**
 * 
 * @summary Hide regions tab hint 
 */
regionUtils.hideHint = function(){
    // Get hint element
    const hintBanner = document.getElementById("regionHintBanner")
    // If banner does not exist, return
    if(!hintBanner) return 
    // Remove hint element
    hintBanner.remove()
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

    let objectID = 0;
    for (let regionID of Object.keys(regionUtils._regions)) {
        const region = regionUtils._regions[regionID];
        regionUtils._regionIDToIndex[regionID] = objectID;  // Update mapping

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
