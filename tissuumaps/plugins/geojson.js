/**
 * @file geojson.js
 * @author Christophe Avenel
 */

/**
 * @namespace geojson
 * @classdesc The root namespace for geojson.
 */
var geojson;
geojson = {
    functions:[
        {
            name:"Load geoJSON regions",
            function:"loadGeoJSON"
        }
    ],
    _bboxSize:11,
    _order_rounds:null,
    _order_channels:null
 }

/**
 * This method is called when the document is loaded. The tmapp object is built as an "app" and init is its main function.
 * Creates the OpenSeadragon (OSD) viewer and adds the handlers for interaction.
 * To know which data one is referring to, there are Object Prefixes (op). For In situ sequencing projects it can be "ISS" for
 * Cell Profiler data it can be "CP".
 * If there are images to be displayed on top of the main image, they are stored in the layers object and, if there are layers
 * it will create the buttons to display them in the settings panel.
 * The SVG overlays for the viewer are also initialized here 
 * @summary After setting up the tmapp object, initialize it*/
geojson.init = function (tmappObject) {
    geojson.tmapp = tmappObject;
    geojson.functions.forEach(function(funElement, i) {
        var aElement = document.createElement("a");
        aElement.href = "#";
        aElement.addEventListener("click",function (event) {
            console.log("Click", event, funElement.function);
            window["geojson"][funElement.function]();
        });
        var spanElement = document.createElement("span");
        aElement.appendChild(spanElement);
        spanElement.innerHTML = funElement.name;
        dropdownMenu = document.getElementById("dropdown-menu-geojson");
        dropdownMenu.appendChild(aElement);
    });
}

geojson.loadGeoJSON = function () {
    var input = document.createElement('input');
    input.type = 'file';
    input.click();
    input.onchange = e => { 
        var file = e.target.files[0]; 
        // setting up the reader
        var reader = new FileReader();
        // here we tell the reader what to do when it's done reading...
        reader.onload = readerEvent => {
            var content = readerEvent.target.result; // this is the content!
            geoJSONObj = JSON.parse(reader.result)
            geojson.geoJSON2regions(geoJSONObj)
        }
        reader.readAsText(file,'UTF-8');
    }
}

function decimalToHexString(number)
{
  if (number < 0)
  {
    number = 0xFFFFFFFF + number + 1;
  }

  return "#" + number.toString(16).toUpperCase().substring(2, 8);
}

geojson.geoJSON2regions = function (geoJSONObjects) {
    var imageHeight = OSDViewerUtils.getImageHeight();

    var viewer = tmapp[tmapp["object_prefix"] + "_viewer"]
    var canvas = overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
    if (!Array.isArray(geoJSONObjects)) {
        geoJSONObjects = [geoJSONObjects];
    }
    geoJSONObjects.forEach(function(geoJSONObj, geoJSONObjIndex) {
        var geometryType = geoJSONObj.geometry.type;
        var coordinates;
        if (geometryType=="Polygon") {
            coordinates = [geoJSONObj.geometry.coordinates];
        }
        else {
            coordinates = geoJSONObj.geometry.coordinates;
        }
        var geoJSONObjClass = "";
        var hexColor = "#ff0000";
        if (geoJSONObj.properties.classification) {
            geoJSONObjClass = geoJSONObj.properties.classification.name;
            hexColor = decimalToHexString(geoJSONObj.properties.classification.colorRGB);
        }
        console.log(hexColor);
        coordinates = coordinates.map (function(coordinateList, i) {
            return coordinateList.map (function(coordinateList_i, index) {
                coordinateList_i = coordinateList_i.map(function(x) {
                    xPoint = new OpenSeadragon.Point(x[0], x[1]);
                    xPixel = viewer.world.getItemAt(0).imageToViewportCoordinates(xPoint);
                    return [xPixel.x, xPixel.y];
                });
                return coordinateList_i.filter(function(value, index, Arr) {
                    return index % 4 == 0;
                });
            });
        })
        console.log(coordinates);
        var regionId = "Region_geoJSON_" + geoJSONObjIndex;
        regionUtils.addRegion(coordinates, regionId, hexColor);

        document.getElementById(regionId + "class_ta").value = geoJSONObjClass;
        document.getElementById(regionId + "name_ta").value = "geojson_" + (geoJSONObjIndex - -1);

        regionobj = d3.select(canvas).append('g').attr('class', "mydrawingclass");
        regionobj.append('path').attr("d", regionUtils.pointsToPath(regionUtils._regions[regionId].points)).attr("id", regionId + "poly")
        .attr("class", "regionpoly").attr("polycolor", hexColor).style('stroke-width', regionUtils._polygonStrokeWidth.toString())
        .style("stroke", hexColor).style("fill", "#FFFFFF00");
        
        regionUtils.changeRegion(regionId);
        //regionUtils.analyzeRegion("Region_" + i);
    });
}