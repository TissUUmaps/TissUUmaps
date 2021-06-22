/**
 * @file transform.js
 * @author Christophe Avenel
 */

/**
 * @namespace transform
 * @classdesc The root namespace for transform.
 */
var transform;
transform = {
    functions:[
        {
            name:"Transform image",
            function:"transformImage"
        },
        {
            name:"Resize image",
            function:"resizeImage"
        },
        /*{
            name:"Transform markers",
            function:"transformMarkers"
        },*/
        {
            name:"Transform regions",
            function:"transformRegions"
        }
    ]
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
transform.init = function (tmappObject) {
    transform.tmapp = tmappObject;
    transform.functions.forEach(function(funElement, i) {
        var aElement = document.createElement("a");
        aElement.href = "#";
        aElement.addEventListener("click",function (event) {
            console.log("Click", event, funElement.function);
            window["transform"][funElement.function]();
        });
        var spanElement = document.createElement("span");
        aElement.appendChild(spanElement);
        spanElement.innerHTML = funElement.name;
        dropdownMenu = document.getElementById("dropdown-menu-transform");
        dropdownMenu.appendChild(aElement);
    });
}

transform.transformImage = function () {
    console.log("Transform Image");
    var op = transform.tmapp["object_prefix"];
    var vname = op + "_viewer";
    if (tmapp.layers.length == 1) {
        selectedLayer = 0;
    }
    else {
        layers = tmapp.layers.map(({ name }, index) => (index - -1) + ". "+ name).join("\n");
        selectedLayer = parseInt(prompt("Which layer do you want to transform (1 - " + tmapp.layers.length + ") ?\n" + layers,"1"))-1;
    }
    console.log("selectedLayer", selectedLayer, tmapp.layers[selectedLayer].tileSource);
    matrixString=prompt("Please paste your transformation matrix here, as six comma separated numbers:","1,0,0,1,0,0");
    matrix = matrixString.match(/\d+(?:\.\d+)?/g).map(Number)
    console.log(matrix);
    console.log(tmapp.fixed_file);
    $("#loadingModal").show();
    $.ajax(
        {
            // Post select to url.
            type : 'post',
            url : '/plugin/transform/transform',
            contentType: 'application/json; charset=utf-8',
            data : JSON.stringify({
                    'path' : tmapp.layers[selectedLayer].tileSource,
                    'outputSuffix':"_transformed",
                    'matrix': matrix //[2.0072846463943876, -0.03895444531413667, -0.05744089589832094, -1.9617630620838924, -137.7703410591635*8, 6284.0522698317745*8]//[0.4978627682471971, -0.009762949861712243, -0.014492830616151765, -0.5089427682566419, 159.98505755852736*8., 3198.1725416545833*8.  ]
            }),
            success : function(data)
            {
                $("#loadingModal").hide();
                transform.loadState(data);
            },
            complete : function(data)
            {
                // do something, not critical.
            },
            error:function (data)
            {
                console.log("Error:", data);
            }
        }
    );
}

transform.transformRegions = function () {
    console.log("Transform Regions");
    var op = transform.tmapp["object_prefix"];
    var vname = op + "_viewer";
    var canvas = overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
    var imageWidth = OSDViewerUtils.getImageWidth();
	
    matrixString=prompt("Please paste your transformation matrix here, as six comma separated numbers:","1,0,0,1,0,0");
    matrix = matrixString.match(/\d+(?:\.\d+)?/g).map(Number)
    for(var region in regionUtils._regions){
		if (regionUtils._regions.hasOwnProperty(region)) {
			regionUtils._regions[region].globalPoints.forEach(function (subregions) {
                subregions.forEach(function (polygons) {
                    var first = true
                    polygons.forEach(function (point) {
                        point.x = point.x * matrix[0] + point.y * matrix[1] + matrix[4];
                        point.y = point.x * matrix[2] + point.y * matrix[3] + matrix[5];
                    });
                });
            });
        regionUtils._regions[region].points.forEach(function (subregions) {
                subregions.forEach(function (polygons) {
                    var first = true
                    polygons.forEach(function (point) {
                        point.x = point.x * matrix[0] + point.y * matrix[1] + matrix[4] / imageWidth;
                        point.y = point.x * matrix[2] + point.y * matrix[3] + matrix[5] / imageWidth;
                    });
                });
            });
		}
        var hexColor = "#ff0000";
        regionobj = d3.select(canvas).append('g').attr('class', "mydrawingclass");
        regionobj.append('path').attr("d", regionUtils.pointsToPath(regionUtils._regions[region].points)).attr("id", region + "poly")
        .attr("class", "regionpoly").attr("polycolor", hexColor).style('stroke-width', regionUtils._polygonStrokeWidth.toString())
        .style("stroke", hexColor).style("fill", "#FFFFFF00");
	}
}

transform.resizeImage = function () {
    console.log("Transform Image");
    var op = transform.tmapp["object_prefix"];
    var vname = op + "_viewer";
    console.log(tmapp.fixed_file);
    factorString =prompt("Please write your scaling factor:","8");
    factor = parseFloat(factorString);
    $("#loadingModal").show();
    $.ajax(
        {
            // Post select to url.
            type : 'post',
            url : '/plugin/transform/transform',
            contentType: 'application/json; charset=utf-8',
            data : JSON.stringify({
                    'path' : tmapp.fixed_file,
                    'outputSuffix':"_resized",
                    'matrix': [1./factor, 0, 0, 1./factor, 0, 0],
            }),
            success : function(data)
            {
                $("#loadingModal").hide();
                transform.loadState(data);
            },
            complete : function(data)
            {
                // do something, not critical.
            },
            error:function (data)
            {
                console.log("Error:", data);
            }
        }
    );
}

/**
 * This method is used to load the TissUUmaps state (gene expression, cell morphology, regions) */
transform.loadState = function(state) {
    tmapp.layers.push({
        name: "Transformed layer",
        tileSource: state["image"] + ".dzi"
    });
    i = tmapp.layers.length - 1;
    overlayUtils.addLayer("Transformed layer", state["image"] + ".dzi", i);
    overlayUtils.addAllLayersSettings();
}
