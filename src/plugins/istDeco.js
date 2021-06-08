/**
 * @file istDeco.js
 * @author Christophe Avenel
 */

/**
 * @namespace istDeco
 * @classdesc The root namespace for istDeco.
 */
var istDeco;
istDeco = {
    functions:[
        {
            name:"Load images",
            function:"loadImages"
        },
        {
            name:"Start plugin",
            function:"run"
        },
        {
            name:"Change box size",
            function:"changeBboxSize"
        },
        {
            name:"Change rounds and channels order",
            function:"changeOrder"
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
istDeco.init = function (tmappObject) {
    istDeco.tmapp = tmappObject;
    istDeco.functions.forEach(function(funElement, i) {
        var aElement = document.createElement("a");
        aElement.href = "#";
        aElement.addEventListener("click",function (event) {
            console.log("Click", event, funElement.function);
            window["istDeco"][funElement.function]();
        });
        var spanElement = document.createElement("span");
        aElement.appendChild(spanElement);
        spanElement.innerHTML = funElement.name;
        dropdownMenu = document.getElementById("dropdown-menu-istDeco");
        dropdownMenu.appendChild(aElement);
    });
}

istDeco.loadImages = function () {
    console.log("Load Images");
    var op = istDeco.tmapp["object_prefix"];
    var vname = op + "_viewer";

    subfolder = window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/'));
    //subfolder = subfolder.substring(0, subfolder.lastIndexOf('/') + 1);
    $("#loadingModal").show();
    $.ajax(
        {
            // Post select to url.
            type : 'post',
            url : '/plugin/istDeco/importFolder',
            contentType: 'application/json; charset=utf-8',
            data : JSON.stringify({
                    'path' : subfolder
            }),
            success : function(data)
            {
                $("#loadingModal").hide();
                istDeco.loadState(data);
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

istDeco.run = function () {
    console.log("Load Images");
    var op = istDeco.tmapp["object_prefix"];
    var vname = op + "_viewer";

    var click_handler = function (event) {
        if (event.quick) {
            var OSDviewer = istDeco.tmapp[tmapp["object_prefix"] + "_viewer"];
            var viewportCoords = OSDviewer.viewport.pointFromPixel(event.position)
            var normCoords = OSDviewer.viewport.viewportToImageCoordinates(viewportCoords);

            var bbox = [Math.round(normCoords.x - istDeco._bboxSize/2), Math.round(normCoords.y - istDeco._bboxSize/2), istDeco._bboxSize, istDeco._bboxSize];
            var layers = istDeco.getLayers();
            var markers = istDeco.getMarkers(bbox);
            if (markers.length > 0) {
                if (!istDeco._order_rounds)
                    istDeco.changeOrder(false);
            }
            img = document.getElementById("ISS_istDeco_img");
            console.log(img);
            if (img) 
                img.style.filter = "blur(5px)";
            istDeco.getMatrix(bbox,layers,markers);
            
            color = "red";
            var boundBoxOverlay = document.getElementById("overlay-istDeco");
            if (boundBoxOverlay) {
                OSDviewer.removeOverlay(boundBoxOverlay);
            }
            var boundBoxRect = OSDviewer.viewport.imageToViewportRectangle(
                bbox[0], bbox[1], bbox[2], bbox[3]);
            boundBoxOverlay = $("<div id=\"overlay-istDeco\"></div>");
            boundBoxOverlay.css({
                border: "2px solid " + color
            });
            OSDviewer.addOverlay(boundBoxOverlay.get(0), boundBoxRect);
            

        } else { //if it is not quick then its panning
            // nothing
        }
    };

    //OSD handlers are not registered manually they have to be registered
    //using MouseTracker OSD objects 
    if (istDeco.ISS_mouse_tracker == undefined) {
        istDeco.ISS_mouse_tracker = new OpenSeadragon.MouseTracker({
            //element: this.fixed_svgov.node().parentNode, 
            element: istDeco.tmapp[vname].canvas,
            clickHandler: click_handler
        }).setTracking(true);
    }
}

istDeco.changeOrder = function (doPrompt) {
    if (doPrompt == undefined)
    doPrompt = true;
    var layers = istDeco.getLayers();
    rounds = [];
    channels = [];
    layers.forEach(function(layer, i) {
        parts = layer.name.split("_");
        round = parts[0];
        channel = parts[1];
        if (!channels.includes(channel)) {
            channels.push(channel);
        }
        if (!rounds.includes(round)) {
            rounds.push(round);
        }
    });
    if (doPrompt) {
        istDeco._order_rounds = prompt("Can you check order of rounds?",rounds.join(";")).split(";");
        istDeco._order_channels = prompt("Can you check order of channels?",channels.join(";")).split(";");
    }
}

istDeco.changeBboxSize = function (doPrompt) {
    istDeco._bboxSize = parseInt(prompt("Select the window region size:",istDeco._bboxSize));
}

istDeco.getMarkers = function (bbox) {
    var op = istDeco.tmapp["object_prefix"];
    var vname = op + "_viewer";
    if (!dataUtils[op + "_barcodeGarden"]) {
        return [];
    }

    const allMarkersCheckbox = document.getElementById("AllMarkers-checkbox-ISS");
    const showAll = allMarkersCheckbox && allMarkersCheckbox.checked;

    var OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];
    var xmin = bbox[0];//OSDviewer.viewport.imageToViewportCoordinates(bbox[0]);
    var ymin = bbox[1];//OSDviewer.viewport.imageToViewportCoordinates(bbox[1]);
    var xmax = xmin+bbox[2];//OSDviewer.viewport.imageToViewportCoordinates(bbox[2]);
    var ymax = ymin+bbox[3];//OSDviewer.viewport.imageToViewportCoordinates(bbox[3]);

    markersInViewportBounds = [];
    Object.keys(dataUtils[op + "_barcodeGarden"]).forEach(function (barcode) {
        const key = glUtils._barcodeToKey[barcode];  // Could be barcode or gene name
        const visible = showAll || markerUtils._checkBoxes[key].checked;
        hexInput = document.getElementById(key + "-color-ISS")
        if (hexInput) {
            var hexColor = document.getElementById(key + "-color-ISS").value;
        }
        else {
            var hexColor = "#000000";
        }

        if (visible) {
            newMarkers = markerUtils.arrayOfMarkersInBox(
                dataUtils[op + "_barcodeGarden"][barcode], xmin, ymin, xmax, ymax, { globalCoords: true }
            );
            newMarkers.forEach(function(m) {
                m.color = hexColor;
            })
            markersInViewportBounds = markersInViewportBounds.concat(newMarkers)
        }
    });
    return markersInViewportBounds;
}

istDeco.getLayers = function () {
    layers = []
    /*if (tmapp.fixed_file && tmapp.fixed_file != "") {
        layers.push( {
            name:tmapp.slideFilename, 
            tileSource: tmapp._url_suffix +  tmapp.fixed_file
        })
    }*/
    tmapp.layers.forEach(function(layer, i) {
        layers.push( {
            name:layer.name, 
            tileSource: layer.tileSource
        })
    });
    return layers;
}

istDeco.getMatrix = function (bbox, layers, markers, order) {
    var op = istDeco.tmapp["object_prefix"];
    var vname = op + "_viewer";
    console.log("Calling ajax getMatrix")
    $.ajax(
        {
            // Post select to url.
            type : 'post',
            url : '/plugin/istDeco/getMatrix',
            contentType: 'application/json; charset=utf-8',
            data : JSON.stringify({
                    'bbox' : bbox,
                    'layers' : layers,
                    'markers' : markers,
                    'order_rounds': istDeco._order_rounds,
                    'order_channels': istDeco._order_channels,
            }),
            success : function(data)
            {
                img = document.getElementById("ISS_istDeco_img");
                console.log("img", img)
                if (!img) {
                    var img = document.createElement("img");
                    img.id = "ISS_istDeco_img";
                    var elt = document.createElement("div");
                    elt.appendChild(img);
                    tmapp[vname].addControl(elt,{anchor: OpenSeadragon.ControlAnchor.BOTTOM_RIGHT});
                }
                img.setAttribute("src", "data:image/png;base64," + data);
                img.style.filter = "none";
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
 istDeco.loadState = function(state) {
    /*
    {
        markerFiles: [
            {
                path: "my/server/path.csv",
                title: "",
                comment: ""
            }
        ],
        CPFiles: [],
        regionFiles: [],
        layers: [
            {
                name:"",
                path:""
            }
        ],
        filters: [
            {
                name:"",
                default:"",
            }
        ],
        compositeMode: ""
    }
    */
    tmapp.fixed_file = "";
    if (state.markerFiles) {
        state.markerFiles.forEach(function(markerFile) {
            HTMLElementUtils.createDLButtonMarkers(
                markerFile.title,
                markerFile.path,
                markerFile.comment,
                markerFile.expectedCSV
            );
        });
    }
    if (state.CPFiles) {
        state.CPFiles.forEach(function(CPFile) {
            HTMLElementUtils.createDLButtonMarkersCP(
                CPFile.title,
                CPFile.path,
                CPFile.comment,
                CPFile.expectedCSV
            );
        });
    }
    if (state.regionFiles) {
        state.regionFiles.forEach(function(regionFile) {
            HTMLElementUtils.createDLButtonRegions(
                regionFile.title,
                regionFile.path,
                regionFile.comment
            );
        });
    }
    if (state.slideFilename) {
        tmapp.slideFilename = state.slideFilename;
        document.getElementById("project_title").innerText = state.slideFilename;
    }
    tmapp.layers = [];
    state.layers.forEach(function(layer) {
        pathname = window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/') + 1);
        //if (tmapp.fixed_file == "") {
        //    tmapp.fixed_file = pathname + layer.path + ".dzi";
        //    tmapp.slideFilename = layer.name;
        //}
        //else {
            tmapp.layers.push(
                {name: layer.name, tileSource: layer.path + ".dzi"}
            )
        //}
    });
    if (state.filters) {
        filterUtils._filtersUsed = state.filters;
    }
    if (state.layerFilters) {
        filterUtils._filterItems = state.layerFilters;
    }
    if (state.compositeMode) {
        filterUtils._compositeMode = state.compositeMode;
    }
    tmapp[tmapp["object_prefix"] + "_viewer"].world.removeAll();
    overlayUtils.addAllLayers();
    filterUtils.setCompositeOperation(filterUtils._compositeMode);
    filterUtils.getFilterItems();
    $(".visible-layers").prop("checked",true);$(".visible-layers").click();
    firstRound = null;
    tmapp.layers.forEach(function(layer, i) {
        round = layer.name.split("_")[0];
        console.log(round, i)
        if (!firstRound)
            firstRound = round;
        if (firstRound == round) {
            console.log(round, i)
            $("#visible-layer-"+i).click();
        }
    });
    
    //tmapp[tmapp["object_prefix"] + "_viewer"].world.resetItems()
}
