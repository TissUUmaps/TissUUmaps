/**
 * @file Spot_Detection.js
 * @author Christophe Avenel
 */

/**
 * @namespace Spot_Detection
 * @classdesc The root namespace for Spot_Detection.
 */
 var Spot_Detection;
 Spot_Detection = {
     name:"Spot detection - Quality Control",
     _bboxSize:11,
     _figureSize:7,
     _order_rounds:null,
     _order_channels:null,
     _only_picked:true,
     _started:false,
     _cmap:'Greys_r'
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
 Spot_Detection.init = function (container) {
     Spot_Detection.changeOrder(false);
     // row0=HTMLElementUtils.createElement({"kind":"h6", "extraAttributes":{"class":""}});
     // row0.innerText = "Options:"
     row1=HTMLElementUtils.createRow({});
        col11=HTMLElementUtils.createColumn({"width":12});
            label112=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":"Spot_Detection_bboxSize"}});
            label112.innerHTML="Box size:";    
            var input112=HTMLElementUtils.createElement({"kind":"input", "id":"Spot_Detection_bboxSize", "extraAttributes":{ "class":"form-text-input form-control", "type":"number", "value":Spot_Detection._bboxSize}});

    input112.addEventListener("change",(event)=>{
        console.log("Spot_Detection._bboxSize", input112.value, parseInt(input112.value));
        Spot_Detection._bboxSize = parseInt(input112.value);
    });

    row7=HTMLElementUtils.createRow({});
        col71=HTMLElementUtils.createColumn({"width":12});
            label712=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":"Spot_Detection_bboxSize"}});
            label712.innerHTML="Figure size:";    
            var input712=HTMLElementUtils.createElement({"kind":"input", "id":"Spot_Detection_figureSize", "extraAttributes":{ "class":"form-text-input form-control", "type":"number", "value":Spot_Detection._figureSize}});
         
    input712.addEventListener("change",(event)=>{
        Spot_Detection._figureSize = parseInt(input712.value);
    });

    row2=HTMLElementUtils.createRow({});
        col21=HTMLElementUtils.createColumn({"width":12});
            label212=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":"Spot_Detection_order_rounds"}});
            label212.innerHTML="Round order:";    
            var input212=HTMLElementUtils.createElement({"kind":"input", "id":"Spot_Detection_order_rounds", "extraAttributes":{ "class":"form-text-input form-control", "type":"text", "value":JSON.stringify(Spot_Detection._order_rounds)}});
            
    input212.addEventListener("change",(event)=>{
        Spot_Detection._order_rounds = JSON.parse(input212.value);
    });
    
    row3=HTMLElementUtils.createRow({});
        col31=HTMLElementUtils.createColumn({"width":12});
            label312=HTMLElementUtils.createElement({"kind":"label","extraAttributes":{"class":"form-check-label","for":"Spot_Detection_order_channels"}});
            label312.innerHTML="Channel order:";    
            var input312=HTMLElementUtils.createElement({"kind":"input", "id":"Spot_Detection_order_channels", "extraAttributes":{ "class":"form-text-input form-control", "type":"text", "value":JSON.stringify(Spot_Detection._order_channels)}});
            
    input312.addEventListener("change",(event)=>{
        Spot_Detection._order_channels = JSON.parse(input312.value);
    });

    row4=HTMLElementUtils.createRow({});
        col41=HTMLElementUtils.createColumn({"width":12});
            var input411=HTMLElementUtils.createElement({"kind":"input", "id":"Spot_Detection_only_picked","extraAttributes":{"class":"form-check-input","type":"checkbox","checked":true}});
            label411=HTMLElementUtils.createElement({"kind":"label", "extraAttributes":{ "for":"Spot_Detection_only_picked" }});
            label411.innerHTML="&nbsp;Only show central selected marker"

    input411.addEventListener("change",(event)=>{
        console.log(input411.checked, Spot_Detection._only_picked);
        Spot_Detection._only_picked = input411.checked;
        console.log(input411.checked, Spot_Detection._only_picked);
    });

    row6=HTMLElementUtils.createRow({});
        col61=HTMLElementUtils.createColumn({"width":12});
            select611=HTMLElementUtils.createElement({"kind":"select","id":"Spot_Detection_colormap","extraAttributes":{"class":"form-select form-select-sm","aria-label":".form-select-sm"}});
            label612=HTMLElementUtils.createElement({"kind":"label", "extraAttributes":{"for":"Spot_Detection_colormap"} });
            label612.innerText="Select colormap";

    select611.addEventListener("change",(event)=>{
        Spot_Detection._cmap = select611.value;
    });

    row5=HTMLElementUtils.createRow({});
        col51=HTMLElementUtils.createColumn({"width":12});
            button511=HTMLElementUtils.createButton({"extraAttributes":{ "class":"btn btn-primary mx-2"}})
            button511.innerText = "Load images";

    button511.addEventListener("click",(event)=>{
        interfaceUtils.prompt("Give the path format of your images, use * for numbers:","Round*_*","Load images")
        .then((pathFormat) => {
            Spot_Detection.loadImages(pathFormat)
        })
    });

    container.innerHTML = "";
    // container.appendChild(row0);
    container.appendChild(row4);
        row4.appendChild(col41);
            col41.appendChild(input411);
            col41.appendChild(label411);
    container.appendChild(row1);
        row1.appendChild(col11);
            col11.appendChild(label112);
            col11.appendChild(input112);
    container.appendChild(row7);
        row7.appendChild(col71);
            col71.appendChild(label712);
            col71.appendChild(input712);
    container.appendChild(row2);
        row2.appendChild(col21);
            col21.appendChild(label212);
            col21.appendChild(input212);
    container.appendChild(row3);
        row3.appendChild(col31);
            col31.appendChild(label312);
            col31.appendChild(input312);
    container.appendChild(row6);
        row6.appendChild(col61);
            col61.appendChild(label612);
            col61.appendChild(select611);
    container.appendChild(row5);
        row5.appendChild(col51);
            col51.appendChild(button511);
    cmap = ['Greys_r', 'Greys', 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'];
    interfaceUtils.addElementsToSelect("Spot_Detection_colormap", cmap);
        
     Spot_Detection.run();
 }

Spot_Detection.loadImages = function (pathFormat) {
    console.log("Load Images");
    var op = tmapp["object_prefix"];
    var vname = op + "_viewer";

    subfolder = window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/'));
    //subfolder = subfolder.substring(0, subfolder.lastIndexOf('/') + 1);
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    const path = urlParams.get('path')
    $.ajax(
        {
            // Post select to url.
            type : 'post',
            url : '/plugins/Spot_Detection/importFolder',
            contentType: 'application/json; charset=utf-8',
            data : JSON.stringify({
                    'path' : path,
                    'pathFormat': pathFormat
            }),
            success : function(data)
            {
                projectUtils.loadProject(data);
                setTimeout(function() {
                    Spot_Detection.changeOrder(false);
                    $("#Spot_Detection_order_rounds")[0].value = JSON.stringify(Spot_Detection._order_rounds);
                    $("#Spot_Detection_order_channels")[0].value = JSON.stringify(Spot_Detection._order_channels);
                }, 500);
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

Spot_Detection.run = function () {
    if (Spot_Detection._started)
        return;
    Spot_Detection._started = true;
    var op = tmapp["object_prefix"];
    var vname = op + "_viewer";

    var click_handler = function (event) {
        if (event.quick) {
            var OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];
            var viewportCoords = OSDviewer.viewport.pointFromPixel(event.position)
            var normCoords = OSDviewer.viewport.viewportToImageCoordinates(viewportCoords);

            var bbox = [Math.round(normCoords.x - Spot_Detection._bboxSize/2), Math.round(normCoords.y - Spot_Detection._bboxSize/2), Spot_Detection._bboxSize, Spot_Detection._bboxSize];
            var layers = Spot_Detection.getLayers();
            var markers = Spot_Detection.getMarkers(bbox);
            if (markers.length > 0) {
                if (!Spot_Detection._order_rounds)
                    Spot_Detection.changeOrder(false);
            }
            img = document.getElementById("ISS_Spot_Detection_img");
            console.log(img);
            if (img) 
                img.style.filter = "blur(5px)";
            Spot_Detection.getMatrix(bbox,layers,markers);
            
            color = "red";
            var boundBoxOverlay = document.getElementById("overlay-Spot_Detection");
            if (boundBoxOverlay) {
                OSDviewer.removeOverlay(boundBoxOverlay);
            }
            var boundBoxRect = OSDviewer.viewport.imageToViewportRectangle(
                bbox[0], bbox[1], bbox[2], bbox[3]);
            boundBoxOverlay = $("<div id=\"overlay-Spot_Detection\"></div>");
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
    if (Spot_Detection.ISS_mouse_tracker == undefined) {
        Spot_Detection.ISS_mouse_tracker = new OpenSeadragon.MouseTracker({
            //element: this.fixed_svgov.node().parentNode, 
            element: tmapp[vname].canvas,
            clickHandler: click_handler
        }).setTracking(true);
    }
}

Spot_Detection.changeOrder = function (doPrompt) {
    if (doPrompt == undefined)
    doPrompt = true;
    var layers = Spot_Detection.getLayers();
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
    Spot_Detection._order_rounds = rounds;
    Spot_Detection._order_channels = channels;
}

Spot_Detection.getMarkers = function (bbox) {
    var xmin = bbox[0];//OSDviewer.viewport.imageToViewportCoordinates(bbox[0]);
    var ymin = bbox[1];//OSDviewer.viewport.imageToViewportCoordinates(bbox[1]);
    var xmax = xmin+bbox[2];//OSDviewer.viewport.imageToViewportCoordinates(bbox[2]);
    var ymax = ymin+bbox[3];//OSDviewer.viewport.imageToViewportCoordinates(bbox[3]);

    markersInViewportBounds = [];
    for (dataset in dataUtils.data) {
        if (Spot_Detection._only_picked && glUtils._pickedMarker[0] != -1) {
            pickedMarker = dataUtils.data[glUtils._pickedMarker[0]]["_processeddata"][glUtils._pickedMarker[1]]
        }
        else {
            pickedMarker = null;
        }
        var allkeys=Object.keys(dataUtils.data[dataset]["_groupgarden"]);
        for (var codeIndex in dataUtils.data[dataset]["_groupgarden"]) {
            var  inputs = interfaceUtils._mGenUIFuncs.getGroupInputs(dataset, codeIndex);
            var hexColor = "color" in inputs ? inputs["color"] : "#ffff00";
            var visible = "visible" in inputs ? inputs["visible"] : true;
            if (visible) {
                var newMarkers=regionUtils.searchTreeForPointsInBbox(dataUtils.data[dataset]["_groupgarden"][codeIndex],
                    xmin,ymin,
                    xmax,ymax, {
                        "globalCoords":true,
                        "xselector":dataUtils.data[dataset]["_X"],
                        "yselector":dataUtils.data[dataset]["_Y"]
                    });
                newMarkers.forEach(function(m) {
                    m.color = hexColor;
                    m.global_X_pos = parseFloat(m[dataUtils.data[dataset]["_X"]]),
                    m.global_Y_pos = parseFloat(m[dataUtils.data[dataset]["_Y"]])
                })
                if (Spot_Detection._only_picked) {
                    console.log(glUtils._pickedMarker);
                    console.log(markersInViewportBounds);
                    markersInViewportBounds = markersInViewportBounds.filter(function(p){return p == pickedMarker})
                }
                markersInViewportBounds = markersInViewportBounds.concat(newMarkers)
            }
        }
    }
    console.log(markersInViewportBounds, glUtils._pickedMarker)
    
    return markersInViewportBounds;
}

Spot_Detection.getLayers = function () {
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

Spot_Detection.getMatrix = function (bbox, layers, markers, order) {
    var op = tmapp["object_prefix"];
    var vname = op + "_viewer";
    console.log("Calling ajax getMatrix")
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    const path = urlParams.get('path')
    $.ajax(
        {
            // Post select to url.
            type : 'post',
            url : '/plugins/Spot_Detection/getMatrix',
            contentType: 'application/json; charset=utf-8',
            data : JSON.stringify({
                    'bbox' : bbox,
                    'figureSize' : Spot_Detection._figureSize,
                    'layers' : layers,
                    'path' : path,
                    'markers' : markers,
                    'order_rounds': Spot_Detection._order_rounds,
                    'order_channels': Spot_Detection._order_channels,
                    'cmap': Spot_Detection._cmap
            }),
            success : function(data)
            {
                img = document.getElementById("ISS_Spot_Detection_img");
                console.log("img", img)
                if (!img) {
                    var img = document.createElement("img");
                    img.id = "ISS_Spot_Detection_img";
                    var elt = document.createElement("div");
                    elt.classList.add("viewer-layer")
                    elt.classList.add("px-1")
                    elt.classList.add("mx-1")
                    elt.appendChild(img);
                    tmapp[vname].addControl(elt,{anchor: OpenSeadragon.ControlAnchor.BOTTOM_RIGHT});
                    elt.parentElement.parentElement.style.zIndex = "100";
                    elt.style.display = "table";
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
 Spot_Detection.loadState = function(state) {
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
