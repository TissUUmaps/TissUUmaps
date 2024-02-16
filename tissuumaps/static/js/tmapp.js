/**
 * @file tmapp.js Main base for TissUUmaps to work
 * @author Christophe Avenel, Fredrik Nysjö, Leslie Solorzano
 * @see {@link tmapp}
 */

/**
 * @namespace tmapp
 * @version tmapp 3.2
 * @classdesc The root namespace for tmapp.
 */
tmapp = {
    _url_suffix: "",
    _scrollDelay: 900,
    fixed_file: "",
    slideFilename:"Main"
}

/** 
 * Get all the buttons from the interface and assign all the functions associated to them */
tmapp.registerActions = function () {
    tmapp["object_prefix"] = tmapp.options_osd.id.split("_")[0];
    var op = tmapp["object_prefix"];

    interfaceUtils.listen(op + '_collapse_btn','click', function () { interfaceUtils.toggleRightPanel() },false);
    interfaceUtils.listen(op + '_export_regions','click', function () { regionUtils.exportRegionsToJSON() },false);
    interfaceUtils.listen(op + '_import_regions','click', function () { regionUtils.importRegionsFromJSON() },false);
    interfaceUtils.listen(op + '_export_regions_csv','click', function () { regionUtils.pointsInRegionsToCSV() },false);
    interfaceUtils.listen(op + '_fillregions_btn','click', function () { regionUtils.fillAllRegions(); },false);
    interfaceUtils.listen("capture_viewport","click",function(){overlayUtils.savePNG()},false)
    interfaceUtils.listen("see_version","click",function(){tmapp.seeVersion()},false)
    interfaceUtils.listen("plus-1-button","click",function(){interfaceUtils.generateDataTabUI()},false)
    interfaceUtils.listen('save_project_menu', 'click', function() { projectUtils.saveProject() }, false);
    interfaceUtils.listen('load_project_menu', 'click', function() { projectUtils.loadProjectFile() }, false);
    interfaceUtils.listen('project_select', 'change', function() { 
        // go to url of selected project value
        window.location.href = window.location.origin + "/" + this.value;
     }, false);
    document.addEventListener("mousedown",function(){tmapp["ISS_viewer"].removeOverlay("ISS_marker_info");});

    // dataUtils.processEventForCSV("morphology",cpop + '_csv');
    //dataUtils.processEventForCSV("gene",op + '_csv');
    
    var navtabs=document.getElementsByClassName("nav-tabs")[0];
    var uls=navtabs.getElementsByTagName("ul");
    for(var i=0;i<uls.length;i++){
        var as=uls[i].getElementsByTagName("a");
        for(var j=0;j<as.length;j++){
            as[j].addEventListener("click",function(){interfaceUtils.hideTabsExcept($(this))});
        }
    }

    const regionTabEl = document.querySelector('#title-tab-regions')
    regionTabEl?.addEventListener('show.bs.tab', function (event) {
        overlayUtils._regionToolbar = true;
        regionUtils.regionToolbarOnOff();
    })
    regionTabEl?.addEventListener('hide.bs.tab', function (event) {
        overlayUtils._regionToolbar = false;
        regionUtils.regionToolbarOnOff();
        regionUtils.setMode(null);
    })
    
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
tmapp.init = function () {
    //This prefix will be called by all other utilities in js/utils
    tmapp["object_prefix"] = tmapp.options_osd.id.split("_")[0];
    var op = tmapp["object_prefix"];
    var vname = op + "_viewer";
    //init OSD viewer
    tmapp[vname] = OpenSeadragon(tmapp.options_osd);
    // Disable keyboard hack
    tmapp[vname].innerTracker.keyHandler = null;
    tmapp[vname].innerTracker.keyDownHandler = null;
    tmapp[vname].innerTracker.keyPressHandler = null;

    if(!tmapp.layers){
        tmapp.layers = [];
    }
    overlayUtils.addAllLayers();
    //Create svgOverlay(); so that anything like D3, or any canvas library can act upon. https://d3js.org/
    var svgovname = tmapp["object_prefix"] + "_svgov";
    tmapp[svgovname] = tmapp[vname].svgOverlay();

                          //main node
    overlayUtils._d3nodes[op + "_svgnode"] = d3.select(tmapp[svgovname].node());
    
    //overlay for marker data                                             //main node
    overlayUtils._d3nodes[op + "_markers_svgnode"] = overlayUtils._d3nodes[op + "_svgnode"].append("g")
        .attr("id", op + "_markers_svgnode");
    //overlay for region data                                              //main node
    overlayUtils._d3nodes[op + "_regions_svgnode"] = overlayUtils._d3nodes[op + "_svgnode"].append("g")
        .attr("id", op + "_regions_svgnode");
    //overlay for CP data   
    var cpop="CP";                                   //main node;
    overlayUtils._d3nodes[cpop+"_svgnode"] = overlayUtils._d3nodes[op + "_svgnode"].append("g")
        .attr("id", cpop+"_svgnode");

    var click_handler = function (event) {
        if (event.quick) {
            if (regionUtils._regionMode == "points") {
                //call region creator and drawer
                regionUtils.manager(event);
            }
        } else { //if it is not quick then its panning
            //scroll_handler();
        }
    };

    function pressHandler(event) {
        if (regionUtils._regionMode == "free") {
          // Call region creator and drawer
          regionUtils.freeHandManager(event);
        }
        if (regionUtils._regionMode == "brush") {
          // Call region creator and drawer
          regionUtils.brushManager(event);
        }
        if (regionUtils._regionMode == "rectangle") {
          // Call region creator and drawer
          regionUtils.rectangleManager(event);
        }
        if (regionUtils._regionMode == "ellipse") {
          // Call region creator and drawer
          regionUtils.ellipseManager(event);
        }
    }
    function moveHandler(event) {
        if (regionUtils._regionMode == "brush") {
          // Call region creator and drawer
          regionUtils.brushHover(event);
        }
    }

    //OSD handlers are not registered manually they have to be registered
    //using MouseTracker OSD objects 
    /*var ISS_mouse_tracker = new OpenSeadragon.MouseTracker({
        element: tmapp[vname].canvas,
        clickHandler: click_handler
    }).setTracking(true);*/
    tmapp["ISS_viewer"].addHandler("canvas-press", ()=>{this.dragging = true;});
    tmapp["ISS_viewer"].addHandler("canvas-release", ()=>{this.dragging = false;});

    tmapp["ISS_viewer"].addHandler("canvas-press", pressHandler);
    tmapp["ISS_viewer"].addHandler('canvas-click', click_handler);
    new OpenSeadragon.MouseTracker({
        element: tmapp["ISS_viewer"].canvas,
        moveHandler: moveHandler
    }).setTracking(true);
    
    tmapp["ISS_viewer"].addHandler("animation", function animationFinishHandler(event){
        const drawingclass = "_brushRegion";
        d3.selectAll("." + drawingclass).selectAll('circle').each(function(el) {
            $(this).attr('r', 0.2 * regionUtils._handleRadius / tmapp["ISS_viewer"].viewport.getZoom());
        });
    });
    tmapp["ISS_viewer"].addHandler("animation-finish", function animationFinishHandler(event){
        d3.selectAll("." + regionUtils._drawingclass).selectAll('polyline').each(function(el) {
            $(this).attr('stroke-width', regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom());
        });
        
        d3.selectAll("." + regionUtils._drawingclass).selectAll('circle').each(function(el) {
            $(this).attr('r', 10* regionUtils._handleRadius / tmapp["ISS_viewer"].viewport.getZoom());
        });
        d3.selectAll(".regionpoly").each(function(el) {
            $(this).attr('stroke-width', regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom());
        });
        d3.selectAll(".region_previewpoly").each(function(el) {
            $(this).attr('stroke-width', 2.5 * regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom());
        });
        var op = tmapp["object_prefix"];
        tmapp[op + "_viewer"].imageLoaderLimit = 50;
        tmapp[op + "_viewer"].immediateRender = true;
        var count = tmapp[op + "_viewer"].world.getItemCount();
        for (var i = 0; i < count; i++) {
            var tiledImage = tmapp[op + "_viewer"].world.getItemAt(i);
            tiledImage.immediateRender = true;
        }

    });
    tmapp["ISS_viewer"].addHandler("zoom", function animationFinishHandler(event){
        const zoom = event.zoom;
        var op = tmapp["object_prefix"];
        if (!tmapp[op + "_viewer"].world.getItemAt(0)) return;
        let imageZoom = tmapp[op + "_viewer"].world.getItemAt(0).viewportToImageZoom(zoom)
        if (imageZoom > 0.8) {
            tmapp[op + "_viewer"].drawer.setImageSmoothingEnabled(false);
        }
        else {
            if (!flask.standalone.backend) {
                tmapp[op + "_viewer"].drawer.setImageSmoothingEnabled(true);
            }
        }
        glUtils._regionStrokeWidth = (regionUtils._regionStrokeAdaptOnZoom) ? 
            imageZoom * regionUtils._regionStrokeWidth :
            regionUtils._regionStrokeWidth;
    });
    tmapp["ISS_viewer"].addHandler("animation-start", function animationFinishHandler(event){
        var op = tmapp["object_prefix"];
        tmapp[op + "_viewer"].imageLoaderLimit = 1;
        tmapp[op + "_viewer"].immediateRender = false;
        var count = tmapp[op + "_viewer"].world.getItemCount();
        for (var i = 0; i < count; i++) {
            var tiledImage = tmapp[op + "_viewer"].world.getItemAt(i);
            tiledImage.immediateRender = false;
        }
    });
    tmapp["ISS_viewer"].addHandler("open", function animationFinishHandler(event){
        tmapp["ISS_viewer"].raiseEvent("zoom", { zoom: tmapp["ISS_viewer"].viewport.getZoom() });
    });
    $(document).on("wheel", "input[type=range]", moveSlider);
    function moveSlider(e){
        var zoomLevel = parseFloat(e.target.value); 
        let step = parseFloat(e.target.step);
        // detect positive or negative scrolling
        if ( e.originalEvent.wheelDelta < 0 ) {
            //scroll down
            $(e.target).val(zoomLevel+step);
        } else {
            //scroll up
            $(e.target).val(zoomLevel-step);
        }

        // Create a new 'change' event
        var event = new Event('input', {
            bubbles: true,
            cancelable: true,
        });
          
        // trigger the change event
        e.target.dispatchEvent(event);

        //prevent page fom scrolling
        return false;
    }
    $(document).on("input", "input[type=range]", updateSlider);
    function updateSlider(e){
        if (e.target.id == "ISS_globalmarkersize_text" || e.target.id == "channelRangeInput") return;
        if ($("#opacity-layer-0").attr('data-bs-original-title') !== undefined) {
            e.target.title = e.target.value;
            $(e.target).tooltip('show');
        }
        $(e.target).attr('data-bs-original-title', e.target.value);
        $(e.target).tooltip('dispose');
        $(e.target).tooltip('show');
        //$(e.target).tooltip({ placement: 'bottom' });

    }
    elt = document.getElementById("ISS_globalmarkersize");
    if (elt) {
        tmapp[vname].addControl(elt,{
            anchor: OpenSeadragon.ControlAnchor.TOP_RIGHT
        });
        elt.classList.add("d-none");
    }
    
    filterUtils.initFilters();
    if (window.hasOwnProperty("glUtils")) {
        console.log("Using GPU-based marker drawing (WebGL canvas)")
        glUtils.init();
    } else {
        console.log("Using CPU-based marker drawing (SVG canvas)")
    }
    
    if (dataUtils._hdf5Api === undefined) {
        dataUtils._hdf5Api = new H5AD_API()
    }
} //finish init

tmapp.seeVersion = function() {
    interfaceUtils.alert("TissUUmaps version: <b>" + tmapp.version + "</b><br/>TissUUmaps-Schema version: <b>" + tmapp.schema_version + "</b>", "Software version");
}

/**
 * Options for the fixed and moving OSD 
 * all options are described here https://openseadragon.github.io/docs/OpenSeadragon.html#.Options */
tmapp.options_osd = {
    id: "ISS_viewer",
    prefixUrl: "js/openseadragon/images/",
    navigatorSizeRatio: 0.15,
    wrapHorizontal: false,
    showNavigator: true,
    navigatorPosition: "BOTTOM_LEFT",
    animationTime: 0.0,
    blendTime: 0,
    minZoomImageRatio: 0.9,
    maxZoomPixelRatio: 30,
    immediateRender: false,
    zoomPerClick: 1.0,
    constrainDuringPan: true,
    visibilityRatio: 0.5,
    showNavigationControl: false,
    maxImageCacheCount:2000,
    imageSmoothingEnabled:false,
    preserveImageSizeOnResize: true,
    imageLoaderLimit: 50,
    gestureSettingsUnknown: {
        flickEnabled: false
    },
    gestureSettingsTouch: {
        flickEnabled: false
    },
    gestureSettingsPen: {
        flickEnabled: false
    },
    debouncePanEvents: false,
}

function toggleFullscreen() {
    let full_ui = document.getElementById("main-ui");
    let bIsFullscreen = document.fullScreen || document.mozFullScreen || document.webkitIsFullScreen;
    if (!bIsFullscreen) {
        if (full_ui.requestFullscreen) {
            full_ui.requestFullscreen();
        } else if (full_ui.webkitRequestFullscreen) {
            full_ui.webkitRequestFullscreen(Element.ALLOW_KEYBOARD_INPUT);
        } else if (full_ui.msRequestFullscreen) {
            full_ui.msRequestFullscreen();
        } else if (full_ui.webkitRequestFullscreen) {
            full_ui.mozRequestFullscreen();
        }
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen()
        } else if (document.webkitCancelFullScreen) {
            document.webkitCancelFullScreen();
        } else if (document.webkitCancelFullScreen) {
            document.msCancelFullScreen();
        } else if (document.mozCancelFullScreen) {
            document.mozCancelFullScreen()
        }
    }
}

function toggleNavbar(turn_on = null) {
    let main_navbar = document.getElementsByTagName("nav")[0];

    if (turn_on === true) {
        main_navbar.classList.remove("d-none");
    } else if (turn_on === false) {
        main_navbar.classList.add("d-none");
    } else if (turn_on === null) {
        if (main_navbar.classList.contains("d-none")) {
            toggleNavbar(true);
        } else {
            toggleNavbar(false);
        }
    }
}

$( document ).ready(function() {
    let ISS_viewer = document.getElementById("ISS_viewer");
    let ISS_viewer_container = document.getElementById("ISS_viewer_container");

    ISS_viewer.addEventListener('dblclick', function (e) {
        interfaceUtils.generateNotification(
            "Fullscreen on double click has been disabled from version 3.2. Please use the F key to go fullscreen.",
            "fullscreen", 
            false,
            2000,
            "info"
        );
    });

    let full_ui = document.getElementById("main-ui");
    full_ui.addEventListener('fullscreenchange', (event) => {
        // document.fullscreenElement will point to the element that
        // is in fullscreen mode if there is one. If not, the value
        // of the property is null.
        if (document.fullscreenElement) {
            toggleNavbar(false);
        } else {
            toggleNavbar(true);
        }
    });

    ISS_viewer_container.addEventListener("keydown", (event) => {
        if (event.key === "0") {
            interfaceUtils.toggleRightPanel();
        } else if (event.key === "f") {
            toggleFullscreen();
        } else if (event.key === "m") {
            for (const [key, value] of Object.entries(dataUtils.data)) {
                $("#"+key+"_all_check").click();
                $("#"+key+"_All_check").click();
            }
        } else if (event.key === "r") {
            $("#regionUI_all_check").click();
        } else if (event.key === "Escape") {
            regionUtils.resetSelection();
            regionUtils.setMode(null);
        }
    });
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            regionUtils.resetSelection();
            regionUtils.setMode(null);
        }
    });
});
