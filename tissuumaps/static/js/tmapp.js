/**
 * @file tmapp.js Main base for TissUUmaps to work
 * @author Leslie Solorzano
 * @see {@link tmapp}
 */

/**
 * @namespace tmapp
 * @version tmapp 2.0
 * @classdesc The root namespace for tmapp.
 */
tmapp = {
    _url_suffix: "",
    _scrollDelay: 900,
    fixed_file: "",
    mpp:0,
    slideFilename:"Main"
}

/** 
 * Get all the buttons from the interface and assign all the functions associated to them */
tmapp.registerActions = function () {
    tmapp["object_prefix"] = tmapp.options_osd.id.split("_")[0];
    var op = tmapp["object_prefix"];
    var cpop="CP";

    interfaceUtils.listen(op + '_add_layer_btn', 'click', function() { overlayUtils.addLayerFromSelect() }, false);
    interfaceUtils.listen(op + '_collapse_btn','click', function () { interfaceUtils.toggeRightPanel() },false);
    interfaceUtils.listen(op + '_bringmarkers_btn','click', function () { dataUtils.processISSRawData(); },false);
    interfaceUtils.listen(op + '_search','input', function () { markerUtils.hideRowsThatDontContain(); },false);
    interfaceUtils.listen(op + '_drawall_btn','click', function () { markerUtils.drawAllToggle(); },false);
    interfaceUtils.listen(op + '_drawregions_btn','click', function () { regionUtils.regionsOnOff() },false);
    interfaceUtils.listen(op + '_export_regions','click', function () { regionUtils.exportRegionsToJSON() },false);
    interfaceUtils.listen(op + '_import_regions','click', function () { regionUtils.importRegionsFromJSON() },false);
    interfaceUtils.listen(op + '_export_regions_csv','click', function () { regionUtils.pointsInRegionsToCSV() },false);
    interfaceUtils.listen(op + '_fillregions_btn','click', function () { regionUtils.fillAllRegions(); },false);
    interfaceUtils.listen(cpop + '_bringmarkers_btn','click', function () { CPDataUtils.processISSRawData() },false);
    var navtabs=document.getElementsByClassName("nav-tabs")[0];
    var uls=navtabs.getElementsByTagName("ul");
    for(var i=0;i<uls.length;i++){
        var as=uls[i].getElementsByTagName("a");
        for(var j=0;j<as.length;j++){
            as[j].addEventListener("click",function(){interfaceUtils.hideTabsExcept($(this))});
        }
    }
    //interfaceUtils.activateMainChildTabs("markers-gui");

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
    //pixelate because we need the exact values of pixels
    tmapp[vname].addHandler("tile-drawn", OSDViewerUtils.pixelateAtMaximumZoomHandler);

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
            if (overlayUtils._drawRegions) {
                //call region creator and drawer
                regionUtils.manager(event);
            }
        } else { //if it is not quick then its panning
            scroll_handler();
        }
    };

    //delay the scroll and the panning options so that there is a bit more time to calcualte which 
    //markers to plot and where and how many
    var isScrolling;
    var scroll_handler = function (event) {

        // Clear our timeout throughout the scroll
        window.clearTimeout(isScrolling);
        // Set a timeout to run after scrolling ends
        isScrolling = setTimeout(function () {

            // Run the callback
            console.log('Scrolling has stopped.');
            //
            overlayUtils.modifyDisplayIfAny();

        }, tmapp._scrollDelay);
    }


    //OSD handlers are not registered manually they have to be registered
    //using MouseTracker OSD objects 
    var ISS_mouse_tracker = new OpenSeadragon.MouseTracker({
        //element: this.fixed_svgov.node().parentNode, 
        element: tmapp[vname].canvas,
        clickHandler: click_handler,
        scrollHandler: scroll_handler
    }).setTracking(true);

    elt = document.getElementById("ISS_globalmarkersize");
    if (elt) {
        tmapp[vname].addControl(elt,{anchor: OpenSeadragon.ControlAnchor.TOP_RIGHT});
        elt.style.display="None";
    }

    if (tmapp.mpp != 0) {
        tmapp[vname].scalebar({
            pixelsPerMeter: tmapp.mpp ? (1e6 / tmapp.mpp) : 0,
            xOffset: 200,
            yOffset: 10,
            barThickness: 3,
            color: '#555555',
            fontColor: '#333333',
            backgroundColor: 'rgba(255, 255, 255, 0.5)',
            sizeAndTextRenderer: OpenSeadragon.ScalebarSizeAndTextRenderer.METRIC_LENGTH,
            location: OpenSeadragon.ScalebarLocation.BOTTOM_RIGHT
        });
    }
    //document.getElementById('cancelsearch-moving-button').addEventListener('click', function(){ markerUtils.showAllRows("moving");}); 
    filterUtils.initFilters();
    if (window.hasOwnProperty("glUtils")) {
        console.log("Using GPU-based marker drawing (WebGL canvas)")
        glUtils.init();
    } else {
        console.log("Using CPU-based marker drawing (SVG canvas)")
    }
} //finish init

/**
 * Options for the fixed and moving OSD 
 * all options are described here https://openseadragon.github.io/docs/OpenSeadragon.html#.Options */
tmapp.options_osd = {
    id: "ISS_viewer",
    prefixUrl: "js/openseadragon/images/",
    navigatorSizeRatio: 1,
    wrapHorizontal: false,
    showNavigator: true,
    navigatorPosition: "BOTTOM_LEFT",
    navigatorSizeRatio: 0.15,
    animationTime: 0.0,
    blendTime: 0,
    minZoomImageRatio: 1,
    maxZoomPixelRatio: 30,
    zoomPerClick: 1.0,
    constrainDuringPan: true,
    visibilityRatio: 1,
    showNavigationControl: false,
    maxImageCacheCount:500,
    imageSmoothingEnabled:false
}
