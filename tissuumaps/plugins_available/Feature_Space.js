/**
 * @file Feature_Space.js
 * @author Christophe Avenel
 */

/**
 * @namespace Feature_Space
 * @classdesc The root namespace for Feature_Space.
 */
 var Feature_Space;
 Feature_Space = {
     name:"Feature_Space Plugin",
     _dataset:null,
     _UMAP1:null,
     _UMAP2:null,
     _region:null,
     _regionPixels:null,
     _newwin:null
  }
 
 /**
  * @summary */
 Feature_Space.init = function (container) {
    row1=HTMLElementUtils.createRow({});
        col11=HTMLElementUtils.createColumn({"width":12});
            button111=HTMLElementUtils.createButton({"extraAttributes":{ "class":"btn btn-primary mx-2"}})
            button111.innerText = "Refresh drop-down lists based on loaded markers";

    row2=HTMLElementUtils.createRow({});
        col21=HTMLElementUtils.createColumn({"width":12});
            select211=HTMLElementUtils.createElement({"kind":"select","id":"Feature_Space_dataset","extraAttributes":{"class":"form-select form-select-sm","aria-label":".form-select-sm"}});
            label212=HTMLElementUtils.createElement({"kind":"label", "extraAttributes":{"for":"Feature_Space_dataset"} });
            label212.innerText="Select marker dataset";

    row3=HTMLElementUtils.createRow({});
        col31=HTMLElementUtils.createColumn({"width":12});
            select311=HTMLElementUtils.createElement({"kind":"select","id":"UMAP1","extraAttributes":{"class":"form-select form-select-sm","aria-label":".form-select-sm"}});
            label312=HTMLElementUtils.createElement({"kind":"label", "extraAttributes":{"for":"UMAP1"} });
            label312.innerText="Select Feature Space X";

    row4=HTMLElementUtils.createRow({});
        col41=HTMLElementUtils.createColumn({"width":12});
            select411=HTMLElementUtils.createElement({"kind":"select","id":"UMAP2","extraAttributes":{"class":"form-select form-select-sm","aria-label":".form-select-sm"}});
            label412=HTMLElementUtils.createElement({"kind":"label", "extraAttributes":{"for":"UMAP2"} });
            label412.innerText="Select Feature Space Y";

    row5=HTMLElementUtils.createRow({});
        col51=HTMLElementUtils.createColumn({"width":12});
            button511=HTMLElementUtils.createButton({"extraAttributes":{ "class":"btn btn-primary mx-2"}})
            button511.innerText = "Display Feature Space";
    
    button111.addEventListener("click",(event)=>{
        interfaceUtils.cleanSelect("Feature_Space_dataset");
        interfaceUtils.cleanSelect("UMAP1");
        interfaceUtils.cleanSelect("UMAP2");
        var datasets = Object.keys(dataUtils.data).map(function(e, i) {
            return {value:e, innerHTML:document.getElementById(e + "_tab-name").value};
        });
        interfaceUtils.addObjectsToSelect("Feature_Space_dataset", datasets);
        var event = new Event('change');
        interfaceUtils.getElementById("Feature_Space_dataset").dispatchEvent(event);
    });

    select211.addEventListener("change",(event)=>{
        Feature_Space._dataset = select211.value;
        if (!dataUtils.data[Feature_Space._dataset]) return;
        interfaceUtils.cleanSelect("UMAP1");
        interfaceUtils.addElementsToSelect("UMAP1", dataUtils.data[Feature_Space._dataset]._csv_header);
        interfaceUtils.cleanSelect("UMAP2");
        interfaceUtils.addElementsToSelect("UMAP2", dataUtils.data[Feature_Space._dataset]._csv_header);
        if (dataUtils.data[Feature_Space._dataset]._csv_header.indexOf("UMAP1") > 0) {
            interfaceUtils.getElementById("UMAP1").value = "UMAP1";
            var event = new Event('change');
            interfaceUtils.getElementById("UMAP1").dispatchEvent(event);
        }
        if (dataUtils.data[Feature_Space._dataset]._csv_header.indexOf("UMAP2") > 0) {
            interfaceUtils.getElementById("UMAP2").value = "UMAP2";
            var event = new Event('change');
            interfaceUtils.getElementById("UMAP2").dispatchEvent(event);
        }
    });
    select311.addEventListener("change",(event)=>{
        Feature_Space._UMAP1 = select311.value;
    });
    select411.addEventListener("change",(event)=>{
        Feature_Space._UMAP2 = select411.value;
    });

    button511.addEventListener("click",(event)=>{
        Feature_Space.run();
    });

    container.innerHTML = "";
    // container.appendChild(row0);
    container.appendChild(row1);
    row1.appendChild(col11);
        col11.appendChild(button111);
    container.appendChild(row2);
        row2.appendChild(col21);
            col21.appendChild(label212);
            col21.appendChild(select211);
    container.appendChild(row3);
        row3.appendChild(col31);
            col31.appendChild(label312);
            col31.appendChild(select311);
    container.appendChild(row4);
        row4.appendChild(col41);
            col41.appendChild(label412);
            col41.appendChild(select411);
    container.appendChild(row5);
        row5.appendChild(col51);
            col51.appendChild(button511);
    var event = new Event('click');
    button111.dispatchEvent(event);
 }

function copyDataset(dataIn, dataOut) {
    var headers = interfaceUtils._mGenUIFuncs.getTabDropDowns(Feature_Space._dataset);
    dataOut["expectedHeader"] = Object.assign({}, ...Object.keys(headers).map((k) => ({[k]: headers[k].value})));
    var radios = interfaceUtils._mGenUIFuncs.getTabRadiosAndChecks(Feature_Space._dataset);
    dataOut["expectedRadios"] = Object.assign({}, ...Object.keys(radios).map((k) => ({[k]: radios[k].checked})));
    dataOut["expectedHeader"]["X"] = Feature_Space._UMAP1;
    dataOut["expectedHeader"]["Y"] = Feature_Space._UMAP2;
    for (var key of Object.keys(dataIn)) {
        if (["_X","_Y","expectedHeader","expectedRadios","_groupgarden"].indexOf(key) == -1) {
            dataOut[key] = dataIn[key];
        }
        else if (key == "_X") {
            dataOut[key] = Feature_Space._UMAP1;
        }
        else if (key == "_Y") {
            dataOut[key] = Feature_Space._UMAP2;
        }
    }
}

Feature_Space.run = function () {
    var op = tmapp["object_prefix"];
    var vname = op + "_viewer";
    var Feature_Space_Control = document.getElementById("Feature_Space_Control");
    if (!Feature_Space_Control) {
        Feature_Space_Control = document.createElement("iframe");
        Feature_Space_Control.id = "Feature_Space_Control";
        Feature_Space_Control.style.width= "100%";
        Feature_Space_Control.style.height= "100%";
        Feature_Space_Control.style.borderLeft= "1px solid #aaa";
        var elt = document.createElement("div");
        elt.style.width= "40%";
        elt.style.height= "100%";
        elt.style.display = "inline-flex";
        elt.appendChild(Feature_Space_Control);
        document.getElementById("ISS_viewer").appendChild(elt);
        $(".openseadragon-container")[0].style.display = "inline-flex";
        $(".openseadragon-container")[0].style.width= "60%";
            
        Feature_Space_Control.addEventListener("load", ev => {
            Feature_Space_Control.classList.add("d-none");
            var timeout = setTimeout(function() {
                var newwin = Feature_Space_Control.contentWindow;
                Feature_Space._newwin = newwin;
                //OSD handlers are not registered manually they have to be registered
                //using MouseTracker OSD objects 
                if (newwin.tmapp.ISS_viewer) {
                    clearInterval(timeout);
                }
                else {return;}
                new Feature_Space._newwin.OpenSeadragon.MouseTracker({
                    element: Feature_Space._newwin.tmapp[vname].canvas,
                    moveHandler: Feature_Space.moveHandler/*,
                    pressHandler: Feature_Space.pressHandler,
                    releaseHandler: Feature_Space.releaseHandler*/
                }).setTracking(true);
                
                Feature_Space._newwin.tmapp["ISS_viewer"].addHandler('canvas-press', (event) => {
                    Feature_Space.pressHandler(event)
                });
                Feature_Space._newwin.tmapp["ISS_viewer"].addHandler('canvas-release', (event) => {
                    Feature_Space.releaseHandler(event)
                });
                Feature_Space._newwin.tmapp["ISS_viewer"].addHandler('canvas-drag', (event) => {
                    if (!event.originalEvent.shiftKey) event.preventDefaultAction = true;
                });
                newwin.projectUtils._activeState = projectUtils._activeState;
                try {
                    newwin.interfaceUtils.generateDataTabUI({uid:Feature_Space._dataset,name:"UMAP"})
                } catch (error) {
                }
                newwin.dataUtils.data[Feature_Space._dataset]={};
                copyDataset(dataUtils.data[Feature_Space._dataset], newwin.dataUtils.data[Feature_Space._dataset]);
                
                newwin.dataUtils.createMenuFromCSV(Feature_Space._dataset, newwin.dataUtils.data[Feature_Space._dataset]["_processeddata"].columns);

                let main_button = newwin.document.getElementById("ISS_collapse_btn");
                main_button.classList.add("d-none");
                newwin.interfaceUtils.toggleRightPanel();
                let main_navbar = newwin.document.getElementsByTagName("nav")[0];
                main_navbar.classList.add("d-none");
                newwin.tmapp.ISS_viewer.close();
                Feature_Space_Control.classList.remove("d-none");
                newwin.document.getElementsByClassName("navigator ")[0].classList.add("d-none");
                setTimeout(function() {
                    var tempfunc = glUtils.draw();;

                    glUtils.draw = function() {
                        tempfunc();
                        setTimeout(function(){
                            copyDataset(dataUtils.data[Feature_Space._dataset], newwin.dataUtils.data[Feature_Space._dataset]);
                            $("."+Feature_Space._dataset + "-marker-input, ."+Feature_Space._dataset + "-marker-hidden, ."+Feature_Space._dataset + "-marker-color, ."+Feature_Space._dataset + "-marker-shape").each(function(i, elt) {
                                newwin.document.getElementById(elt.id).value = elt.value;
                                newwin.document.getElementById(elt.id).checked = elt.checked;
                            }).promise().done(function(){
                                newwin.glUtils.loadMarkers(Feature_Space._dataset);
                                newwin.glUtils.draw();
                            });
                        },100);
                    }
                },200);
            }, 200);
        });
    }
    Feature_Space_Control.classList.add("d-none");
    Feature_Space_Control.setAttribute("src", "/");
}

Feature_Space.pressHandler = function (event) {
    console.log(event, event.originalEvent, event.originalEvent.shiftKey);
    var OSDviewer = Feature_Space._newwin.tmapp[tmapp["object_prefix"] + "_viewer"];

    if (! event.originalEvent.shiftKey) {
        Feature_Space._newwin.tmapp.ISS_viewer.gestureSettingsMouse.dragToPan = false;
        var normCoords = OSDviewer.viewport.pointFromPixel(event.position);
        var nextpoint = [normCoords.x, normCoords.y];
        Feature_Space._region = [normCoords];
        Feature_Space._regionPixels = [event.position];
    }
    else {
        Feature_Space._newwin.tmapp.ISS_viewer.gestureSettingsMouse.dragToPan = true;
        Feature_Space._region == []
    }
    return
};

Feature_Space.releaseHandler = function (event) {
    if (Feature_Space._region == []) {
        return;
    }
    if (event.originalEvent.shiftKey) { return; }
    var OSDviewer = Feature_Space._newwin.tmapp[tmapp["object_prefix"] + "_viewer"];

    var canvas = Feature_Space._newwin.overlayUtils._d3nodes[Feature_Space._newwin.tmapp["object_prefix"] + "_regions_svgnode"].node();
    var regionobj = d3.select(canvas).append('g').attr('class', "_UMAP_region");
    var elements = Feature_Space._newwin.document.getElementsByClassName("region_UMAP")
    if (elements.length > 0)
        elements[0].parentNode.removeChild(elements[0]);

    Feature_Space._region.push(Feature_Space._region[0]);

    regionobj.append('path').attr("d", regionUtils.pointsToPath([[Feature_Space._region]]))
        .attr("id", "path_UMAP")
        .attr('class', "region_UMAP")
        .style('stroke-width', 0.005)
        .style("stroke", '#aaaaaa').style("fill", "none")
    
    var pointsIn = Feature_Space.analyzeRegion(Feature_Space._region);
    var scalePropertyName = "UMAP_Region"
    Feature_Space._newwin.dataUtils.data[Feature_Space._dataset]["_scale_col"] = scalePropertyName;
    dataUtils.data[Feature_Space._dataset]["_scale_col"] = scalePropertyName;
    var markerData = Feature_Space._newwin.dataUtils.data[Feature_Space._dataset]["_processeddata"];
    markerData[scalePropertyName] = new Float64Array(markerData[Feature_Space._newwin.dataUtils.data[Feature_Space._dataset]["_X"]].length);
    if (pointsIn.length == 0) {
        markerData[scalePropertyName] = markerData[scalePropertyName].map(function() {return 1;});
    }
    for (var d of pointsIn) {
        markerData[scalePropertyName][d] = 1;
    }
    /*for (var scale in markerData[scalePropertyName]) {
        scale = 0
    for (var index of markerData[scalePropertyName].keys()) {
        index_ = markerData[""][index];
        if (pointsIn.indexOf(index_) == -1)
            markerData[scalePropertyName][index] = 0;
        else
            markerData[scalePropertyName][index] = 1;
    }*/
    
    Feature_Space._newwin.glUtils.loadMarkers(Feature_Space._dataset);
    Feature_Space._newwin.glUtils.draw();
    glUtils.loadMarkers(Feature_Space._dataset);
    glUtils.draw();
    Feature_Space._region = [];
    return
};

Feature_Space.moveHandler = function (event) {
    if (event.buttons != 1 || Feature_Space._region == []){ //|| !event.shift) {
        //Feature_Space._region = [];
        //Feature_Space._regionPixels = [];
        //Feature_Space._newwin.tmapp.ISS_viewer.setMouseNavEnabled(true);
        return;
    }
    if (event.originalEvent.shiftKey) { return; }
    var OSDviewer = Feature_Space._newwin.tmapp[tmapp["object_prefix"] + "_viewer"];

    var normCoords = OSDviewer.viewport.pointFromPixel(event.position);
    
    var nextpoint = normCoords;//[normCoords.x, normCoords.y];
    Feature_Space._regionPixels.push(event.position);
    function distance(a, b) {
        return Math.hypot(a.x-b.x, a.y-b.y)
    }
    if (Feature_Space._regionPixels.length > 1) {
        dis = distance(Feature_Space._regionPixels[Feature_Space._regionPixels.length-1], Feature_Space._regionPixels[Feature_Space._regionPixels.length-2])
        if (dis < 5) {
            Feature_Space._regionPixels.pop();
            return;
        }
    }
    Feature_Space._region.push(nextpoint);
    var canvas = Feature_Space._newwin.overlayUtils._d3nodes[Feature_Space._newwin.tmapp["object_prefix"] + "_regions_svgnode"].node();
    var regionobj = d3.select(canvas).append('g').attr('class', "_UMAP_region");
    var elements = Feature_Space._newwin.document.getElementsByClassName("region_UMAP")
    for (var element of elements)
        element.parentNode.removeChild(element);

    var polyline = regionobj.append('polyline').attr('points', Feature_Space._region.map(function(x){return [x.x,x.y];}))
        .style('fill', 'none')
        .attr('stroke-width', 0.005)
        .attr('stroke', '#aaaaaa').attr('class', "region_UMAP");
    return;
};

Feature_Space.analyzeRegion = function (points) {
    var op = Feature_Space._newwin.tmapp["object_prefix"];

    function clone(obj) {
        if (null == obj || "object" != typeof obj) return obj;
        var copy = obj.constructor();
        for (var attr in obj) {
            if (obj.hasOwnProperty(attr)) copy[attr] = obj[attr];
        }
        return copy;
    }

    associatedPoints=[];
    allDatasets = Object.keys(Feature_Space._newwin.dataUtils.data);
    var pointsInside=[];
    for (var dataset of allDatasets) {
        var allkeys=Object.keys(Feature_Space._newwin.dataUtils.data[dataset]["_groupgarden"]);
        for (var codeIndex in allkeys) {
            var code = allkeys[codeIndex];
            
            var quadtree = Feature_Space._newwin.dataUtils.data[dataset]["_groupgarden"][code]
            var imageWidth = Feature_Space._newwin.OSDViewerUtils.getImageWidth();
            var x0 = Math.min(...points.map(function(x){return x.x})) * imageWidth;
            var y0 = Math.min(...points.map(function(x){return x.y})) * imageWidth;
            var x3 = Math.max(...points.map(function(x){return x.x})) * imageWidth;
            var y3 = Math.max(...points.map(function(x){return x.y})) * imageWidth;
            var options = {
                "globalCoords":true,
                "xselector":Feature_Space._newwin.dataUtils.data[dataset]["_X"],
                "yselector":Feature_Space._newwin.dataUtils.data[dataset]["_Y"],
                "dataset":dataset
            }
            var xselector = options.xselector;
            var yselector = options.yselector;
            var imageWidth = Feature_Space._newwin.OSDViewerUtils.getImageWidth();
            var countsInsideRegion = 0;
            regionPath=Feature_Space._newwin.document.getElementById("path_UMAP");
            var svgovname = Feature_Space._newwin.tmapp["object_prefix"] + "_svgov";
            var svg = Feature_Space._newwin.tmapp[svgovname]._svg;
            tmpPoint = svg.createSVGPoint();
            pointInBbox = Feature_Space.searchTreeForPointsInBbox(quadtree, x0, y0, x3, y3, options);
            markerData = Feature_Space._newwin.dataUtils.data[dataset]["_processeddata"];
            for (var d of pointInBbox) {
                var x = markerData[xselector][d];
                var y = markerData[yselector][d];
                if (Feature_Space._newwin.regionUtils.globalPointInPath(x / imageWidth, y / imageWidth, regionPath, tmpPoint)) {
                    countsInsideRegion += 1;
                    pointsInside.push(d);
                }
            }
        }
    }
    return pointsInside;
}


/** 
 *  @param {Object} quadtree d3.quadtree where the points are stored
 *  @param {Number} x0 X coordinate of one point in a bounding box
 *  @param {Number} y0 Y coordinate of one point in a bounding box
 *  @param {Number} x3 X coordinate of diagonal point in a bounding box
 *  @param {Number} y3 Y coordinate of diagonal point in a bounding box
 *  @param {Object} options Tell the function 
 *  Search for points inside a particular region */
 Feature_Space.searchTreeForPointsInBbox = function (quadtree, x0, y0, x3, y3, options) {    
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
                    pointsInside.push(d);
                }
            }
        }
        return x1 >= x3 || y1 >= y3 || x2 < x0 || y2 < y0;
    });
    return pointsInside;
 }