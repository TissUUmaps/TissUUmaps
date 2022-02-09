/**
 * @file UMAP.js
 * @author Christophe Avenel
 */

/**
 * @namespace UMAP
 * @classdesc The root namespace for UMAP.
 */
 var UMAP;
 UMAP = {
     name:"UMAP Plugin",
     _dataset:null,
     _UMAP_X:null,
     _UMAP_Y:null,
     _region:null,
     _regionPixels:null,
     _newwin:null
  }
 
 /**
  * @summary */
 UMAP.init = function (container) {
    row1=HTMLElementUtils.createRow({});
        col11=HTMLElementUtils.createColumn({"width":12});
            button111=HTMLElementUtils.createButton({"extraAttributes":{ "class":"btn btn-primary mx-2"}})
            button111.innerText = "Update marker datasets";

    row2=HTMLElementUtils.createRow({});
        col21=HTMLElementUtils.createColumn({"width":12});
            select211=HTMLElementUtils.createElement({"kind":"select","id":"UMAP_dataset","extraAttributes":{"class":"form-select form-select-sm","aria-label":".form-select-sm"}});
            label212=HTMLElementUtils.createElement({"kind":"label", "extraAttributes":{"for":"UMAP_dataset"} });
            label212.innerText="Select marker dataset";

    row3=HTMLElementUtils.createRow({});
        col31=HTMLElementUtils.createColumn({"width":12});
            select311=HTMLElementUtils.createElement({"kind":"select","id":"UMAP_X","extraAttributes":{"class":"form-select form-select-sm","aria-label":".form-select-sm"}});
            label312=HTMLElementUtils.createElement({"kind":"label", "extraAttributes":{"for":"UMAP_X"} });
            label312.innerText="Select UMAP X";

    row4=HTMLElementUtils.createRow({});
        col41=HTMLElementUtils.createColumn({"width":12});
            select411=HTMLElementUtils.createElement({"kind":"select","id":"UMAP_Y","extraAttributes":{"class":"form-select form-select-sm","aria-label":".form-select-sm"}});
            label412=HTMLElementUtils.createElement({"kind":"label", "extraAttributes":{"for":"UMAP_Y"} });
            label412.innerText="Select UMAP Y";

    row5=HTMLElementUtils.createRow({});
        col51=HTMLElementUtils.createColumn({"width":12});
            button511=HTMLElementUtils.createButton({"extraAttributes":{ "class":"btn btn-primary mx-2"}})
            button511.innerText = "Show UMAP";
    
    button111.addEventListener("click",(event)=>{
        interfaceUtils.cleanSelect("UMAP_dataset");
        interfaceUtils.cleanSelect("UMAP_X");
        interfaceUtils.cleanSelect("UMAP_Y");
        var datasets = Object.keys(dataUtils.data).map(function(e, i) {
            return {value:e, innerHTML:document.getElementById(e + "_tab-name").value};
        });
        interfaceUtils.addObjectsToSelect("UMAP_dataset", datasets);
        var event = new Event('change');
        interfaceUtils.getElementById("UMAP_dataset").dispatchEvent(event);
    });

    select211.addEventListener("change",(event)=>{
        UMAP._dataset = select211.value;
        interfaceUtils.cleanSelect("UMAP_X");
        interfaceUtils.addElementsToSelect("UMAP_X", dataUtils.data[UMAP._dataset]._csv_header);
        interfaceUtils.cleanSelect("UMAP_Y");
        interfaceUtils.addElementsToSelect("UMAP_Y", dataUtils.data[UMAP._dataset]._csv_header);
        if (dataUtils.data[UMAP._dataset]._csv_header.indexOf("UMAP_X") > 0) {
            interfaceUtils.getElementById("UMAP_X").value = "UMAP_X";
            var event = new Event('change');
            interfaceUtils.getElementById("UMAP_X").dispatchEvent(event);
        }
        if (dataUtils.data[UMAP._dataset]._csv_header.indexOf("UMAP_Y") > 0) {
            interfaceUtils.getElementById("UMAP_Y").value = "UMAP_Y";
            var event = new Event('change');
            interfaceUtils.getElementById("UMAP_Y").dispatchEvent(event);
        }
    });
    select311.addEventListener("change",(event)=>{
        UMAP._UMAP_X = select311.value;
    });
    select411.addEventListener("change",(event)=>{
        UMAP._UMAP_Y = select411.value;
    });

    button511.addEventListener("click",(event)=>{
        UMAP.run();
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
    var headers = interfaceUtils._mGenUIFuncs.getTabDropDowns(UMAP._dataset);
    dataOut["expectedHeader"] = Object.assign({}, ...Object.keys(headers).map((k) => ({[k]: headers[k].value})));
    var radios = interfaceUtils._mGenUIFuncs.getTabRadiosAndChecks(UMAP._dataset);
    dataOut["expectedRadios"] = Object.assign({}, ...Object.keys(radios).map((k) => ({[k]: radios[k].checked})));
    dataOut["expectedHeader"]["X"] = UMAP._UMAP_X;
    dataOut["expectedHeader"]["Y"] = UMAP._UMAP_Y;
    for (var key of Object.keys(dataIn)) {
        if (["_X","_Y","expectedHeader","expectedRadios"].indexOf(key) == -1) {
            dataOut[key] = dataIn[key];
        }
        else if (key == "_X") {
            dataOut[key] = UMAP._UMAP_X;
        }
        else if (key == "_Y") {
            dataOut[key] = UMAP._UMAP_Y;
        }
    }
}

UMAP.run = function () {
    var op = tmapp["object_prefix"];
    var vname = op + "_viewer";
    var UMAPControl = document.getElementById("UMAPControl");
    console.log("UMAPControl", UMAPControl)
    if (!UMAPControl) {
        UMAPControl = document.createElement("iframe");
        UMAPControl.id = "UMAPControl";
        UMAPControl.style.width= "100%";
        UMAPControl.style.height= "100%";
        var elt = document.createElement("div");
        elt.style.width= "40%";
        elt.style.height= "100%";
        elt.style.display = "inline-flex";
        elt.appendChild(UMAPControl);
        document.getElementById("ISS_viewer").appendChild(elt);
        $(".openseadragon-container")[0].style.display = "inline-flex";
        $(".openseadragon-container")[0].style.width= "60%";
            
        UMAPControl.addEventListener("load", ev => {
            UMAPControl.classList.add("d-none");
            var timeout = setTimeout(function() {
                var newwin = UMAPControl.contentWindow;
                UMAP._newwin = newwin;
                //OSD handlers are not registered manually they have to be registered
                //using MouseTracker OSD objects 
                console.log("newwin",newwin);
                if (newwin.tmapp.ISS_viewer) {
                    clearInterval(timeout);
                }
                else {return;}
                console.log("canvas", UMAP._newwin.tmapp[vname].canvas)
                new UMAP._newwin.OpenSeadragon.MouseTracker({
                    element: UMAP._newwin.tmapp[vname].canvas,
                    pressHandler: UMAP.pressHandler,
                    moveHandler: UMAP.moveHandler,
                    releaseHandler: UMAP.releaseHandler
                }).setTracking(true);
                            
                UMAP._newwin.tmapp["ISS_viewer"].addHandler('canvas-drag', (event) => {
                    event.preventDefaultAction = true;
                });
                /*OpenSeadragon.MouseTracker({
                    element: UMAP._newwin.tmapp[vname].canvas,
                    pressHandler: UMAP.pressHandler,
                    moveHandler: UMAP.moveHandler,
                    releaseHandler: UMAP.releaseHandler
                }).setTracking(true);*/
                
                /*UMAP._newwin.tmapp["ISS_viewer"].addHandler('canvas-press', UMAP.clickHandler);
                UMAP._newwin.tmapp["ISS_viewer"].addHandler('canvas-drag', UMAP.moveHandler);
                UMAP._newwin.tmapp["ISS_viewer"].addHandler('canvas-dragend', UMAP.releaseHandler);*/

                newwin.projectUtils._activeState = projectUtils._activeState;
                try {
                    newwin.interfaceUtils.generateDataTabUI({uid:UMAP._dataset,name:"UMAP"})
                } catch (error) {
                }
                newwin.dataUtils.data[UMAP._dataset]={};
                copyDataset(dataUtils.data[UMAP._dataset], newwin.dataUtils.data[UMAP._dataset]);
                
                newwin.dataUtils.createMenuFromCSV(UMAP._dataset, newwin.dataUtils.data[UMAP._dataset]["_processeddata"].columns);

                let main_button = newwin.document.getElementById("ISS_collapse_btn");
                main_button.classList.add("d-none");
                newwin.interfaceUtils.toggleRightPanel();
                let main_navbar = newwin.document.getElementsByTagName("nav")[0];
                main_navbar.classList.add("d-none");
                newwin.tmapp.ISS_viewer.close();
                UMAPControl.classList.remove("d-none");
                setTimeout(function() {
                    $(document).mouseup(function(e) {
                        setTimeout(function(){
                            copyDataset(dataUtils.data[UMAP._dataset], newwin.dataUtils.data[UMAP._dataset]);
                            $("."+UMAP._dataset + "-marker-input, ."+UMAP._dataset + "-marker-hidden, ."+UMAP._dataset + "-marker-color, ."+UMAP._dataset + "-marker-shape").each(function(i, elt) {
                                newwin.document.getElementById(elt.id).value = elt.value;
                                newwin.document.getElementById(elt.id).checked = elt.checked;
                            }).promise().done(function(){
                                newwin.glUtils.loadMarkers(UMAP._dataset);
                                newwin.glUtils.draw();
                            });
                        },100);
                    });
                    $(document).mousedown(function(e) {
                        setTimeout(function(){
                            copyDataset(dataUtils.data[UMAP._dataset], newwin.dataUtils.data[UMAP._dataset]);
                            $("."+UMAP._dataset + "-marker-input, ."+UMAP._dataset + "-marker-hidden, ."+UMAP._dataset + "-marker-color, ."+UMAP._dataset + "-marker-shape").each(function(i, elt) {
                                newwin.document.getElementById(elt.id).value = elt.value;
                                newwin.document.getElementById(elt.id).checked = elt.checked;
                            }).promise().done(function(){
                                newwin.glUtils.loadMarkers(UMAP._dataset);
                                newwin.glUtils.draw();
                            });
                        },100);
                    });

                },200);
            }, 200);
        });
    }
    UMAPControl.classList.add("d-none");
    UMAPControl.setAttribute("src", "/");
}

UMAP.pressHandler = function (event) {
    var OSDviewer = UMAP._newwin.tmapp[tmapp["object_prefix"] + "_viewer"];

    //if (event.shift) {
        UMAP._newwin.tmapp.ISS_viewer.gestureSettingsMouse.dragToPan = false;
        var normCoords = OSDviewer.viewport.pointFromPixel(event.position);
        var nextpoint = [normCoords.x, normCoords.y];
        UMAP._region = [normCoords];
        UMAP._regionPixels = [event.position];
    //
    return
};

UMAP.releaseHandler = function (event) {
    if (UMAP._region == []) {
        return;
    }
    var OSDviewer = UMAP._newwin.tmapp[tmapp["object_prefix"] + "_viewer"];

    var canvas = UMAP._newwin.overlayUtils._d3nodes[UMAP._newwin.tmapp["object_prefix"] + "_regions_svgnode"].node();
    var regionobj = d3.select(canvas).append('g').attr('class', "_UMAP_region");
    var elements = UMAP._newwin.document.getElementsByClassName("region_UMAP")
    if (elements.length > 0)
        elements[0].parentNode.removeChild(elements[0]);

    UMAP._region.push(UMAP._region[0]);

    regionobj.append('path').attr("d", regionUtils.pointsToPath([[UMAP._region]]))
        .attr("id", "path_UMAP")
        .attr('class', "region_UMAP")
        .style('stroke-width', 0.005)
        .style("stroke", '#aaaaaa').style("fill", "none")
    //UMAP._region = [];
    
    var pointsIn = UMAP.analyzeRegion(UMAP._region);
    var scalePropertyName = "UMAP_Region"
    UMAP._newwin.dataUtils.data[UMAP._dataset]["_scale_col"] = scalePropertyName;
    dataUtils.data[UMAP._dataset]["_scale_col"] = scalePropertyName;
    var markerData = UMAP._newwin.dataUtils.data[UMAP._dataset]["_processeddata"];
    markerData[scalePropertyName] = new Float64Array(markerData[""].length);
    if (pointsIn.length == 0) pointsIn = markerData[""]
    console.log("pointsIn", pointsIn);
    for (var index of markerData[scalePropertyName].keys()) {
        if (pointsIn.indexOf(index) == -1)
            markerData[scalePropertyName][index] = 0;
        else
            markerData[scalePropertyName][index] = 1;
    }
    
    UMAP._newwin.glUtils.loadMarkers(UMAP._dataset);
    UMAP._newwin.glUtils.draw();
    glUtils.loadMarkers(UMAP._dataset);
    glUtils.draw();
    UMAP._region = [];
    return
};

UMAP.moveHandler = function (event) {
    console.log("move", event);
    if (event.buttons != 1 || UMAP._region == []){ //|| !event.shift) {
        //UMAP._region = [];
        //UMAP._regionPixels = [];
        //UMAP._newwin.tmapp.ISS_viewer.setMouseNavEnabled(true);
        return;
    }
    var OSDviewer = UMAP._newwin.tmapp[tmapp["object_prefix"] + "_viewer"];

    var normCoords = OSDviewer.viewport.pointFromPixel(event.position);
    
    var nextpoint = normCoords;//[normCoords.x, normCoords.y];
    UMAP._regionPixels.push(event.position);
    function distance(a, b) {
        return Math.hypot(a.x-b.x, a.y-b.y)
    }
    if (UMAP._regionPixels.length > 1) {
        dis = distance(UMAP._regionPixels[UMAP._regionPixels.length-1], UMAP._regionPixels[UMAP._regionPixels.length-2])
        if (dis < 5) {
            UMAP._regionPixels.pop();
            return;
        }
    }
    UMAP._region.push(nextpoint);
    var canvas = UMAP._newwin.overlayUtils._d3nodes[UMAP._newwin.tmapp["object_prefix"] + "_regions_svgnode"].node();
    var regionobj = d3.select(canvas).append('g').attr('class', "_UMAP_region");
    var elements = UMAP._newwin.document.getElementsByClassName("region_UMAP")
    for (var element of elements)
        element.parentNode.removeChild(element);

    var polyline = regionobj.append('polyline').attr('points', UMAP._region.map(function(x){return [x.x,x.y];}))
        .style('fill', 'none')
        .attr('stroke-width', 0.005)
        .attr('stroke', '#aaaaaa').attr('class', "region_UMAP");
    return;
};

UMAP.analyzeRegion = function (points) {
    var op = UMAP._newwin.tmapp["object_prefix"];

    function clone(obj) {
        if (null == obj || "object" != typeof obj) return obj;
        var copy = obj.constructor();
        for (var attr in obj) {
            if (obj.hasOwnProperty(attr)) copy[attr] = obj[attr];
        }
        return copy;
    }

    associatedPoints=[];
    allDatasets = Object.keys(UMAP._newwin.dataUtils.data);
    var pointsInside=[];
    for (var dataset of allDatasets) {
        var allkeys=Object.keys(UMAP._newwin.dataUtils.data[dataset]["_groupgarden"]);
        for (var codeIndex in allkeys) {
            var code = allkeys[codeIndex];
            
            var quadtree = UMAP._newwin.dataUtils.data[dataset]["_groupgarden"][code]
            var imageWidth = UMAP._newwin.OSDViewerUtils.getImageWidth();
            var x0 = 0;//Math.min(...points.map(function(x){return x.x})) * imageWidth;
            var y0 = 0;//Math.min(...points.map(function(x){return x.y})) * imageWidth;
            var x3 = imageWidth;//Math.max(...points.map(function(x){return x.x})) * imageWidth;
            var y3 = imageWidth;//Math.max(...points.map(function(x){return x.y})) * imageWidth;
            var options = {
                "globalCoords":true,
                "xselector":UMAP._newwin.dataUtils.data[dataset]["_X"],
                "yselector":UMAP._newwin.dataUtils.data[dataset]["_Y"],
                "dataset":dataset
            }
            var xselector = options.xselector;
            var yselector = options.yselector;
            var imageWidth = UMAP._newwin.OSDViewerUtils.getImageWidth();
            var countsInsideRegion = 0;
            regionPath=UMAP._newwin.document.getElementById("path_UMAP");
            var svgovname = UMAP._newwin.tmapp["object_prefix"] + "_svgov";
            var svg = UMAP._newwin.tmapp[svgovname]._svg;
            tmpPoint = svg.createSVGPoint();
            pointInBbox = UMAP._newwin.regionUtils.searchTreeForPointsInBbox(quadtree, x0, y0, x3, y3, options);
            for (d of pointInBbox) {
                if (UMAP._newwin.regionUtils.globalPointInPath(d[xselector] / imageWidth, d[yselector] / imageWidth, regionPath, tmpPoint)) {
                    countsInsideRegion += 1;
                    pointsInside.push(d[""]);
                }
            }
        }
    }
    return pointsInside;
}