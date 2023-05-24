/**
 * @file Plot_Histogram.js
 * @author Christophe Avenel
 */

/**
 * @namespace Plot_Histogram
 * @classdesc The root namespace for Plot_Histogram.
 */
var Plot_Histogram;
Plot_Histogram = {
  name: "Plot_Histogram Plugin",
  _dataset: null,
  _region: null,
  _regionPixels: null,
  _showHisto: true,
  _histoKey: false,
  _nbrow: 30,
  _isInit: false,
};

/**
 * @summary */
Plot_Histogram.init = function (container) {
  var script = document.createElement("script");
  script.src = "https://cdn.plot.ly/plotly-2.9.0.min.js";
  document.head.appendChild(script);
  row1 = HTMLElementUtils.createRow({});
  col11 = HTMLElementUtils.createColumn({ width: 12 });
  button111 = HTMLElementUtils.createButton({
    extraAttributes: { class: "btn btn-primary mx-2" },
  });
  button111.innerText = "Refresh drop-down lists based on loaded markers";

  row2 = HTMLElementUtils.createRow({});
  col21 = HTMLElementUtils.createColumn({ width: 12 });
  select211 = HTMLElementUtils.createElement({
    kind: "select",
    id: "Plot_Histogram_dataset",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label212 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "Plot_Histogram_dataset" },
  });
  label212.innerText = "Select marker dataset";

  row7 = HTMLElementUtils.createRow({});
  col71 = HTMLElementUtils.createColumn({ width: 12 });
  select711 = HTMLElementUtils.createElement({
    kind: "select",
    id: "Plot_Histogram_histoKey",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label712 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "Plot_Histogram_histoKey" },
  });
  label712.innerText = "Select Histogram Key";

  button111.addEventListener("click", (event) => {
    interfaceUtils.cleanSelect("Plot_Histogram_dataset");
    interfaceUtils.cleanSelect("Plot_Histogram_histoKey");

    var datasets = Object.keys(dataUtils.data).map(function (e, i) {
      return {
        value: e,
        innerHTML: document.getElementById(e + "_tab-name").value,
      };
    });
    interfaceUtils.addObjectsToSelect("Plot_Histogram_dataset", datasets);
    var event = new Event("change");
    interfaceUtils
      .getElementById("Plot_Histogram_dataset")
      .dispatchEvent(event);
  });
  select211.addEventListener("change", (event) => {
    Plot_Histogram._dataset = select211.value;
    if (!dataUtils.data[Plot_Histogram._dataset]) return;
    interfaceUtils.cleanSelect("Plot_Histogram_histoKey");
    interfaceUtils.addElementsToSelect(
      "Plot_Histogram_histoKey",
      dataUtils.data[Plot_Histogram._dataset]._csv_header
    );
    if (
      dataUtils.data[Plot_Histogram._dataset]._csv_header.indexOf(
        dataUtils.data[Plot_Histogram._dataset]._gb_col
      ) > 0
    ) {
      interfaceUtils.getElementById("Plot_Histogram_histoKey").value =
        dataUtils.data[Plot_Histogram._dataset]._gb_col;
      var event = new Event("change");
      interfaceUtils
        .getElementById("Plot_Histogram_histoKey")
        .dispatchEvent(event);
    }
  });
  select711.addEventListener("change", (event) => {
    Plot_Histogram._histoKey = select711.value;
    Plot_Histogram.run();
    var pointsIn = Plot_Histogram.analyzeRegion(Plot_Histogram._region);
    Plot_Histogram.getHisto();
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
  container.appendChild(row7);
  row7.appendChild(col71);
  col71.appendChild(label712);
  col71.appendChild(select711);
  var event = new Event("click");
  button111.dispatchEvent(event);

  var textInfo = document.createElement("div");
  textInfo.style.marginTop = "10px";
  textInfo.innerHTML = "Hold shift to draw a region on the image.";
  container.appendChild(textInfo);

  var plotHistoView = document.createElement("div");
  plotHistoView.id = "plotHistoView";
  plotHistoView.style.marginTop = "20px";
  container.appendChild(plotHistoView);
};

Plot_Histogram.run = function () {
  var op = tmapp["object_prefix"];
  var vname = op + "_viewer";
  if (Plot_Histogram._isInit) return;
  //OSD handlers are not registered manually they have to be registered
  //using MouseTracker OSD objects
  Plot_Histogram._isInit = true;
  new OpenSeadragon.MouseTracker({
    element: tmapp[vname].canvas,
    moveHandler: Plot_Histogram.moveHandler /*,
        pressHandler: Plot_Histogram.pressHandler,
        releaseHandler: Plot_Histogram.releaseHandler*/,
  }).setTracking(true);

  tmapp["ISS_viewer"].addHandler("canvas-press", (event) => {
    Plot_Histogram.pressHandler(event);
  });
  tmapp["ISS_viewer"].addHandler("canvas-release", (event) => {
    Plot_Histogram.releaseHandler(event);
  });
  tmapp["ISS_viewer"].addHandler("canvas-drag", (event) => {
    if (event.originalEvent.shiftKey) event.preventDefaultAction = true;
  });
  tmapp["ISS_viewer"].addHandler(
    "animation-finish",
    function animationFinishHandler(event) {
      console.log(d3.selectAll(".region_histo"));
      d3.selectAll(".region_histo")
        .selectAll("polyline")
        .each(function (el) {
          $(this).attr(
            "stroke-width",
            regionUtils._polygonStrokeWidth /
              tmapp["ISS_viewer"].viewport.getZoom()
          );
        });
      d3.selectAll(".region_histo")
        .selectAll("circle")
        .each(function (el) {
          $(this).attr(
            "r",
            (10 * regionUtils._handleRadius) /
              tmapp["ISS_viewer"].viewport.getZoom()
          );
        });
      d3.selectAll(".region_histo").each(function (el) {
        $(this).attr(
          "stroke-width",
          regionUtils._polygonStrokeWidth /
            tmapp["ISS_viewer"].viewport.getZoom()
        );
      });
    }
  );
};

Plot_Histogram.pressHandler = function (event) {
  var OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];

  if (event.originalEvent.shiftKey) {
    tmapp.ISS_viewer.gestureSettingsMouse.dragToPan = false;
    var normCoords = OSDviewer.viewport.pointFromPixel(event.position);
    var nextpoint = [normCoords.x, normCoords.y];
    Plot_Histogram._region = [normCoords];
    Plot_Histogram._regionPixels = [event.position];
  } else {
    tmapp.ISS_viewer.gestureSettingsMouse.dragToPan = true;
  }
  return;
};

Plot_Histogram.releaseHandler = function (event) {
  if (Plot_Histogram._region == []) {
    return;
  }
  if (!event.originalEvent.shiftKey) {
    return;
  }
  var OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];

  var canvas =
    overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
  var regionobj = d3.select(canvas).append("g").attr("class", "_UMAP_region");
  var elements = document.getElementsByClassName("region_histo");
  if (elements.length > 0) elements[0].parentNode.removeChild(elements[0]);

  Plot_Histogram._region.push(Plot_Histogram._region[0]);

  regionobj
    .append("path")
    .attr("d", regionUtils.pointsToPath([[Plot_Histogram._region]]))
    .attr("id", "path_UMAP")
    .attr("class", "region_histo")
    .attr(
      "stroke-width",
      regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom()
    )
    .style("stroke", "#ff0000")
    .style("fill", "none");

  var pointsIn = Plot_Histogram.analyzeRegion(Plot_Histogram._region);
  var scalePropertyName = "UMAP_Region_scale";
  dataUtils.data[Plot_Histogram._dataset]["_scale_col"] = scalePropertyName;
  dataUtils.data[Plot_Histogram._dataset]["_scale_col"] = scalePropertyName;
  var markerData = dataUtils.data[Plot_Histogram._dataset]["_processeddata"];
  markerData[scalePropertyName] = new Float64Array(
    markerData[dataUtils.data[Plot_Histogram._dataset]["_X"]].length
  );
  var opacityPropertyName = "UMAP_Region_opacity";
  dataUtils.data[Plot_Histogram._dataset]["_opacity_col"] = opacityPropertyName;
  dataUtils.data[Plot_Histogram._dataset]["_opacity_col"] = opacityPropertyName;
  markerData[opacityPropertyName] = new Float64Array(
    markerData[dataUtils.data[Plot_Histogram._dataset]["_X"]].length
  );
  markerData[opacityPropertyName] = markerData[opacityPropertyName].map(
    function () {
      return 0.3;
    }
  );
  markerData[scalePropertyName] = markerData[scalePropertyName].map(
    function () {
      return 0.3;
    }
  );
  if (pointsIn.length == 0) {
    markerData[scalePropertyName] = markerData[scalePropertyName].map(
      function () {
        return 1;
      }
    );
    markerData[opacityPropertyName] = markerData[opacityPropertyName].map(
      function () {
        return 1;
      }
    );
  }
  for (var d of pointsIn) {
    markerData[scalePropertyName][d] = 1;
    markerData[opacityPropertyName][d] = 1;
  }
  Plot_Histogram.getHisto();

  glUtils.loadMarkers(Plot_Histogram._dataset);
  glUtils.draw();
  return;
};

Plot_Histogram.moveHandler = function (event) {
  if (event.buttons != 1 || Plot_Histogram._region == []) {
    //|| !event.shift) {
    //Plot_Histogram._region = [];
    //Plot_Histogram._regionPixels = [];
    //tmapp.ISS_viewer.setMouseNavEnabled(true);
    return;
  }
  if (!event.originalEvent.shiftKey) {
    return;
  }
  var OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];

  var normCoords = OSDviewer.viewport.pointFromPixel(event.position);

  var nextpoint = normCoords; //[normCoords.x, normCoords.y];
  Plot_Histogram._regionPixels.push(event.position);
  function distance(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }
  if (Plot_Histogram._regionPixels.length > 1) {
    dis = distance(
      Plot_Histogram._regionPixels[Plot_Histogram._regionPixels.length - 1],
      Plot_Histogram._regionPixels[Plot_Histogram._regionPixels.length - 2]
    );
    if (dis < 5) {
      Plot_Histogram._regionPixels.pop();
      return;
    }
  }
  Plot_Histogram._region.push(nextpoint);
  var canvas =
    overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
  var regionobj = d3.select(canvas).append("g").attr("class", "_UMAP_region");
  var elements = document.getElementsByClassName("region_histo");
  for (var element of elements) element.parentNode.removeChild(element);

  var polyline = regionobj
    .append("polyline")
    .attr(
      "points",
      Plot_Histogram._region.map(function (x) {
        return [x.x, x.y];
      })
    )
    .style("fill", "none")
    .attr(
      "stroke-width",
      regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom()
    )
    .attr("stroke", "#ff0000")
    .attr("class", "region_histo");
  return;
};

Plot_Histogram.analyzeRegion = function (points) {
  var associatedPoints = [];
  _histogram = [];
  var pointsInside = [];
  var dataset = Plot_Histogram._dataset;
  var allkeys = Object.keys(dataUtils.data[dataset]["_groupgarden"]);
  var countsInsideRegion = {};
  for (var codeIndex in allkeys) {
    var code = allkeys[codeIndex];

    var quadtree = dataUtils.data[dataset]["_groupgarden"][code];
    var imageWidth = OSDViewerUtils.getImageWidth();
    var x0 =
      Math.min(
        ...points.map(function (x) {
          return x.x;
        })
      ) * imageWidth;
    var y0 =
      Math.min(
        ...points.map(function (x) {
          return x.y;
        })
      ) * imageWidth;
    var x3 =
      Math.max(
        ...points.map(function (x) {
          return x.x;
        })
      ) * imageWidth;
    var y3 =
      Math.max(
        ...points.map(function (x) {
          return x.y;
        })
      ) * imageWidth;
    var options = {
      globalCoords: true,
      xselector: dataUtils.data[dataset]["_X"],
      yselector: dataUtils.data[dataset]["_Y"],
      dataset: dataset,
    };
    var xselector = options.xselector;
    var yselector = options.yselector;
    var imageWidth = OSDViewerUtils.getImageWidth();
    var regionPath = document.getElementById("path_UMAP");
    var svgovname = tmapp["object_prefix"] + "_svgov";
    var svg = tmapp[svgovname]._svg;
    var tmpPoint = svg.createSVGPoint();
    var inputs = interfaceUtils._mGenUIFuncs.getGroupInputs(dataset, code);
    var visible = "visible" in inputs ? inputs["visible"] : true;
    if (visible) {
      var pointInBbox = Plot_Histogram.searchTreeForPointsInBbox(
        quadtree,
        x0,
        y0,
        x3,
        y3,
        options
      );
      var markerData = dataUtils.data[dataset]["_processeddata"];
      for (var d of pointInBbox) {
        var x = markerData[xselector][d];
        var y = markerData[yselector][d];
        var key;
        if (
          regionUtils.globalPointInPath(
            x / imageWidth,
            y / imageWidth,
            regionPath,
            tmpPoint
          )
        ) {
          if (Plot_Histogram._histoKey) {
            key = markerData[Plot_Histogram._histoKey][d];
          } else {
            key = quadtree.treeName;
          }
          if (countsInsideRegion[key] === undefined) {
            countsInsideRegion[key] = 0;
          }
          countsInsideRegion[key] += 1;
          pointsInside.push(d);
        }
      }
    }
  }
  for (var key in countsInsideRegion) {
    var hexColor;
    if (
      !Plot_Histogram._histoKey ||
      Plot_Histogram._histoKey == dataUtils.data[dataset]._gb_col
    ) {
      var inputs = interfaceUtils._mGenUIFuncs.getGroupInputs(dataset, key);
      hexColor = "color" in inputs ? inputs["color"] : "#ff0000";
    } else {
      hexColor = "#ff0000";
    }
    _histogram.push({
      key: key,
      name: key,
      count: countsInsideRegion[key],
      color: hexColor,
    });
  }
  function compare(a, b) {
    if (a.count > b.count) return -1;
    if (a.count < b.count) return 1;
    return 0;
  }
  _histogram.sort(compare);
  return pointsInside;
};

/**
 *  @param {Object} quadtree d3.quadtree where the points are stored
 *  @param {Number} x0 X coordinate of one point in a bounding box
 *  @param {Number} y0 Y coordinate of one point in a bounding box
 *  @param {Number} x3 X coordinate of diagonal point in a bounding box
 *  @param {Number} y3 Y coordinate of diagonal point in a bounding box
 *  @param {Object} options Tell the function
 *  Search for points inside a particular region */
Plot_Histogram.searchTreeForPointsInBbox = function (
  quadtree,
  x0,
  y0,
  x3,
  y3,
  options
) {
  if (options.globalCoords) {
    var xselector = options.xselector;
    var yselector = options.yselector;
  } else {
    throw {
      name: "NotImplementedError",
      message: "ViewerPointInPath not yet implemented.",
    };
  }
  var pointsInside = [];
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
};

Plot_Histogram.getHisto = function () {
  var op = tmapp["object_prefix"];
  var vname = op + "_viewer";

  plotHistoView = document.getElementById("plotHistoView");
  if (_histogram === undefined) {
    plotHistoView.innerHTML = "";
    return;
  }
  if (Object.keys(_histogram).length == 0) {
    plotHistoView.innerHTML = "";
    return;
  }
  if (_histogram[0].count == 0) {
    plotHistoView.innerHTML = "";
    return;
  }
  var histogram = _histogram.slice(0, Plot_Histogram._nbrow).reverse();
  Plot_Histogram._plot = Plotly.newPlot(
    plotHistoView,
    [
      {
        y: histogram.map(function (x) {
          return x.key + " -";
        }),
        x: histogram.map(function (x) {
          return x.count;
        }),
        type: "bar",
        orientation: "h",
        marker: {
          color: histogram.map(function (x) {
            return x.color;
          }),
        },
      },
    ],
    {
      margin: { t: 0, r: 0, b: 20, l: 20 },
      height: 18 * (histogram.length + 1),
      yaxis: {
        automargin: true,
      },
    },
    { responsive: true, displayModeBar: false }
  );
};
