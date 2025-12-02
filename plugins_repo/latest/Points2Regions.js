/**
 * @file Points2Regions.js
 * @author Axel Andersson, Christophe Avenel
 */

/**
 * @namespace Points2Regions
 * @classdesc The root namespace for Points2Regions.
 */
var Points2Regions;
Points2Regions = {
  name: "Points2Regions Plugin",
  parameters: {
    _nclusters: {
      label: "Number of clusters (default 8):",
      type: "number",
      default: 8,
    },
    _min_pts_per_pixel: {
      label:
        "Min points per pixel (increase to avoid regions with few markers):",
      type: "number",
      default: 0,
    },
    _pixel_size: {
      label: "Pixel size (increase/decrease for coarser/finer regions):",
      type: "number",
      default: 1,
    },
    _pixel_smoothing: {
      label:
        "Smoothing (increase to aggregate information over larger distances):",
      type: "number",
      default: 5,
    },
    _selectStride: {
      label: "Select pixel size on tissue (optional)",
      type: "button",
    },
    _run: {
      label: "Run Points2Regions",
      type: "button",
    },
    _downloadCSV: {
      label: "Download data as CSV",
      type: "button",
    },
    _advancedSection: {
      label: "Only change these settings if you know what you are doing!",
      title: "Advanced settings",
      type: "section",
      collapsed: true,
    },
    _refresh: {
      label: "Refresh drop-down lists based on loaded markers",
      type: "button",
    },
    _dataset: {
      label: "Select marker dataset:",
      type: "select",
      default: "lol",
    },
    _clusterKey: {
      label: "Select Points2Regions Key:",
      type: "select",
    },
    _seed: {
      label: "Random seed (used during KMeans):",
      type: "number",
      default: 0,
      attributes: { step: 1 },
    },
    _format: {
      label: "Output regions as",
      type: "select",
      default: "New label per marker",
      options: ["GeoJSON polygons", "New label per marker"],
    },
    _server: {
      label: "Run Points2Regions on the server",
      type: "checkbox",
      default: true,
    },
  },
  _region_name: "Clusters",
};

/**
 * @summary */
Points2Regions.init = function (container) {
  Points2Regions.inputTrigger("_refresh");
  Points2Regions.container = container;
  Points2Regions.initPython();
  Points2Regions._api(
    "checkServer",
    null,
    function (data) {
      console.log("Server check success:", data);
      if (data["return"] === "error") {
        Points2Regions.set("_server", false);
        Points2Regions.initPython();
        let serverCheckBoxID = Points2Regions.getInputID("_server");
        let serverCheckbox = document.getElementById(serverCheckBoxID);
        serverCheckbox.disabled = true;
        serverCheckbox.parentElement.title = data["message"];
        var tooltip = new bootstrap.Tooltip(serverCheckbox.parentElement, {
          placement: "right",
        });
        tooltip.enable();
      }
    },
    function (xhr, status, error) {
      console.log("Server check failed:", status, error);
      Points2Regions.set("_server", false);
      Points2Regions.initPython();
      let serverCheckBoxID = Points2Regions.getInputID("_server");
      let serverCheckbox = document.getElementById(serverCheckBoxID);
      serverCheckbox.disabled = true;
      serverCheckbox.parentElement.title =
        "Unable to run on server, check that you have all dependencies installed (scikit-learn).";
      var tooltip = new bootstrap.Tooltip(serverCheckbox.parentElement, {
        placement: "right",
      });
      tooltip.enable();
    },
  );
};

Points2Regions.run = function () {
  if (Points2Regions.get("_server")) {
    var csvFile = dataUtils.data[Points2Regions.get("_dataset")]._csv_path;
    if (typeof csvFile === "object") {
      interfaceUtils.alert(
        "This plugin can only run on datasets generated from buttons. Please convert your dataset to a button (Markers > Advanced Options > Generate button from tab)",
      );
      return;
    }
    // Get the path from url:
    if (dataUtils.data[Points2Regions.get("_dataset")]._filetype !== "h5") {
      const path = dataUtils.getPath();
      if (path !== null) {
        csvFile = path + "/" + csvFile;
      }
    }
    loadingModal = interfaceUtils.loadingModal(
      "Points2Regions... Please wait.",
    );
    console.log("test");
    $.ajax({
      type: "post",
      url: "/plugins/Points2Regions/Points2Regions",
      contentType: "application/json; charset=utf-8",
      data: JSON.stringify({
        xKey: dataUtils.data[Points2Regions.get("_dataset")]._X,
        yKey: dataUtils.data[Points2Regions.get("_dataset")]._Y,
        clusterKey: Points2Regions.get("_clusterKey"),
        nclusters: Points2Regions.get("_nclusters"),
        min_pts_per_pixel: Points2Regions.get("_min_pts_per_pixel"),
        pixel_size: Points2Regions.get("_pixel_size"),
        pixel_smoothing: Points2Regions.get("_pixel_smoothing"),
        region_name: Points2Regions.get("_region_name"),
        seed: Points2Regions.get("_seed"),
        format: Points2Regions.get("_format"),
        csv_path: csvFile,
        filetype: dataUtils.data[Points2Regions.get("_dataset")]._filetype,
      }),
      success: function (data) {
        if (Points2Regions.get("_format") === "GeoJSON polygons") {
          console.log(data);
          Points2Regions.loadRegions(data);
        } else {
          console.log(data);
          data = data.substring(1, data.length - 1);
          let clusters = data.split(",").map(function (x) {
            return parseInt(x);
          });
          console.log(clusters);
          Points2Regions.loadClusters(clusters);
        }
        setTimeout(function () {
          $(loadingModal).modal("hide");
        }, 500);
      },
      complete: function (data) {
        // do something, not critical.
      },
      error: function (data) {
        console.log("Error:", data);
        setTimeout(function () {
          $(loadingModal).modal("hide");
        }, 500);
        interfaceUtils.alert(
          "Error during Points2Regions, check logs. This plugin only works on a pip installation of TissUUmaps, with the extra packages: pandas, sklearn, skimage",
        );
      },
    });
  } else {
    var content = `
from points2regions._points2regions import Points2Regions as p2r
from js import dataUtils
from js import Points2Regions
from pyodide.ffi import to_js
import numpy as np
print("asd")
Points2Regions.setMessage("HEJHEJHEJ")
#Points2Regions.setMessage("Run failed.")
data = dict(dataUtils.data.object_entries())
data_obj = data[Points2Regions.get("_dataset")]
processeddata = dict(data_obj._processeddata.object_entries())
x_field = data_obj._X
y_field = data_obj._Y

x = np.asarray(processeddata[x_field].to_py(), dtype="float32")
y = np.asarray(processeddata[y_field].to_py(), dtype="float32")
if (data_obj._collectionItem_col in processeddata.keys()):
    lib_id = np.asarray(processeddata[data_obj._collectionItem_col].to_py())
else:
    lib_id = None
xy = np.vstack((x,y)).T

labels = np.asarray(processeddata[Points2Regions.get("_clusterKey")].to_py())
from os.path import join

Points2Regions.setMessage("Run failed.")
pixel_smoothing = float(Points2Regions.get("_pixel_smoothing"))
pixel_size = float(Points2Regions.get("_pixel_size"))
nclusters = int(Points2Regions.get("_nclusters"))
min_pts_per_pixel = float(Points2Regions.get("_min_pts_per_pixel"))
seed = int(Points2Regions.get("_seed"))
region_name = Points2Regions.get("_region_name")

if (Points2Regions.get("_format")== "GeoJSON polygons"):
    compute_regions = True
else:
    compute_regions = False


print(pixel_size, pixel_smoothing, min_pts_per_pixel)
mdl = None
try:
  mdl = p2r(xy, labels, pixel_size, pixel_smoothing, min_pts_per_pixel, lib_id)
except Exception as e:
  Points2Regions.setMessage("Run failed. " + str(e))
if mdl is not None:
  c = mdl.fit_predict(num_clusters=nclusters, output="marker")

  if (Points2Regions.get("_format")== "GeoJSON polygons"):
      import json
      print("hej")
      r = mdl.predict(output='geojson')
      print(r)
      Points2Regions.loadRegions(json.dumps(r))
  else:
      Points2Regions.loadClusters(to_js(c))
  Points2Regions.setMessage("")

`;
    if (Points2Regions.get("_dataset") === "") {
      Points2Regions.set("_dataset", Object.keys(dataUtils.data)[0]);
    }
    if (Points2Regions.get("_clusterKey") === undefined) {
      Points2Regions.set(
        "_clusterKey",
        dataUtils.data[Points2Regions.get("_dataset")]._gb_col,
      );
    }
    Points2Regions.setMessage("Running Python code...");
    setTimeout(() => {
      Points2Regions.executePythonString(content);
    }, 10);
  }
};

Points2Regions.inputTrigger = function (parameterName) {
  if (parameterName == "_refresh") {
    interfaceUtils.cleanSelect(Points2Regions.getInputID("_dataset"));
    interfaceUtils.cleanSelect(Points2Regions.getInputID("_clusterKey"));

    var datasets = Object.keys(dataUtils.data).map(function (e, i) {
      return {
        value: e,
        innerHTML: document.getElementById(e + "_tab-name").value,
      };
    });
    interfaceUtils.addObjectsToSelect(
      Points2Regions.getInputID("_dataset"),
      datasets,
    );
    var event = new Event("change");
    interfaceUtils
      .getElementById(Points2Regions.getInputID("_dataset"))
      .dispatchEvent(event);
  } else if (parameterName == "_dataset") {
    if (!dataUtils.data[Points2Regions.get("_dataset")]) return;
    interfaceUtils.cleanSelect(Points2Regions.getInputID("_clusterKey"));
    interfaceUtils.addElementsToSelect(
      Points2Regions.getInputID("_clusterKey"),
      dataUtils.data[Points2Regions.get("_dataset")]._csv_header,
    );
    Points2Regions.set(
      "_clusterKey",
      dataUtils.data[Points2Regions.get("_dataset")]._gb_col,
    );

    if (dataUtils.data[Points2Regions.get("_dataset")]._filetype == "h5") {
      select311 = interfaceUtils._mGenUIFuncs.intputToH5(
        Points2Regions.get("_dataset"),
        interfaceUtils.getElementById(Points2Regions.getInputID("_clusterKey")),
      );
      Points2Regions.set(
        "_clusterKey",
        dataUtils.data[Points2Regions.get("_dataset")]._gb_col,
      );
      select311.addEventListener("change", (event) => {
        Points2Regions.set("_clusterKey", select311.value);
      });
    }
  } else if (parameterName == "_selectStride") {
    Points2Regions.selectStride();
  } else if (parameterName == "_run") {
    Points2Regions.run();
  } else if (parameterName == "_downloadCSV") {
    Points2Regions.downloadCSV();
  } else if (parameterName == "_server") {
    if (!Points2Regions.get("_server")) {
      Points2Regions.initPython();
    }
  }
};

Points2Regions.selectStride = function (parameterName) {
  var startSelection = null;
  var pressHandler = function (event) {
    console.log("Pressed!");
    var OSDviewer = tmapp["ISS_viewer"];
    startSelection = OSDviewer.viewport.pointFromPixel(event.position);
  };
  var moveHandler = function (event) {
    if (startSelection == null) return;
    let OSDviewer = tmapp["ISS_viewer"];

    let normCoords = OSDviewer.viewport.pointFromPixel(event.position);
    let tiledImage = OSDviewer.world.getItemAt(0);
    let rectangle = tiledImage.viewportToImageRectangle(
      new OpenSeadragon.Rect(
        startSelection.x,
        startSelection.y,
        normCoords.x - startSelection.x,
        normCoords.y - startSelection.y,
      ),
    );
    let canvas =
      overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
    let regionobj = d3
      .select(canvas)
      .append("g")
      .attr("class", "_stride_region");
    let elements = document.getElementsByClassName("stride_region");
    for (let element of elements) element.parentNode.removeChild(element);
    elements = document.getElementsByClassName("stride_region");
    for (let element of elements) element.parentNode.removeChild(element);

    let width = Math.max(
      normCoords.x - startSelection.x,
      normCoords.y - startSelection.y,
    );
    console.log(width, normCoords.x, normCoords.y);
    regionobj
      .append("rect")
      .attr("width", width)
      .attr("height", width)
      .attr("x", startSelection.x)
      .attr("y", startSelection.y)
      .attr("fill", "#ADD8E6")
      .attr("stroke", "#ADD8E6")
      .attr("fill-opacity", 0.3)
      .attr("stroke-opacity", 0.7)
      .attr("stroke-width", 0.002 / tmapp["ISS_viewer"].viewport.getZoom())
      .attr(
        "stroke-dasharray",
        0.004 / tmapp["ISS_viewer"].viewport.getZoom() +
          "," +
          0.004 / tmapp["ISS_viewer"].viewport.getZoom(),
      )
      .attr("class", "stride_region");
    regionobj
      .append("rect")
      .attr("width", width * Points2Regions.get("_pixel_smoothing"))
      .attr("height", width * Points2Regions.get("_pixel_smoothing"))
      .attr(
        "x",
        startSelection.x +
          width / 2 -
          (width * Points2Regions.get("_pixel_smoothing")) / 2,
      )
      .attr(
        "y",
        startSelection.y +
          width / 2 -
          (width * Points2Regions.get("_pixel_smoothing")) / 2,
      )
      .attr("fill", "#ADD8E6")
      .attr("stroke", "#ADD8E6")
      .attr("fill-opacity", 0.3)
      .attr("stroke-opacity", 0.7)
      .attr("stroke-width", 0.002 / tmapp["ISS_viewer"].viewport.getZoom())
      .attr(
        "stroke-dasharray",
        0.004 / tmapp["ISS_viewer"].viewport.getZoom() +
          "," +
          0.004 / tmapp["ISS_viewer"].viewport.getZoom(),
      )
      .attr("class", "stride_region");
    pixel_smoothing = Points2Regions.get("_pixel_smoothing");
    Points2Regions.set("_pixel_size", Math.abs(rectangle.width).toFixed(2));
    return;
  };
  var dragHandler = function (event) {
    event.preventDefaultAction = true;
  };
  var releaseHandler = function (event) {
    console.log("Released!", pressHandler, releaseHandler, dragHandler);

    startSelection = null;
    tmapp["ISS_viewer"].removeHandler("canvas-press", pressHandler);
    tmapp["ISS_viewer"].removeHandler("canvas-release", releaseHandler);
    tmapp["ISS_viewer"].removeHandler("canvas-drag", dragHandler);
    let elements = document.getElementsByClassName("stride_region");
    for (var element of elements) element.parentNode.removeChild(element);
    elements = document.getElementsByClassName("stride_region");
    for (var element of elements) element.parentNode.removeChild(element);
  };
  tmapp["ISS_viewer"].addHandler("canvas-press", pressHandler);
  tmapp["ISS_viewer"].addHandler("canvas-release", releaseHandler);
  tmapp["ISS_viewer"].addHandler("canvas-drag", dragHandler);
  new OpenSeadragon.MouseTracker({
    element: tmapp["ISS_viewer"].canvas,
    moveHandler: (event) => moveHandler(event),
  }).setTracking(true);
};

Points2Regions.loadClusters = function (data) {
  let data_obj = dataUtils.data[Points2Regions.get("_dataset")];
  data_obj._processeddata["Points2Regions"] = data;
  data_obj._gb_col = "Points2Regions";
  data_obj._processeddata.columns.push("Points2Regions");
  interfaceUtils.addElementsToSelect(
    Points2Regions.get("_dataset") + "_gb-col-value",
    ["Points2Regions"],
  );
  document.getElementById(
    Points2Regions.get("_dataset") + "_gb-col-value",
  ).value = "Points2Regions";
  let colors = {
    "-1": "#000000",
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#279e68",
    3: "#d62728",
    4: "#aa40fc",
    5: "#8c564b",
    6: "#e377c2",
    7: "#b5bd61",
    8: "#17becf",
    9: "#aec7e8",
    10: "#ffbb78",
    11: "#98df8a",
    12: "#ff9896",
    13: "#c5b0d5",
    14: "#c49c94",
    15: "#f7b6d2",
    16: "#dbdb8d",
    17: "#9edae5",
    18: "#ad494a",
    19: "#8c6d31",
  };
  document
    .getElementById(Points2Regions.get("_dataset") + "_cb-bygroup-dict")
    .click();
  document.getElementById(
    Points2Regions.get("_dataset") + "_cb-bygroup-dict-val",
  ).value = JSON.stringify(colors);
  dataUtils._quadtreesLastInputs = null;
  glUtils._markerInputsCached[Points2Regions.get("_dataset")] = null;
  dataUtils.updateViewOptions(Points2Regions.get("_dataset"));
};

Points2Regions.loadRegions = function (data) {
  // Change stroke width for computation reasons:
  regionUtils._polygonStrokeWidth = 0.0005;
  groupRegions = Object.values(regionUtils._regions)
    .filter((x) => x.regionClass == Points2Regions._region_name)
    .forEach(function (region) {
      regionUtils.deleteRegion(region.id);
    });

  regionsobj = JSON.parse(data);
  console.log(regionsobj);
  regionUtils.JSONValToRegions(regionsobj);
  $("#title-tab-regions").tab("show");
  $(
    document.getElementById("regionClass-" + Points2Regions._region_name),
  ).collapse("show");
  $("#" + Points2Regions._region_name + "_group_fill_ta").click();
};

/*
 * Only helper functions below
 *
 */
Points2Regions.executePythonString = function (text) {
  // prepare objects exposed to Python

  // pyscript
  let div = document.createElement("div");
  let html = `
        <py-script>
  ${text}
        </py-script>
        `;
  div.innerHTML = html;

  // if we did this before, remove the script from the body
  if (Points2Regions.myPyScript) {
    Points2Regions.myPyScript.remove();
  }
  // now remember the new script
  Points2Regions.myPyScript = div.firstElementChild;
  try {
    document.body.appendChild(Points2Regions.myPyScript);
    // execute the code / evaluate the expression
    //Points2Regions.myPyScript.evaluate();
  } catch (error) {
    console.error("Python error:");
    console.error(error);
  }
};
Points2Regions.executePythonFile = function (url) {
  // prepare objects exposed to Python

  // pyscript
  let div = document.createElement("div");
  let html = `
        <py-script src="${url}"></py-script>
        `;
  div.innerHTML = html;

  // if we did this before, remove the script from the body
  if (Points2Regions.myPyScriptFilde) {
    Points2Regions.myPyScriptFilde.remove();
  }
  // now remember the new script
  Points2Regions.myPyScriptFilde = div.firstElementChild;
  try {
    document.body.appendChild(Points2Regions.myPyScriptFilde);
    // execute the code / evaluate the expression
    //Points2Regions.myPyScriptFilde.evaluate();
  } catch (error) {
    console.error("Python error:");
    console.error(error);
  }
};

Points2Regions.initPython = function () {
  if (!document.getElementById("pyScript")) {
    Points2Regions.setMessage("Loading Python interpreter...");
    var link = document.createElement("link");
    link.src = "https://pyscript.net/releases/2023.11.1/core.css";
    link.id = "pyScript";
    link.rel = "stylesheet";
    document.head.appendChild(link);

    var script = document.createElement("script");
    script.src = "https://pyscript.net/releases/2023.11.1/core.js";
    script.type = "module";
    script.defer = true;
    document.head.appendChild(script);

    var pyconfig = document.createElement("py-config");
    pyconfig.innerHTML = `
    packages=['scikit-learn','scikit-image','typing-extensions']
    [[fetch]]
    from = "https://raw.githubusercontent.com/wahlby-lab/Points2Regions/main/"
    files = ["points2regions/__init__.py", "points2regions/geojson.py", "points2regions/utils.py", "points2regions/_points2regions.py"]`;
    document.head.appendChild(pyconfig);

    Points2Regions.executePythonString(`
        from js import Points2Regions
        Points2Regions.pythonLoaded()
      `);
  }
};

Points2Regions.pythonLoaded = function () {
  Points2Regions.setMessage("");
};

Points2Regions.setMessage = function (text) {
  if (!document.getElementById("Points2Regions_message")) {
    var label_row = HTMLElementUtils.createRow({});
    var label_col = HTMLElementUtils.createColumn({ width: 12 });
    var label = HTMLElementUtils.createElement({
      kind: "p",
      id: "Points2Regions_message",
    });
    label.setAttribute("class", "badge bg-warning text-dark");
    console.log(text);
    label_row.appendChild(label_col);
    label_col.appendChild(label);
    Points2Regions.container.appendChild(label_row);
  }
  document.getElementById("Points2Regions_message").innerText = text;
};

Points2Regions.downloadCSV = function () {
  var csvRows = [];
  let alldata = dataUtils.data[Points2Regions.get("_dataset")]._processeddata;
  let headers = Object.keys(alldata);
  headers.splice(headers.indexOf("columns"), 1);
  csvRows.push(headers.join(","));
  let zip = (...rows) => [...rows[0]].map((_, c) => rows.map((row) => row[c]));
  let rows = zip(...headers.map((h) => alldata[h]));
  const escape = (text) =>
    text.replace(/\\/g, "\\\\").replace(/\n/g, "\\n").replace(/,/g, "\\,");

  //let escaped_array = rows.map(fields => fields.map(escape))
  let csv =
    headers.toString() +
    "\n" +
    rows.map((fields) => fields.join(",")).join("\n");

  regionUtils.downloadPointsInRegionsCSV(csv);
};

Points2Regions._api = function (endpoint, data, success, error) {
  $.ajax({
    // Post select to url.
    type: "post",
    url: "/plugins/Points2Regions" + "/" + endpoint,
    contentType: "application/json; charset=utf-8",
    data: JSON.stringify(data),
    success: function (data) {
      success(data);
    },
    complete: function (data) {
      // do something, not critical.
    },
    error: error
      ? error
      : function (data) {
          interfaceUtils.alert(
            data.responseText.replace("\n", "<br/>"),
            "Error on the plugin's server response:",
          );
        },
  });
};
