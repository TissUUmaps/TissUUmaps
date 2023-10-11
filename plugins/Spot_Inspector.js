/**
 * @file Spot_Inspector.js
 * @author Christophe Avenel, Axel Andersson
 */

/**
 * @namespace Spot_Inspector
 * @classdesc The root namespace for Spot_Inspector.
 */
var Spot_Inspector;
Spot_Inspector = {
  name: "Spot Inspector Plugin",
  _bboxSize: 11,
  _figureSize: 7,
  _layer_format: null,
  _only_picked: false,
  _show_trace: true,
  _marker_row: "rounds",
  _marker_col: "channels",
  _cmap: "None",
  _use_raw: false,
};

/**
 * This method is called when the document is loaded. The tmapp object is built as an "app" and init is its main function.
 * Creates the OpenSeadragon (OSD) viewer and adds the handlers for interaction.
 * To know which data one is referring to, there are Object Prefixes (op). For In situ sequencing projects it can be "ISS" for
 * Cell Profiler data it can be "CP".
 * If there are images to be displayed on top of the main image, they are stored in the layers object and, if there are layers
 * it will create the buttons to display them in the settings panel.
 * The SVG overlays for the viewer are also initialized here
 * @summary After setting up the tmapp object, initialize it*/
Spot_Inspector.init = function (container) {
  var script = document.createElement("script");
  script.src =
    "https://raw.githubusercontent.com/jonTrent/PatienceDiff/dev/PatienceDiff.js";
  document.head.appendChild(script);

  Spot_Inspector.updateLayerFormat(false);
  row0 = HTMLElementUtils.createElement({
    kind: "h6",
    extraAttributes: { class: "" },
  });
  row0.innerText = "IMAGE LAYERS";
  row0.style.borderBottom = "1px solid #aaa";
  row0.style.padding = "3px";
  row0.style.marginTop = "8px";

  row01 = HTMLElementUtils.createElement({
    kind: "p",
    extraAttributes: { class: "" },
  });
  row01.innerHTML =
    "<i>You can use existing layers or load a group of images from the project folder.</i>";

  row1 = HTMLElementUtils.createRow({});
  col11 = HTMLElementUtils.createColumn({ width: 12 });
  label112 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: {
      class: "form-check-label",
      for: "Spot_Inspector_bboxSize",
    },
  });
  label112.innerHTML = "Box size:";
  var input112 = HTMLElementUtils.createElement({
    kind: "input",
    id: "Spot_Inspector_bboxSize",
    extraAttributes: {
      class: "form-text-input form-control",
      type: "number",
      value: Spot_Inspector._bboxSize,
    },
  });

  input112.addEventListener("change", (event) => {
    Spot_Inspector._bboxSize = parseInt(input112.value);
  });

  row7 = HTMLElementUtils.createRow({});
  col71 = HTMLElementUtils.createColumn({ width: 12 });
  label712 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: {
      class: "form-check-label",
      for: "Spot_Inspector_bboxSize",
    },
  });
  label712.innerHTML = "Figure size:";
  var input712 = HTMLElementUtils.createElement({
    kind: "input",
    id: "Spot_Inspector_figureSize",
    extraAttributes: {
      class: "form-text-input form-control",
      type: "number",
      value: Spot_Inspector._figureSize,
    },
  });

  input712.addEventListener("change", (event) => {
    Spot_Inspector._figureSize = parseInt(input712.value);
  });

  row2 = HTMLElementUtils.createRow({});
  col21 = HTMLElementUtils.createColumn({ width: 12 });
  label212 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: {
      class: "form-check-label",
      for: "Spot_Inspector_layer_format",
    },
  });
  label212.innerHTML =
    "Layer name format - use <b>{row}</b> and <b>{col}</b> to define dimensions:";
  var input212 = HTMLElementUtils.createElement({
    kind: "input",
    id: "Spot_Inspector_layer_format",
    extraAttributes: {
      class: "form-text-input form-control",
      type: "text",
      value: Spot_Inspector._layer_format,
      placeholder: "Round{0}_{1}",
    },
  });

  input212.addEventListener("change", (event) => {
    Spot_Inspector._layer_format = input212.value;
  });

  row3b = HTMLElementUtils.createRow({});
  col3b1 = HTMLElementUtils.createColumn({ width: 12 });
  var input3b11 = HTMLElementUtils.createElement({
    kind: "input",
    id: "Spot_Inspector_use_raw",
    extraAttributes: { class: "form-check-input", type: "checkbox" },
  });
  label3b11 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "Spot_Inspector_use_raw" },
  });
  label3b11.innerHTML = "&nbsp;Use raw images (slow)";

  input3b11.addEventListener("change", (event) => {
    Spot_Inspector._use_raw = input3b11.checked;
  });

  row6 = HTMLElementUtils.createRow({});
  col61 = HTMLElementUtils.createColumn({ width: 12 });
  select611 = HTMLElementUtils.createElement({
    kind: "select",
    id: "Spot_Inspector_colormap",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label612 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "Spot_Inspector_colormap" },
  });
  label612.innerText = "Select colormap";

  select611.addEventListener("change", (event) => {
    Spot_Inspector._cmap = select611.value;
  });

  row5 = HTMLElementUtils.createRow({});
  col51 = HTMLElementUtils.createColumn({ width: 12 });
  button511 = HTMLElementUtils.createButton({
    extraAttributes: { class: "btn btn-secondary btn-sm" },
  });
  button511.innerHTML = "Import folder of images into layers <i>(optional)</i>";

  button511.addEventListener("click", (event) => {
    interfaceUtils
      .prompt(
        "<i>This will replace all layers of the current project.</i><br/>Give the path format of your images, use * for numbers:",
        "Round*_*",
        "Import images into layers",
      )
      .then((pathFormat) => {
        Spot_Inspector.loadImages(pathFormat);
      });
  });

  row8 = HTMLElementUtils.createElement({
    kind: "h6",
    extraAttributes: { class: "" },
  });
  row8.innerText = "MARKERS (Optional)";
  row8.style.borderBottom = "1px solid #aaa";
  row8.style.marginTop = "8px";
  row8.style.padding = "3px";

  row81 = HTMLElementUtils.createElement({
    kind: "p",
    extraAttributes: { class: "" },
  });
  row81.innerHTML =
    '<i>If you want to visualize markers on top of the image, you need to have a csv column for successive {row} values separated by semi-colons, and a csv column for successive {col} values separated by semi-colons (e.g. "1;2;3;4" and "C;T;G;A" for layers in the format Round1_C.tif</i>';

  row9 = HTMLElementUtils.createRow({});
  col91 = HTMLElementUtils.createColumn({ width: 12 });
  select911 = HTMLElementUtils.createElement({
    kind: "select",
    id: "marker_row",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label912 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "marker_row" },
  });
  label912.innerText = "Select {row} column of markers";

  select911.addEventListener("change", (event) => {
    Spot_Inspector._marker_row = select911.value;
  });

  row10 = HTMLElementUtils.createRow({});
  col101 = HTMLElementUtils.createColumn({ width: 12 });
  select1011 = HTMLElementUtils.createElement({
    kind: "select",
    id: "marker_col",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label1012 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "marker_col" },
  });
  label1012.innerText = "Select {col} column of markers";

  select1011.addEventListener("change", (event) => {
    Spot_Inspector._marker_col = select1011.value;
  });

  row4 = HTMLElementUtils.createRow({});
  col41 = HTMLElementUtils.createColumn({ width: 12 });
  var input411 = HTMLElementUtils.createElement({
    kind: "input",
    id: "Spot_Inspector_only_picked",
    extraAttributes: { class: "form-check-input", type: "checkbox" },
  });
  label411 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "Spot_Inspector_only_picked" },
  });
  label411.innerHTML = "&nbsp;Only show central selected marker";

  input411.addEventListener("change", (event) => {
    Spot_Inspector._only_picked = input411.checked;
  });

  row4b = HTMLElementUtils.createRow({});
  col4b1 = HTMLElementUtils.createColumn({ width: 12 });
  var input4b11 = HTMLElementUtils.createElement({
    kind: "input",
    id: "Spot_Inspector_show_trace",
    extraAttributes: {
      class: "form-check-input",
      type: "checkbox",
      checked: Spot_Inspector._show_trace,
    },
  });
  label4b11 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "Spot_Inspector_show_trace" },
  });
  label4b11.innerHTML = "&nbsp;Connect dots with a line";

  input4b11.addEventListener("change", (event) => {
    Spot_Inspector._show_trace = input4b11.checked;
  });

  container.innerHTML = "";
  container.appendChild(row0);
  container.appendChild(row01);
  container.appendChild(row5);
  row5.appendChild(col51);
  col51.appendChild(button511);
  container.appendChild(row3b);
  row3b.appendChild(col3b1);
  col3b1.appendChild(input3b11);
  col3b1.appendChild(label3b11);
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
  container.appendChild(row6);
  row6.appendChild(col61);
  col61.appendChild(label612);
  col61.appendChild(select611);

  container.appendChild(row8);
  container.appendChild(row81);

  container.appendChild(row9);
  row9.appendChild(col91);
  col91.appendChild(label912);
  col91.appendChild(select911);

  container.appendChild(row10);
  row10.appendChild(col101);
  col101.appendChild(label1012);
  col101.appendChild(select1011);

  container.appendChild(row4);
  row4.appendChild(col41);
  col41.appendChild(input411);
  col41.appendChild(label411);
  container.appendChild(row4b);
  row4b.appendChild(col4b1);
  col4b1.appendChild(input4b11);
  col4b1.appendChild(label4b11);

  cmap = [
    "None",
    "Greys_r",
    "Greys",
    "Accent",
    "Accent_r",
    "Blues",
    "Blues_r",
    "BrBG",
    "BrBG_r",
    "BuGn",
    "BuGn_r",
    "BuPu",
    "BuPu_r",
    "CMRmap",
    "CMRmap_r",
    "Dark2",
    "Dark2_r",
    "GnBu",
    "GnBu_r",
    "Greens",
    "Greens_r",
    "OrRd",
    "OrRd_r",
    "Oranges",
    "Oranges_r",
    "PRGn",
    "PRGn_r",
    "Paired",
    "Paired_r",
    "Pastel1",
    "Pastel1_r",
    "Pastel2",
    "Pastel2_r",
    "PiYG",
    "PiYG_r",
    "PuBu",
    "PuBuGn",
    "PuBuGn_r",
    "PuBu_r",
    "PuOr",
    "PuOr_r",
    "PuRd",
    "PuRd_r",
    "Purples",
    "Purples_r",
    "RdBu",
    "RdBu_r",
    "RdGy",
    "RdGy_r",
    "RdPu",
    "RdPu_r",
    "RdYlBu",
    "RdYlBu_r",
    "RdYlGn",
    "RdYlGn_r",
    "Reds",
    "Reds_r",
    "Set1",
    "Set1_r",
    "Set2",
    "Set2_r",
    "Set3",
    "Set3_r",
    "Spectral",
    "Spectral_r",
    "Wistia",
    "Wistia_r",
    "YlGn",
    "YlGnBu",
    "YlGnBu_r",
    "YlGn_r",
    "YlOrBr",
    "YlOrBr_r",
    "YlOrRd",
    "YlOrRd_r",
    "afmhot",
    "afmhot_r",
    "autumn",
    "autumn_r",
    "binary",
    "binary_r",
    "bone",
    "bone_r",
    "brg",
    "brg_r",
    "bwr",
    "bwr_r",
    "cividis",
    "cividis_r",
    "cool",
    "cool_r",
    "coolwarm",
    "coolwarm_r",
    "copper",
    "copper_r",
    "cubehelix",
    "cubehelix_r",
    "flag",
    "flag_r",
    "gist_earth",
    "gist_earth_r",
    "gist_gray",
    "gist_gray_r",
    "gist_heat",
    "gist_heat_r",
    "gist_ncar",
    "gist_ncar_r",
    "gist_rainbow",
    "gist_rainbow_r",
    "gist_stern",
    "gist_stern_r",
    "gist_yarg",
    "gist_yarg_r",
    "gnuplot",
    "gnuplot2",
    "gnuplot2_r",
    "gnuplot_r",
    "gray",
    "gray_r",
    "hot",
    "hot_r",
    "hsv",
    "hsv_r",
    "inferno",
    "inferno_r",
    "jet",
    "jet_r",
    "magma",
    "magma_r",
    "nipy_spectral",
    "nipy_spectral_r",
    "ocean",
    "ocean_r",
    "pink",
    "pink_r",
    "plasma",
    "plasma_r",
    "prism",
    "prism_r",
    "rainbow",
    "rainbow_r",
    "seismic",
    "seismic_r",
    "spring",
    "spring_r",
    "summer",
    "summer_r",
    "tab10",
    "tab10_r",
    "tab20",
    "tab20_r",
    "tab20b",
    "tab20b_r",
    "tab20c",
    "tab20c_r",
    "terrain",
    "terrain_r",
    "turbo",
    "turbo_r",
    "twilight",
    "twilight_r",
    "twilight_shifted",
    "twilight_shifted_r",
    "viridis",
    "viridis_r",
    "winter",
    "winter_r",
  ];
  interfaceUtils.addElementsToSelect("Spot_Inspector_colormap", cmap);
  if (Object.keys(dataUtils.data).length > 0) {
    interfaceUtils.cleanSelect("marker_row");
    interfaceUtils.addElementsToSelect(
      "marker_row",
      Object.values(dataUtils.data)[0]._csv_header,
    );

    interfaceUtils.cleanSelect("marker_col");
    interfaceUtils.addElementsToSelect(
      "marker_col",
      Object.values(dataUtils.data)[0]._csv_header,
    );
    if (
      Object.values(dataUtils.data)[0]._csv_header.indexOf(
        Spot_Inspector._marker_row,
      ) > 0
    ) {
      interfaceUtils.getElementById("marker_row").value =
        Spot_Inspector._marker_row;
    }
    if (
      Object.values(dataUtils.data)[0]._csv_header.indexOf(
        Spot_Inspector._marker_col,
      ) > 0
    ) {
      interfaceUtils.getElementById("marker_col").value =
        Spot_Inspector._marker_col;
    }
  }
  Spot_Inspector.run();
};

Spot_Inspector.loadImages = function (pathFormat) {
  console.log("Import images into layers");
  var op = tmapp["object_prefix"];
  var vname = op + "_viewer";

  subfolder = window.location.pathname.substring(
    0,
    window.location.pathname.lastIndexOf("/"),
  );
  //subfolder = subfolder.substring(0, subfolder.lastIndexOf('/') + 1);
  const queryString = window.location.search;
  const urlParams = new URLSearchParams(queryString);
  const path = urlParams.get("path");
  $.ajax({
    // Post select to url.
    type: "post",
    url: "/plugins/Spot_Inspector/importFolder",
    contentType: "application/json; charset=utf-8",
    data: JSON.stringify({
      path: path,
      pathFormat: pathFormat,
    }),
    success: function (data) {
      if (projectUtils.loadLayers) projectUtils.loadLayers(data);
      else projectUtils.loadProject(data);
      setTimeout(function () {
        Spot_Inspector.updateLayerFormat(false);
        $("#Spot_Inspector_layer_format")[0].value =
          Spot_Inspector._layer_format;
      }, 500);
    },
    complete: function (data) {
      // do something, not critical.
    },
    error: function (data) {
      interfaceUtils.alert(
        data.responseText.replace("\n", "<br/>"),
        "Error on the plugin's server response",
      );
    },
  });
};

Spot_Inspector.run = function () {
  if (window.Spot_Inspector_started) return;
  window.Spot_Inspector_started = true;
  var op = tmapp["object_prefix"];
  var vname = op + "_viewer";

  var click_handler = function (event) {
    if (event.quick) {
      var OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];
      var tiledImage = OSDviewer.world.getItemAt(0);
      var viewportCoords = OSDviewer.viewport.pointFromPixel(event.position);
      var normCoords = tiledImage.viewportToImageCoordinates(viewportCoords);

      var bbox = [
        Math.round(normCoords.x - Spot_Inspector._bboxSize / 2),
        Math.round(normCoords.y - Spot_Inspector._bboxSize / 2),
        Spot_Inspector._bboxSize,
        Spot_Inspector._bboxSize,
      ];
      var markers = Spot_Inspector.getMarkers(bbox);
      if (markers.length > 0) {
        if (!Spot_Inspector._layer_format)
          Spot_Inspector.updateLayerFormat(false);
      }
      img = document.getElementById("ISS_Spot_Inspector_img");
      if (img) img.style.filter = "blur(5px)";
      Spot_Inspector.getMatrix(bbox, tmapp.layers, markers);

      color = "red";
      var boundBoxOverlay = document.getElementById("overlay-Spot_Inspector");
      if (boundBoxOverlay) {
        OSDviewer.removeOverlay(boundBoxOverlay);
      }
      var boundBoxRect = tiledImage.imageToViewportRectangle(
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3],
      );
      boundBoxOverlay = $('<div id="overlay-Spot_Inspector"></div>');
      boundBoxOverlay.css({
        border: "2px solid " + color,
      });
      OSDviewer.addOverlay(boundBoxOverlay.get(0), boundBoxRect);
    } else {
      //if it is not quick then its panning
      // nothing
    }
  };

  //OSD handlers are not registered manually they have to be registered
  //using MouseTracker OSD objects
  if (Spot_Inspector.ISS_mouse_tracker == undefined) {
    /*Spot_Inspector.ISS_mouse_tracker = new OpenSeadragon.MouseTracker({
             //element: this.fixed_svgov.node().parentNode,
             element: tmapp[vname].canvas,
             clickHandler: click_handler
         }).setTracking(true);*/

    Spot_Inspector.ISS_mouse_tracker = tmapp["ISS_viewer"].addHandler(
      "canvas-click",
      (event) => {
        click_handler(event);
      },
    );
  }
};

Spot_Inspector.updateLayerFormat = function (doPrompt) {
  if (doPrompt == undefined) doPrompt = true;
  rounds = [];
  channels = [];

  var difference = patienceDiff(
    tmapp.layers[0].name.split(""),
    tmapp.layers[tmapp.layers.length - 1].name.split(""),
  );
  fieldNames = ["row", "col"];
  var format = difference.lines.reduce(function (a, b) {
    if (b.aIndex == -1 || b.bIndex == -1) {
      if (a.substring(a.length - 1) == "}") return a;
      var formatIndex = (a.match(/\{/g) || []).length;
      return a + "{" + fieldNames[formatIndex] + "}";
    }
    return a + b.line;
  }, "");
  if ((format.match(/\{/g) || []).length > 2) {
    format = "{col}";
  }
  Spot_Inspector._layer_format = format;
};

Spot_Inspector.getMarkers = function (bbox) {
  var xmin = bbox[0]; //OSDviewer.viewport.imageToViewportCoordinates(bbox[0]);
  var ymin = bbox[1]; //OSDviewer.viewport.imageToViewportCoordinates(bbox[1]);
  var xmax = xmin + bbox[2]; //OSDviewer.viewport.imageToViewportCoordinates(bbox[2]);
  var ymax = ymin + bbox[3]; //OSDviewer.viewport.imageToViewportCoordinates(bbox[3]);

  markersInViewportBounds = [];
  for (dataset in dataUtils.data) {
    var allkeys = Object.keys(dataUtils.data[dataset]["_groupgarden"]);
    for (var codeIndex in dataUtils.data[dataset]["_groupgarden"]) {
      var inputs = interfaceUtils._mGenUIFuncs.getGroupInputs(
        dataset,
        codeIndex,
      );
      var hexColor = "color" in inputs ? inputs["color"] : "#ffff00";
      var visible = "visible" in inputs ? inputs["visible"] : true;
      if (visible) {
        var newMarkers = regionUtils.searchTreeForPointsInBbox(
          dataUtils.data[dataset]["_groupgarden"][codeIndex],
          xmin,
          ymin,
          xmax,
          ymax,
          {
            globalCoords: true,
            xselector: dataUtils.data[dataset]["_X"],
            yselector: dataUtils.data[dataset]["_Y"],
            dataset: dataset,
          },
        );
        newMarkers.forEach(function (m) {
          m.color = hexColor;
          (m.global_X_pos = parseFloat(m[dataUtils.data[dataset]["_X"]])),
            (m.global_Y_pos = parseFloat(m[dataUtils.data[dataset]["_Y"]]));
        });
        if (Spot_Inspector._only_picked && newMarkers.length > 0) {
          newMarkers = newMarkers.filter(function (p) {
            return (
              p[""] == glUtils._pickedMarker[1] &&
              dataset == glUtils._pickedMarker[0]
            );
          });
        }
        markersInViewportBounds = markersInViewportBounds.concat(newMarkers);
      }
    }
  }

  return markersInViewportBounds;
};

Spot_Inspector.getMatrix = function (bbox, layers, markers, order) {
  var op = tmapp["object_prefix"];
  var vname = op + "_viewer";
  console.log("Calling ajax getMatrix");
  const queryString = window.location.search;
  const urlParams = new URLSearchParams(queryString);
  const path = urlParams.get("path");
  $.ajax({
    // Post select to url.
    type: "post",
    url: "/plugins/Spot_Inspector/getMatrix",
    contentType: "application/json; charset=utf-8",
    data: JSON.stringify({
      bbox: bbox,
      figureSize: Spot_Inspector._figureSize,
      show_trace: Spot_Inspector._show_trace,
      layers: layers,
      path: path,
      markers: markers,
      marker_row: Spot_Inspector._marker_row,
      marker_col: Spot_Inspector._marker_col,
      layer_format: Spot_Inspector._layer_format,
      cmap: Spot_Inspector._cmap,
      use_raw: Spot_Inspector._use_raw,
    }),
    success: function (data) {
      img = document.getElementById("ISS_Spot_Inspector_img");
      if (!img) {
        var img = document.createElement("img");
        img.id = "ISS_Spot_Inspector_img";
        var elt = document.createElement("div");
        elt.classList.add("viewer-layer");
        //elt.classList.add("px-1");
        //elt.classList.add("mx-1");
        elt.style.maxWidth = "800px";
        elt.style.maxHeight = "800px";
        elt.style.overflow = "auto";

        elt.appendChild(img);
        tmapp[vname].addControl(elt, {
          anchor: OpenSeadragon.ControlAnchor.BOTTOM_RIGHT,
        });
        elt.parentElement.parentElement.style.zIndex = "100";
        elt.style.display = "inherit";

        let eltClose = document.createElement("div");
        eltClose.className = "closeFeature_Space px-1 mx-1 viewer-layer";
        eltClose.id = "closeSpot_Inspector";
        eltClose.style.zIndex = "100";
        eltClose.style.cursor = "pointer";
        eltClose.style.position = "absolute";
        eltClose.style.left = "5px";
        eltClose.style.top = "5px";
        eltClose.innerHTML = "<i class='bi bi-x-lg'></i>";
        eltClose.addEventListener("click", function (event) {
          img.parentElement.remove();
        });
        elt.appendChild(eltClose);
      }
      img.setAttribute("src", "data:image/png;base64," + data);
      img.style.filter = "none";
    },
    complete: function (data) {
      // do something, not critical.
    },
    error: function (data) {
      interfaceUtils.alert(
        data.responseText.replace("\n", "<br/>"),
        "Error on the plugin's server response",
      );
    },
  });
};

///////////////////////////////////////////

/**
 * program: "patienceDiff" algorithm implemented in javascript.
 * author: Jonathan Trent
 * version: 2.0
 *
 * use:  patienceDiff( aLines[], bLines[], diffPlusFlag )
 *
 * where:
 *      aLines[] contains the original text lines.
 *      bLines[] contains the new text lines.
 *      diffPlusFlag if true, returns additional arrays with the subset of lines that were
 *          either deleted or inserted.  These additional arrays are used by patienceDiffPlus.
 *
 * returns an object with the following properties:
 *      lines[] with properties of:
 *          line containing the line of text from aLines or bLines.
 *          aIndex referencing the index in aLines[].
 *          bIndex referencing the index in bLines[].
 *              (Note:  The line is text from either aLines or bLines, with aIndex and bIndex
 *               referencing the original index. If aIndex === -1 then the line is new from bLines,
 *               and if bIndex === -1 then the line is old from aLines.)
 *      lineCountDeleted is the number of lines from aLines[] not appearing in bLines[].
 *      lineCountInserted is the number of lines from bLines[] not appearing in aLines[].
 *      lineCountMoved is 0. (Only set when using patienceDiffPlus.)
 *
 */

function patienceDiff(aLines, bLines, diffPlusFlag) {
  //
  // findUnique finds all unique values in arr[lo..hi], inclusive.  This
  // function is used in preparation for determining the longest common
  // subsequence.  Specifically, it first reduces the array range in question
  // to unique values.
  //
  // Returns an ordered Map, with the arr[i] value as the Map key and the
  // array index i as the Map value.
  //

  function findUnique(arr, lo, hi) {
    const lineMap = new Map();

    for (let i = lo; i <= hi; i++) {
      let line = arr[i];

      if (lineMap.has(line)) {
        lineMap.get(line).count++;
        lineMap.get(line).index = i;
      } else {
        lineMap.set(line, {
          count: 1,
          index: i,
        });
      }
    }

    lineMap.forEach((val, key, map) => {
      if (val.count !== 1) {
        map.delete(key);
      } else {
        map.set(key, val.index);
      }
    });

    return lineMap;
  }

  //
  // uniqueCommon finds all the unique common entries between aArray[aLo..aHi]
  // and bArray[bLo..bHi], inclusive.  This function uses findUnique to pare
  // down the aArray and bArray ranges first, before then walking the comparison
  // between the two arrays.
  //
  // Returns an ordered Map, with the Map key as the common line between aArray
  // and bArray, with the Map value as an object containing the array indexes of
  // the matching unique lines.
  //

  function uniqueCommon(aArray, aLo, aHi, bArray, bLo, bHi) {
    const ma = findUnique(aArray, aLo, aHi);
    const mb = findUnique(bArray, bLo, bHi);

    ma.forEach((val, key, map) => {
      if (mb.has(key)) {
        map.set(key, {
          indexA: val,
          indexB: mb.get(key),
        });
      } else {
        map.delete(key);
      }
    });

    return ma;
  }

  //
  // longestCommonSubsequence takes an ordered Map from the function uniqueCommon
  // and determines the Longest Common Subsequence (LCS).
  //
  // Returns an ordered array of objects containing the array indexes of the
  // matching lines for a LCS.
  //

  function longestCommonSubsequence(abMap) {
    const ja = [];

    // First, walk the list creating the jagged array.

    abMap.forEach((val, key, map) => {
      let i = 0;

      while (ja[i] && ja[i][ja[i].length - 1].indexB < val.indexB) {
        i++;
      }

      if (!ja[i]) {
        ja[i] = [];
      }

      if (0 < i) {
        val.prev = ja[i - 1][ja[i - 1].length - 1];
      }

      ja[i].push(val);
    });

    // Now, pull out the longest common subsequence.

    let lcs = [];

    if (0 < ja.length) {
      let n = ja.length - 1;
      lcs = [ja[n][ja[n].length - 1]];

      while (lcs[lcs.length - 1].prev) {
        lcs.push(lcs[lcs.length - 1].prev);
      }
    }

    return lcs.reverse();
  }

  // "result" is the array used to accumulate the aLines that are deleted, the
  // lines that are shared between aLines and bLines, and the bLines that were
  // inserted.

  const result = [];
  let deleted = 0;
  let inserted = 0;

  // aMove and bMove will contain the lines that don't match, and will be returned
  // for possible searching of lines that moved.

  const aMove = [];
  const aMoveIndex = [];
  const bMove = [];
  const bMoveIndex = [];

  //
  // addToResult simply pushes the latest value onto the "result" array.  This
  // array captures the diff of the line, aIndex, and bIndex from the aLines
  // and bLines array.
  //

  function addToResult(aIndex, bIndex) {
    if (bIndex < 0) {
      aMove.push(aLines[aIndex]);
      aMoveIndex.push(result.length);
      deleted++;
    } else if (aIndex < 0) {
      bMove.push(bLines[bIndex]);
      bMoveIndex.push(result.length);
      inserted++;
    }

    result.push({
      line: 0 <= aIndex ? aLines[aIndex] : bLines[bIndex],
      aIndex: aIndex,
      bIndex: bIndex,
    });
  }

  //
  // addSubMatch handles the lines between a pair of entries in the LCS.  Thus,
  // this function might recursively call recurseLCS to further match the lines
  // between aLines and bLines.
  //

  function addSubMatch(aLo, aHi, bLo, bHi) {
    // Match any lines at the beginning of aLines and bLines.

    while (aLo <= aHi && bLo <= bHi && aLines[aLo] === bLines[bLo]) {
      addToResult(aLo++, bLo++);
    }

    // Match any lines at the end of aLines and bLines, but don't place them
    // in the "result" array just yet, as the lines between these matches at
    // the beginning and the end need to be analyzed first.

    let aHiTemp = aHi;

    while (aLo <= aHi && bLo <= bHi && aLines[aHi] === bLines[bHi]) {
      aHi--;
      bHi--;
    }

    // Now, check to determine with the remaining lines in the subsequence
    // whether there are any unique common lines between aLines and bLines.
    //
    // If not, add the subsequence to the result (all aLines having been
    // deleted, and all bLines having been inserted).
    //
    // If there are unique common lines between aLines and bLines, then let's
    // recursively perform the patience diff on the subsequence.

    const uniqueCommonMap = uniqueCommon(aLines, aLo, aHi, bLines, bLo, bHi);

    if (uniqueCommonMap.size === 0) {
      while (aLo <= aHi) {
        addToResult(aLo++, -1);
      }

      while (bLo <= bHi) {
        addToResult(-1, bLo++);
      }
    } else {
      recurseLCS(aLo, aHi, bLo, bHi, uniqueCommonMap);
    }

    // Finally, let's add the matches at the end to the result.

    while (aHi < aHiTemp) {
      addToResult(++aHi, ++bHi);
    }
  }

  //
  // recurseLCS finds the longest common subsequence (LCS) between the arrays
  // aLines[aLo..aHi] and bLines[bLo..bHi] inclusive.  Then for each subsequence
  // recursively performs another LCS search (via addSubMatch), until there are
  // none found, at which point the subsequence is dumped to the result.
  //

  function recurseLCS(aLo, aHi, bLo, bHi, uniqueCommonMap) {
    const x = longestCommonSubsequence(
      uniqueCommonMap || uniqueCommon(aLines, aLo, aHi, bLines, bLo, bHi),
    );

    if (x.length === 0) {
      addSubMatch(aLo, aHi, bLo, bHi);
    } else {
      if (aLo < x[0].indexA || bLo < x[0].indexB) {
        addSubMatch(aLo, x[0].indexA - 1, bLo, x[0].indexB - 1);
      }

      let i;
      for (i = 0; i < x.length - 1; i++) {
        addSubMatch(
          x[i].indexA,
          x[i + 1].indexA - 1,
          x[i].indexB,
          x[i + 1].indexB - 1,
        );
      }

      if (x[i].indexA <= aHi || x[i].indexB <= bHi) {
        addSubMatch(x[i].indexA, aHi, x[i].indexB, bHi);
      }
    }
  }

  recurseLCS(0, aLines.length - 1, 0, bLines.length - 1);

  if (diffPlusFlag) {
    return {
      lines: result,
      lineCountDeleted: deleted,
      lineCountInserted: inserted,
      lineCountMoved: 0,
      aMove: aMove,
      aMoveIndex: aMoveIndex,
      bMove: bMove,
      bMoveIndex: bMoveIndex,
    };
  }

  return {
    lines: result,
    lineCountDeleted: deleted,
    lineCountInserted: inserted,
    lineCountMoved: 0,
  };
}

/**
 * program: "patienceDiffPlus" algorithm implemented in javascript.
 * author: Jonathan Trent
 * version: 2.0
 *
 * use:  patienceDiffPlus( aLines[], bLines[] )
 *
 * where:
 *      aLines[] contains the original text lines.
 *      bLines[] contains the new text lines.
 *
 * returns an object with the following properties:
 *      lines[] with properties of:
 *          line containing the line of text from aLines or bLines.
 *          aIndex referencing the index in aLine[].
 *          bIndex referencing the index in bLines[].
 *              (Note:  The line is text from either aLines or bLines, with aIndex and bIndex
 *               referencing the original index. If aIndex === -1 then the line is new from bLines,
 *               and if bIndex === -1 then the line is old from aLines.)
 *          moved is true if the line was moved from elsewhere in aLines[] or bLines[].
 *      lineCountDeleted is the number of lines from aLines[] not appearing in bLines[].
 *      lineCountInserted is the number of lines from bLines[] not appearing in aLines[].
 *      lineCountMoved is the number of lines that moved.
 *
 */

function patienceDiffPlus(aLines, bLines) {
  const difference = patienceDiff(aLines, bLines, true);

  let aMoveNext = difference.aMove;
  let aMoveIndexNext = difference.aMoveIndex;
  let bMoveNext = difference.bMove;
  let bMoveIndexNext = difference.bMoveIndex;

  delete difference.aMove;
  delete difference.aMoveIndex;
  delete difference.bMove;
  delete difference.bMoveIndex;

  let lastLineCountMoved;

  do {
    let aMove = aMoveNext;
    let aMoveIndex = aMoveIndexNext;
    let bMove = bMoveNext;
    let bMoveIndex = bMoveIndexNext;

    aMoveNext = [];
    aMoveIndexNext = [];
    bMoveNext = [];
    bMoveIndexNext = [];

    let subDiff = patienceDiff(aMove, bMove);

    lastLineCountMoved = difference.lineCountMoved;

    subDiff.lines.forEach((v, i) => {
      if (0 <= v.aIndex && 0 <= v.bIndex) {
        difference.lines[aMoveIndex[v.aIndex]].moved = true;
        difference.lines[bMoveIndex[v.bIndex]].aIndex = aMoveIndex[v.aIndex];
        difference.lines[bMoveIndex[v.bIndex]].moved = true;
        difference.lineCountInserted--;
        difference.lineCountDeleted--;
        difference.lineCountMoved++;
      } else if (v.bIndex < 0) {
        aMoveNext.push(aMove[v.aIndex]);
        aMoveIndexNext.push(aMoveIndex[v.aIndex]);
      } else {
        bMoveNext.push(bMove[v.bIndex]);
        bMoveIndexNext.push(bMoveIndex[v.bIndex]);
      }
    });
  } while (0 < difference.lineCountMoved - lastLineCountMoved);

  return difference;
}
