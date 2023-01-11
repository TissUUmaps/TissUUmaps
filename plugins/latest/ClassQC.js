/**
 * @file ClassQC.js
 * @author
 */

/**
 * @namespace ClassQC
 * @classdesc The root namespace for ClassQC.
 */
var ClassQC;
ClassQC = {
  name: "ClassQC Plugin",
  _dataset1: null,
  _dataset2: null,
  _dataset1_column: null,
  _dataset2_column: null,
  _bboxSize: 50,
  _figureSize: 5,
  _showMatrix: false,
  _cmap: "Greys_r",
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
ClassQC.init = function (container) {
  var script = document.createElement("script");
  script.src = "https://cdn.plot.ly/plotly-latest.min.js";
  document.head.appendChild(script);
  row1 = HTMLElementUtils.createRow({});
  col11 = HTMLElementUtils.createColumn({ width: 12 });
  button111 = HTMLElementUtils.createButton({
    extraAttributes: { class: "btn btn-primary mx-2" },
  });
  button111.innerText = "Refresh drop-down lists based on loaded markers";

  // Refresh button Refresh drop-down lists based on loaded markers
  button111.addEventListener("click", (event) => {
    interfaceUtils.cleanSelect("ClassQC_dataset1");
    interfaceUtils.cleanSelect("ClassQC_dataset2");
    interfaceUtils.cleanSelect("Column");
    interfaceUtils.cleanSelect("matrix");

    var datasets = Object.keys(dataUtils.data).map(function (e, i) {
      return {
        value: e,
        innerHTML: document.getElementById(e + "_tab-name").value,
      };
    });
    interfaceUtils.addObjectsToSelect("ClassQC_dataset1", datasets);
    interfaceUtils.addObjectsToSelect("ClassQC_dataset2", datasets);
    var event = new Event("change");
    interfaceUtils.getElementById("ClassQC_dataset1").dispatchEvent(event);
    interfaceUtils.getElementById("ClassQC_dataset2").dispatchEvent(event);
  });

  // Select marker dataset 1
  row2 = HTMLElementUtils.createRow({});
  col21 = HTMLElementUtils.createColumn({ width: 12 });
  select211 = HTMLElementUtils.createElement({
    kind: "select",
    id: "ClassQC_dataset1",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label212 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "ClassQC_dataset1" },
  });
  label212.innerText = "Select marker Dataset 1";

  // Marker dataset selector
  select211.addEventListener("change", (event) => {
    ClassQC._dataset1 = select211.value;
    if (!dataUtils.data[ClassQC._dataset1]) return;

    interfaceUtils.cleanSelect("ClassQC_dataset1_column");
    interfaceUtils.addElementsToSelect(
      "ClassQC_dataset1_column",
      dataUtils.data[ClassQC._dataset1]._csv_header
    );
  });
  // END Select marker dataset 1

  // Select column dataset 1
  row3 = HTMLElementUtils.createRow({});
  col31 = HTMLElementUtils.createColumn({ width: 12 });
  select311 = HTMLElementUtils.createElement({
    kind: "select",
    id: "ClassQC_dataset1_column",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label312 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "ClassQC_dataset1_column" },
  });
  label312.innerText = "Select column of Dataset 1 - Circles";

  select311.addEventListener("change", (event) => {
    ClassQC._dataset1_column = select311.value;
  });
  // END Select dataset 1

  // Select dataset 2
  row4 = HTMLElementUtils.createRow({});
  col41 = HTMLElementUtils.createColumn({ width: 12 });
  select411 = HTMLElementUtils.createElement({
    kind: "select",
    id: "ClassQC_dataset2",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label412 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "ClassQC_dataset2" },
  });
  label412.innerText = "Select marker Dataset 2";

  // Marker dataset selector
  select411.addEventListener("change", (event) => {
    ClassQC._dataset2 = select411.value;
    if (!dataUtils.data[ClassQC._dataset2]) return;

    interfaceUtils.cleanSelect("ClassQC_dataset2_column");
    interfaceUtils.addElementsToSelect(
      "ClassQC_dataset2_column",
      dataUtils.data[ClassQC._dataset2]._csv_header
    );
  });
  // END Select dataset 2

  // Select column dataset 2
  row5 = HTMLElementUtils.createRow({});
  col51 = HTMLElementUtils.createColumn({ width: 12 });
  select511 = HTMLElementUtils.createElement({
    kind: "select",
    id: "ClassQC_dataset2_column",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label512 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "ClassQC_dataset2_column" },
  });
  label512.innerText = "Select column of Dataset 2  - Stars";

  select511.addEventListener("change", (event) => {
    ClassQC._dataset2_column = select511.value;
  });
  // END Select column dataset 2

  row6 = HTMLElementUtils.createRow({});
  col61 = HTMLElementUtils.createColumn({ width: 12 });
  var input611 = HTMLElementUtils.createElement({
    kind: "input",
    id: "ClassQC_show_confusion",
    extraAttributes: {
      class: "form-check-input",
      type: "checkbox",
    },
  });
  label611 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "ClassQC_show_confusion" },
  });
  label611.innerHTML =
    "&nbsp;Show confusion matrix of two selected classifications";

  // Marker dataset selector
  input611.addEventListener("change", (event) => {
    ClassQC._showMatrix = input611.checked;
  });

  row7 = HTMLElementUtils.createRow({});
  col71 = HTMLElementUtils.createColumn({ width: 12 });
  label712 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { class: "form-check-label", for: "ClassQC_bboxSize" },
  });
  label712.innerHTML = "Box size:";
  var input712 = HTMLElementUtils.createElement({
    kind: "input",
    id: "ClassQC_bboxSize",
    extraAttributes: {
      class: "form-text-input form-control",
      type: "number",
      value: ClassQC._bboxSize,
    },
  });

  input712.addEventListener("change", (event) => {
    console.log("ClassQC._bboxSize", input712.value, parseInt(input712.value));
    ClassQC._bboxSize = parseInt(input712.value);
  });

  row8 = HTMLElementUtils.createRow({});
  col81 = HTMLElementUtils.createColumn({ width: 12 });
  label812 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { class: "form-check-label", for: "ClassQC_bboxSize" },
  });
  label812.innerHTML = "Figure size:";
  var input812 = HTMLElementUtils.createElement({
    kind: "input",
    id: "ClassQC_figureSize",
    extraAttributes: {
      class: "form-text-input form-control",
      type: "number",
      value: ClassQC._figureSize,
    },
  });

  input812.addEventListener("change", (event) => {
    ClassQC._figureSize = parseInt(input812.value);
  });

  row10 = HTMLElementUtils.createRow({});
  col101 = HTMLElementUtils.createColumn({ width: 12 });
  select1011 = HTMLElementUtils.createElement({
    kind: "select",
    id: "ClassQC_colormap",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label1012 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "ClassQC_colormap" },
  });
  label1012.innerText = "Select colormap";

  select1011.addEventListener("change", (event) => {
    ClassQC._cmap = select1011.value;
  });

  row11 = HTMLElementUtils.createRow({});
  col111 = HTMLElementUtils.createColumn({ width: 12 });
  button1111 = HTMLElementUtils.createButton({
    extraAttributes: { class: "btn btn-primary mx-2" },
  });
  button1111.innerText = "Display";

  button1111.addEventListener("click", (event) => {
    ClassQC.display();
  });

  row12 = HTMLElementUtils.createRow({});
  col121 = HTMLElementUtils.createElement({
    kind: "div",
    id: "ClassQC_matrix",
  });
  col131 = HTMLElementUtils.createElement({
    kind: "div",
    id: "ClassQC_legend",
  });

  container.innerHTML = "";
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
  col51.appendChild(label512);
  col51.appendChild(select511);

  container.appendChild(row6);
  row6.appendChild(col61);
  col61.appendChild(input611);
  col61.appendChild(label611);

  container.appendChild(row7);
  row7.appendChild(col71);
  col71.appendChild(label712);
  col71.appendChild(input712);

  container.appendChild(row8);
  row8.appendChild(col81);
  col81.appendChild(label812);
  col81.appendChild(input812);

  container.appendChild(row10);
  row10.appendChild(col101);
  col101.appendChild(label1012);
  col101.appendChild(select1011);

  container.appendChild(row11);
  row11.appendChild(col111);
  col111.appendChild(button1111);

  container.appendChild(row12);
  row12.appendChild(col121);
  row12.appendChild(col131);

  cmap = [
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
    "sxg",
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
  interfaceUtils.addElementsToSelect("ClassQC_colormap", cmap);

  var event = new Event("click");
  button111.dispatchEvent(event);
  ClassQC.run();
};

ClassQC.display = function (pathFormat) {
  console.log("Display");
  // Change shape
  $("#" + ClassQC._dataset1 + "_shape-fixed-value")[0].value = "disc";
  $("#" + ClassQC._dataset1 + "_shape-fixed")[0].checked = false;
  $("#" + ClassQC._dataset1 + "_shape-fixed")[0].click();
  dataUtils.updateViewOptions(ClassQC._dataset1);
  $("#" + ClassQC._dataset2 + "_shape-fixed-value")[0].value = "star";
  $("#" + ClassQC._dataset2 + "_shape-fixed")[0].checked = false;
  $("#" + ClassQC._dataset2 + "_shape-fixed")[0].click();
  dataUtils.updateViewOptions(ClassQC._dataset2);

  // change opacity and scale:
  dataUtils.data[ClassQC._dataset1]._opacity = 1;
  dataUtils.data[ClassQC._dataset2]._scale_factor = 1;
  dataUtils.data[ClassQC._dataset1]._opacity = 0.4;
  dataUtils.data[ClassQC._dataset2]._scale_factor = 0.6;

  glUtils.loadMarkers(ClassQC._dataset1);
  glUtils.loadMarkers(ClassQC._dataset2);
  glUtils.draw();

  // add condition if checkbox is checked
  if (ClassQC._showMatrix) {
    // Add matrix to div #ClassQC_matrix

    // Get types from both datasets:
    var typeSet1 = Object.keys(
      dataUtils.data[ClassQC._dataset1]._groupgarden
    ).sort();
    var typeSet2 = Object.keys(
      dataUtils.data[ClassQC._dataset2]._groupgarden
    ).sort();
    // Make union of the two type sets into one:
    var typeSet = [...new Set([...typeSet1, ...typeSet2])];

    var types_dataset1 =
      dataUtils.data[ClassQC._dataset1]._processeddata[
        ClassQC._dataset1_column
      ];
    var types_dataset2 =
      dataUtils.data[ClassQC._dataset2]._processeddata[
        ClassQC._dataset2_column
      ];
    dataUtils.data[ClassQC._dataset1]._processeddata["ClassQC_confusion"] =
      Array(types_dataset1.length);

    let _matrix = new Array(typeSet.length); // create an empty array of length n
    for (var i = 0; i < typeSet.length; i++) {
      _matrix[i] = new Array(typeSet.length); // make each element an array
      _matrix[i].fill(0);
    }
    for (var typeIndex in types_dataset1) {
      matrix_index1 = typeSet.indexOf(types_dataset1[typeIndex]);
      matrix_index2 = typeSet.indexOf(types_dataset2[typeIndex]);
      dataUtils.data[ClassQC._dataset1]._processeddata["ClassQC_confusion"][
        typeIndex
      ] = types_dataset1[typeIndex] + "_" + types_dataset2[typeIndex];
      _matrix[typeSet.length - matrix_index1 - 1][matrix_index2] += 1;
    }
    ClassQC.displayConfusion(typeSet, _matrix);
  }
};

ClassQC.displayConfusion = function (_matrix_header, _matrix) {
  var data = [
    {
      z: _matrix,
      x: _matrix_header,
      y: _matrix_header.slice().reverse(),
      type: "heatmap",
      hoverongaps: false,
      colorscale: "Hot",
    },
  ];

  var layout = {
    autosize: true,
    automargin: true,
    showlegend: false,
    yaxis: {
      side: "top",
      tickmode: "array",
      tickvals: _matrix_header,
      ticktext: _matrix_header.map(function (type) {
        try {
          typecolor = document.getElementById(
            ClassQC._dataset1 + "_" + type + "_color"
          ).value;
        } catch (error) {
          typecolor = document.getElementById(
            ClassQC._dataset2 + "_" + type + "_color"
          ).value;
        }
        return (
          "<span style='font-weight:bold;color:" + typecolor + "'>███</span>"
        );
      }),
      ticks: "",
      tickangle: 90,
      title: {
        text: "Dataset 1",
        font: {
          size: 18,
          color: "black",
        },
      },
    },
    xaxis: {
      //   "scaleanchor":"x",
      tickvals: _matrix_header.slice().reverse(),
      ticktext: _matrix_header
        .slice()
        .reverse()
        .map(function (type) {
          try {
            typecolor = document.getElementById(
              ClassQC._dataset2 + "_" + type + "_color"
            ).value;
          } catch (error) {
            typecolor = document.getElementById(
              ClassQC._dataset1 + "_" + type + "_color"
            ).value;
          }
          return (
            "<span style='font-weight:bold;color:" + typecolor + "'>███</span>"
          );
        }),
      ticks: "",
      tickangle: 0,
      title: {
        text: "Dataset 2",
        font: {
          size: 18,
          color: "black",
        },
      },
      ticklabelposition: "top",
      side: "top",
    },
    annotations: [],
    title: null,
  };
  var maxRowVal = _matrix.map(function (row) {
    return Math.max.apply(Math, row);
  });
  var maxVal = Math.max.apply(null, maxRowVal);

  for (var i = 0; i < _matrix_header.length; i++) {
    for (var j = 0; j < _matrix_header.length; j++) {
      var currentValue = _matrix[_matrix_header.length - i - 1][j];
      if (currentValue > maxVal / 2) {
        var textColor = "black";
      } else {
        var textColor = "white";
      }
      var result = {
        xref: "x1",
        yref: "y1",
        x: _matrix_header[j],
        y: _matrix_header[i],
        text: currentValue,
        font: {
          family: "Arial",
          size: 12,
          color: "rgb(50, 171, 96)",
        },
        showarrow: false,
        font: {
          color: textColor,
        },
      };
      layout.annotations.push(result);
    }
  }
  Plotly.newPlot(document.getElementById("ClassQC_matrix"), data, layout, {
    responsive: true,
    displayModeBar: false,
  });

  legend = "";
  for (type of _matrix_header) {
    try {
      typecolor = document.getElementById(
        ClassQC._dataset1 + "_" + type + "_color"
      ).value;
    } catch (error) {
      typecolor = document.getElementById(
        ClassQC._dataset2 + "_" + type + "_color"
      ).value;
    }
    legend +=
      "<div style='display:inline-block;margin-right:10px;'><span style='width:15px;color:" +
      typecolor +
      "'>█</span><span style='min-width:150px;margin: 0px 5px;'>" +
      type +
      "</span></div>";
  }
  document.getElementById("ClassQC_legend").innerHTML = legend;

  document.getElementById("ClassQC_matrix").on("plotly_click", function (data) {
    console.log(data.points[0].x, data.points[0].y);
    var clicked_x = data.points[0].x;
    var clicked_y = data.points[0].y;
    data_column = clicked_x + "_" + clicked_y;
    var opacityPropertyName = "ClassQC_opacity";

    // DATASET 1
    var markerData = dataUtils.data[ClassQC._dataset1]["_processeddata"];

    dataUtils.data[ClassQC._dataset1]["_opacity_col"] = opacityPropertyName;
    markerData[opacityPropertyName] = new Float64Array(
      markerData[dataUtils.data[ClassQC._dataset1]["_X"]].length
    );
    markerData[opacityPropertyName] = markerData[opacityPropertyName].map(
      function () {
        return 0.3;
      }
    );
    var confusions =
      dataUtils.data[ClassQC._dataset1]._processeddata["ClassQC_confusion"];

    for (var d in confusions) {
      markerData[opacityPropertyName][d] =
        dataUtils.data[ClassQC._dataset1]._processeddata["ClassQC_confusion"][
          d
        ] ==
        clicked_y + "_" + clicked_x;
    }
    glUtils.loadMarkers(ClassQC._dataset1);

    // DATASET 2
    var markerData = dataUtils.data[ClassQC._dataset2]["_processeddata"];

    dataUtils.data[ClassQC._dataset2]["_opacity_col"] = opacityPropertyName;
    markerData[opacityPropertyName] = new Float64Array(
      markerData[dataUtils.data[ClassQC._dataset2]["_X"]].length
    );
    markerData[opacityPropertyName] = markerData[opacityPropertyName].map(
      function () {
        return 0.3;
      }
    );
    var confusions =
      dataUtils.data[ClassQC._dataset1]._processeddata["ClassQC_confusion"];

    for (var d in confusions) {
      markerData[opacityPropertyName][d] =
        dataUtils.data[ClassQC._dataset1]._processeddata["ClassQC_confusion"][
          d
        ] ==
        clicked_y + "_" + clicked_x;
    }
    glUtils.loadMarkers(ClassQC._dataset2);

    glUtils.draw();
  });
};

ClassQC.run = function () {
  console.log(window["ClassQC_started"]);
  if (window["ClassQC_started"]) return;
  window["ClassQC_started"] = true;
  var op = tmapp["object_prefix"];
  var vname = op + "_viewer";

  var click_handler = function (event) {
    if (event.quick) {
      var OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];
      var viewportCoords = OSDviewer.viewport.pointFromPixel(event.position);
      var normCoords =
        OSDviewer.viewport.viewportToImageCoordinates(viewportCoords);

      var bbox = [
        Math.round(normCoords.x - ClassQC._bboxSize / 2),
        Math.round(normCoords.y - ClassQC._bboxSize / 2),
        ClassQC._bboxSize,
        ClassQC._bboxSize,
      ];
      var layers = ClassQC.getLayers();

      img = document.getElementById("ISS_ClassQC_img");
      console.log(img);
      if (img) img.style.filter = "blur(5px)";
      ClassQC.getMatrix(bbox, layers);

      color = "red";
      var boundBoxOverlay = document.getElementById("overlay-ClassQC");
      if (boundBoxOverlay) {
        OSDviewer.removeOverlay(boundBoxOverlay);
      }
      var boundBoxRect = OSDviewer.viewport.imageToViewportRectangle(
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3]
      );
      boundBoxOverlay = $('<div id="overlay-ClassQC"></div>');
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
  if (ClassQC.ISS_mouse_tracker == undefined) {
    /*ClassQC.ISS_mouse_tracker = new OpenSeadragon.MouseTracker({
            //element: this.fixed_svgov.node().parentNode,
            element: tmapp[vname].canvas,
            clickHandler: click_handler
        }).setTracking(true);*/

    ClassQC.ISS_mouse_tracker = tmapp["ISS_viewer"].addHandler(
      "canvas-click",
      (event) => {
        click_handler(event);
      }
    );
  }
};

ClassQC.changeOrder = function (doPrompt) {
  if (doPrompt == undefined) doPrompt = true;
  var layers = ClassQC.getLayers();
  rounds = [];
  channels = [];
  layers.forEach(function (layer, i) {
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
  ClassQC._order_rounds = rounds;
  ClassQC._order_channels = channels;
};

ClassQC.getLayers = function () {
  layers = [];
  /*if (tmapp.fixed_file && tmapp.fixed_file != "") {
        layers.push( {
            name:tmapp.slideFilename,
            tileSource: tmapp._url_suffix +  tmapp.fixed_file
        })
    }*/
  tmapp.layers.forEach(function (layer, i) {
    layers.push({
      name: layer.name,
      tileSource: layer.tileSource,
    });
  });
  return layers;
};

ClassQC.getMatrix = function (bbox, layers) {
  var op = tmapp["object_prefix"];
  var vname = op + "_viewer";
  console.log("Calling ajax getMatrix");
  const queryString = window.location.search;
  const urlParams = new URLSearchParams(queryString);
  const path = urlParams.get("path");
  $.ajax({
    // Post select to url.
    type: "post",
    url: "/plugins/ClassQC/getMatrix",
    contentType: "application/json; charset=utf-8",
    data: JSON.stringify({
      bbox: bbox,
      figureSize: ClassQC._figureSize,
      layers: layers,
      path: path,
      cmap: ClassQC._cmap,
    }),
    success: function (data) {
      img = document.getElementById("ISS_ClassQC_img");
      console.log("img", img);
      if (!img) {
        var img = document.createElement("img");
        img.id = "ISS_ClassQC_img";
        var elt = document.createElement("div");
        elt.classList.add("viewer-layer");
        elt.classList.add("px-1");
        elt.classList.add("mx-1");
        elt.appendChild(img);
        tmapp[vname].addControl(elt, {
          anchor: OpenSeadragon.ControlAnchor.BOTTOM_RIGHT,
        });
        elt.parentElement.parentElement.style.zIndex = "100";
        elt.style.display = "table";
      }
      img.setAttribute("src", "data:image/png;base64," + data);
      img.style.filter = "none";
    },
    complete: function (data) {
      // do something, not critical.
    },
    error: function (data) {
      console.log("Error:", data);
    },
  });
};
