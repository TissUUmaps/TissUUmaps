/**
 * @file Feature_Space_3D.js
 * @author Christophe Avenel
 */

/**
 * @namespace Feature_Space_3D
 * @classdesc The root namespace for Feature_Space_3D.
 */
var Feature_Space_3D;
Feature_Space_3D = {
  name: "Feature_Space_3D Plugin",
  _dataset: null,
  _UMAP1: null,
  _UMAP2: null,
  _UMAP3: null,
  _ortho: false,
};

/**
 * @summary */
Feature_Space_3D.init = function (container) {
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
    id: "Feature_Space_3D_dataset",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label212 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "Feature_Space_3D_dataset" },
  });
  label212.innerText = "Select marker dataset";

  row3 = HTMLElementUtils.createRow({});
  col31 = HTMLElementUtils.createColumn({ width: 12 });
  select311 = HTMLElementUtils.createElement({
    kind: "select",
    id: "Feature_Space_3D_UMAP1",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label312 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "Feature_Space_3D_UMAP1" },
  });
  label312.innerText = "Select Dimension 1";

  row4 = HTMLElementUtils.createRow({});
  col41 = HTMLElementUtils.createColumn({ width: 12 });
  select411 = HTMLElementUtils.createElement({
    kind: "select",
    id: "Feature_Space_3D_UMAP2",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label412 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "Feature_Space_3D_UMAP2" },
  });
  label412.innerText = "Select Dimension 2";

  row5 = HTMLElementUtils.createRow({});
  col51 = HTMLElementUtils.createColumn({ width: 12 });
  select511 = HTMLElementUtils.createElement({
    kind: "select",
    id: "Feature_Space_3D_UMAP3",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label512 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "Feature_Space_3D_UMAP3" },
  });
  label512.innerText = "Select Dimension 3";

  row6 = HTMLElementUtils.createRow({});
  col61 = HTMLElementUtils.createColumn({ width: 12 });
  var input611 = HTMLElementUtils.createElement({
    kind: "input",
    id: "Feature_Space_3D_ortho",
    extraAttributes: { class: "form-check-input", type: "checkbox" },
  });
  label611 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "Feature_Space_3D_ortho" },
  });
  label611.innerHTML = "&nbsp;Orthographic projection";

  input611.addEventListener("change", (event) => {
    Feature_Space_3D._ortho = input611.checked;
  });

  row7 = HTMLElementUtils.createRow({});
  col71 = HTMLElementUtils.createColumn({ width: 12 });
  button711 = HTMLElementUtils.createButton({
    extraAttributes: { class: "btn btn-primary mx-2" },
  });
  button711.innerText = "Display Feature Space";

  button111.addEventListener("click", (event) => {
    interfaceUtils.cleanSelect("Feature_Space_3D_dataset");
    interfaceUtils.cleanSelect("Feature_Space_3D_UMAP1");
    interfaceUtils.cleanSelect("Feature_Space_3D_UMAP2");
    interfaceUtils.cleanSelect("Feature_Space_3D_UMAP3");
    var datasets = Object.keys(dataUtils.data).map(function (e, i) {
      return {
        value: e,
        innerHTML: document.getElementById(e + "_tab-name").value,
      };
    });
    interfaceUtils.addObjectsToSelect("Feature_Space_3D_dataset", datasets);
    var event = new Event("change");
    interfaceUtils
      .getElementById("Feature_Space_3D_dataset")
      .dispatchEvent(event);
  });

  select211.addEventListener("change", (event) => {
    Feature_Space_3D._dataset = select211.value;
    if (!dataUtils.data[Feature_Space_3D._dataset]) return;
    interfaceUtils.cleanSelect("Feature_Space_3D_UMAP1");
    interfaceUtils.addElementsToSelect(
      "Feature_Space_3D_UMAP1",
      dataUtils.data[Feature_Space_3D._dataset]._csv_header,
    );
    interfaceUtils.cleanSelect("Feature_Space_3D_UMAP2");
    interfaceUtils.addElementsToSelect(
      "Feature_Space_3D_UMAP2",
      dataUtils.data[Feature_Space_3D._dataset]._csv_header,
    );
    interfaceUtils.cleanSelect("Feature_Space_3D_UMAP3");
    interfaceUtils.addElementsToSelect(
      "Feature_Space_3D_UMAP3",
      dataUtils.data[Feature_Space_3D._dataset]._csv_header,
    );
    if (
      dataUtils.data[Feature_Space_3D._dataset]._csv_header.indexOf("UMAP1") > 0
    ) {
      interfaceUtils.getElementById("Feature_Space_3D_UMAP1").value = "UMAP1";
      var event = new Event("change");
      interfaceUtils
        .getElementById("Feature_Space_3D_UMAP1")
        .dispatchEvent(event);
    }
    if (
      dataUtils.data[Feature_Space_3D._dataset]._csv_header.indexOf("UMAP2") > 0
    ) {
      interfaceUtils.getElementById("Feature_Space_3D_UMAP2").value = "UMAP2";
      var event = new Event("change");
      interfaceUtils
        .getElementById("Feature_Space_3D_UMAP2")
        .dispatchEvent(event);
    }
    if (
      dataUtils.data[Feature_Space_3D._dataset]._csv_header.indexOf("UMAP3") > 0
    ) {
      interfaceUtils.getElementById("Feature_Space_3D_UMAP3").value = "UMAP3";
      var event = new Event("change");
      interfaceUtils
        .getElementById("Feature_Space_3D_UMAP3")
        .dispatchEvent(event);
    }
  });
  select311.addEventListener("change", (event) => {
    Feature_Space_3D._UMAP1 = select311.value;
  });
  select411.addEventListener("change", (event) => {
    Feature_Space_3D._UMAP2 = select411.value;
  });
  select511.addEventListener("change", (event) => {
    Feature_Space_3D._UMAP3 = select511.value;
  });

  button711.addEventListener("click", (event) => {
    Feature_Space_3D.run();
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
  col51.appendChild(label512);
  col51.appendChild(select511);
  container.appendChild(row6);
  row6.appendChild(col61);
  col61.appendChild(input611);
  col61.appendChild(label611);
  container.appendChild(row7);
  row7.appendChild(col71);
  col71.appendChild(button711);
  var event = new Event("click");
  button111.dispatchEvent(event);
};

Feature_Space_3D.run = function () {
  var op = tmapp["object_prefix"];
  var vname = op + "_viewer";
  var Feature_Space_3D_Control = document.getElementById(
    "Feature_Space_3D_Control",
  );
  if (!Feature_Space_3D_Control) {
    Feature_Space_3D_Control = document.createElement("div");
    Feature_Space_3D_Control.id = "Feature_Space_3D_Control";
    Feature_Space_3D_Control.style.width = "100%";
    Feature_Space_3D_Control.style.height = "100%";
    Feature_Space_3D_Control.style.borderLeft = "1px solid #aaa";
    Feature_Space_3D_Control.style.display = "inline-block";
    var elt = document.createElement("div");
    elt.style.width = "40%";
    elt.style.height = "100%";
    elt.style.display = "inline-block";
    elt.style.verticalAlign = "top";
    elt.appendChild(Feature_Space_3D_Control);
    document.getElementById("ISS_viewer").appendChild(elt);
    $(".openseadragon-container")[0].style.display = "inline-flex";
    $(".openseadragon-container")[0].style.width = "60%";
  }
  data = Feature_Space_3D.getData();
  console.log(data);
  var layout = {
    margin: {
      l: 0,
      r: 0,
      b: 0,
      t: 0,
    },
    showlegend: false,
    scene: {
      xaxis: { showspikes: false },
      camera: {
        eye: { x: 1, y: 1, z: 1 },
        projection: {
          type: Feature_Space_3D._ortho ? "orthographic" : "perspective",
        },
      },
    },
  };
  Plotly.newPlot(Feature_Space_3D_Control, data, layout, {
    responsive: true,
    displayModeBar: false,
  });

  if (interfaceUtils.Feature_Space_3D_toggleRightPanel === undefined) {
    interfaceUtils.Feature_Space_3D_toggleRightPanel =
      interfaceUtils.toggleRightPanel;
    interfaceUtils.toggleRightPanel = function () {
      interfaceUtils.Feature_Space_3D_toggleRightPanel();
      Plotly.Plots.resize(document.getElementById("Feature_Space_3D_Control"));
    };
  }
  var copySettings = function () {
    setTimeout(function () {
      var plotDiv = document.getElementById("Feature_Space_3D_Control");
      const data = Feature_Space_3D.getData();
      var plotData = plotDiv.data;
      for (var trace in data) {
        plotData[trace].opacity = data[trace].marker.opacity;
      }
      Plotly.redraw(plotDiv);
      //Plotly.redraw(document.getElementById("Feature_Space_3D_Control"))
      //Plotly.newPlot(document.getElementById("Feature_Space_3D_Control"), data);;
    }, 50);
  };
  var copyData = function () {
    setTimeout(function () {
      var plotDiv = document.getElementById("Feature_Space_3D_Control");
      const data = Feature_Space_3D.getData();
      var plotData = plotDiv.data;
      for (var trace in data) {
        plotData[trace].x = data[trace].x;
        plotData[trace].y = data[trace].y;
        plotData[trace].z = data[trace].z;
        plotData[trace].marker = data[trace].marker;
      }
      Plotly.redraw(plotDiv);
      copySettings();
      //Plotly.redraw(document.getElementById("Feature_Space_3D_Control"))
      //Plotly.newPlot(document.getElementById("Feature_Space_3D_Control"), data);;
    }, 50);
  };
  if (glUtils.Feature_Space_3D_draw === undefined) {
    glUtils.Feature_Space_3D_draw = glUtils.draw;
    glUtils.draw = function () {
      glUtils.Feature_Space_3D_draw();
      copySettings();
    };
    glUtils.Feature_Space_3D_updateColorLUTTexture =
      glUtils._updateColorLUTTexture;
    glUtils._updateColorLUTTexture = function (gl, uid, texture) {
      glUtils.Feature_Space_3D_updateColorLUTTexture(gl, uid, texture);
      copySettings();
    };
    dataUtils.Feature_Space_3D_updateViewOptions = dataUtils.updateViewOptions;
    dataUtils.updateViewOptions = function (data_id) {
      dataUtils.Feature_Space_3D_updateViewOptions(data_id);
      copyData();
    };
  }

  function run() {
    rotate("scene", Math.PI / 360);
    requestAnimationFrame(run);
  }
  //run();

  function rotate(id, angle) {
    var eye0 = document.getElementById("Feature_Space_3D_Control").layout[id]
      .camera.eye;
    var rtz = xyz2rtz(eye0);
    rtz.t += angle;

    var eye1 = rtz2xyz(rtz);
    Plotly.relayout(
      document.getElementById("Feature_Space_3D_Control"),
      id + ".camera.eye",
      eye1,
    );
  }

  function xyz2rtz(xyz) {
    return {
      r: Math.sqrt(xyz.x * xyz.x + xyz.y * xyz.y),
      t: Math.atan2(xyz.y, xyz.x),
      z: xyz.z,
    };
  }

  function rtz2xyz(rtz) {
    return {
      x: rtz.r * Math.cos(rtz.t),
      y: rtz.r * Math.sin(rtz.t),
      z: rtz.z,
    };
  }
};
if (!Array.prototype.chunk) {
  Object.defineProperty(Array.prototype, "chunk", {
    value: function (n) {
      return Array(Math.ceil(this.length / n))
        .fill()
        .map((_, i) => this.slice(i * n, i * n + n));
    },
  });
}

Feature_Space_3D.getData = function () {
  const dataset = Feature_Space_3D._dataset;
  const markerData = dataUtils.data[dataset]["_processeddata"];
  const numPoints = markerData[markerData.columns[0]].length;
  const keyName = dataUtils.data[uid]["_gb_col"];

  const scalarPropertyName = dataUtils.data[uid]["_cb_col"];
  const useColorFromMarker =
    dataUtils.data[uid]["_cb_col"] != null &&
    dataUtils.data[uid]["_cb_cmap"] == null;
  const colorscaleName = dataUtils.data[uid]["_cb_cmap"];
  const useColorFromColormap = dataUtils.data[uid]["_cb_cmap"] != null;

  var key, X, Y, Z;
  var markers3D = {};
  for (let i = 0, index = 0; i < numPoints; ++i) {
    key = keyName ? markerData[keyName][i] : "All";
    X = markerData[Feature_Space_3D._UMAP1][i];
    Y = markerData[Feature_Space_3D._UMAP2][i];
    Z = markerData[Feature_Space_3D._UMAP3][i];
    if (markers3D[key] === undefined) {
      const inputs = interfaceUtils._mGenUIFuncs.getGroupInputs(dataset, key);
      var hexColor;
      if (useColorFromColormap || useColorFromMarker) {
        hexColor = [];
      } else {
        hexColor = "color" in inputs ? inputs["color"] : "#ffff00";
      }
      var visible = "visible" in inputs ? inputs["visible"] : true;
      const shape = "shape" in inputs ? inputs["shape"] : "circle";
      const hidden = "hidden" in inputs ? inputs["hidden"] : true;
      if (hidden) visible = false;
      markers3D[key] = {
        x: [],
        y: [],
        z: [],
        mode: "markers",
        marker: {
          color: hexColor,
          size: 2,
          symbol: shape,
          opacity: visible ? 1 : 0.000001,
        },
        type: "scatter3d",
        name: key,
        meta: key.toString(),
        hovertemplate: "%{meta}<extra></extra>",
      };
      if (useColorFromColormap) {
        markers3D[key].marker.colorscale = glUtils._colorscaleData[dataset]
          .chunk(4)
          .map(function (RGBA, index) {
            return [
              index / 255,
              "rgb(" +
                parseInt(RGBA[0]) +
                "," +
                parseInt(RGBA[1]) +
                "," +
                parseInt(RGBA[2]) +
                ")",
            ];
          });
      }
      if (useColorFromColormap || useColorFromMarker) {
        markers3D[key].hovertemplate = "%{marker.color}<extra></extra>";
      }
    }
    markers3D[key].x.push(X);
    markers3D[key].y.push(Y);
    markers3D[key].z.push(Z);
    if (useColorFromColormap || useColorFromMarker) {
      markers3D[key].marker.color.push(markerData[scalarPropertyName][i]);
    }
  }
  return Object.values(markers3D);
};
