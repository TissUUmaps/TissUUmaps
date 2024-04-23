/**
 * @file DGE.js
 * @author Christophe Avenel
 */

/**
 * @namespace DGE
 * @classdesc The root namespace for DGE.
 */
var DGE;
DGE = {
  name: "Differential Gene Expression",
  parameters: {},
};

/**
 * This method is called when the document is loaded.
 * The container element is a div where the plugin options will be displayed.
 * @summary After setting up the tmapp object, initialize it*/
DGE.init = function (container) {
  DGE.container = container;
  var script = document.createElement("script");
  script.src = "https://cdn.plot.ly/plotly-latest.min.js";
  document.head.appendChild(script);
  DGE._pickedMarker = null;

  if (glUtils.temp_pick === undefined) {
    glUtils.temp_pick = glUtils.pick;
    tmapp["ISS_viewer"].removeHandler("canvas-click", glUtils.pick);
    glUtils.pick = function (event) {
      glUtils.temp_pick(event);
      if (DGE._pickedMarker == glUtils._pickedMarker) return;
      DGE._pickedMarker = glUtils._pickedMarker;
      console.log(glUtils._pickedMarker);
      let _dataset = Object.keys(dataUtils.data)[0];
      data_obj = dataUtils.data[_dataset];
      if (data_obj === undefined) return;
      let obs = data_obj._gb_col;
      if (dataUtils.data[_dataset]._processeddata[obs] === undefined) return;
      cluster =
        dataUtils.data[_dataset]._processeddata[obs][glUtils._pickedMarker[1]];

      console.log(cluster);
      DGE.getDGE(cluster);
    };
    tmapp["ISS_viewer"].addHandler("canvas-click", glUtils.pick);
  }
};

DGE.getDGE = async function (cluster_name) {
  DGE.container.innerHTML = "";
  let _dataset = Object.keys(dataUtils.data)[0];
  data_obj = dataUtils.data[_dataset];

  let obs = data_obj._gb_col.replace(/\/obs/g, "").replace(/\//g, "");
  console.log(obs);
  let matrix = await dataUtils._hdf5Api.get(data_obj._csv_path, {
    path: "/uns/" + obs + "_wilcoxon/names",
  });
  if (matrix.type == "error") {
    DGE.container.innerHTML =
      "<i style='color:red;'>No DGE found in /uns/" +
      obs +
      "_wilcoxon. Please run the wilcoxon test first.</i>";
    return;
  }
  if (matrix === undefined) {
    console.log("No DGE found");
    return;
  }
  console.log(matrix);
  names = matrix.value;
  console.log(names);
  //transpose names
  names = names[0].map((col, i) => names.map((row) => row[i]));
  clusters = await dataUtils._hdf5Api.get(data_obj._csv_path, {
    path: "/obs/" + obs + "/categories",
  });
  clusters = clusters.value;
  cluster = clusters.indexOf(cluster_name);
  console.log(cluster_name, clusters, cluster);
  names = names[cluster];
  console.log(names);
  if (names === undefined) return;
  let matrix2 = await dataUtils._hdf5Api.get(data_obj._csv_path, {
    path: "/uns/" + obs + "_wilcoxon/scores",
  });
  scores = matrix2.value;
  console.log(scores);
  //transpose scores
  scores = scores[0].map((col, i) => scores.map((row) => row[i]));
  scores = scores[cluster];
  console.log(scores);
  var var_names = await dataUtils._hdf5Api.get(data_obj._csv_path, {
    path: "/var/_index",
  });
  if (var_names.type == "error") {
    var_names = await dataUtils._hdf5Api.get(data_obj._csv_path, {
      path: "/var/Gene",
    });
  }
  var_names = var_names.value;
  var tabName = interfaceUtils.getElementById(
    _dataset + "_marker-tab-name",
  ).textContent;
  var hovering = false;
  async function update_color(data) {
    hovering = true;
    // set plot cursor to wait
    dragLayer = document.getElementsByClassName("nsewdrag");
    for (let i = 0; i < dragLayer.length; i++) {
      dragLayer[i].style.cursor = "wait";
    }
    // await 1 second
    await new Promise((r) => setTimeout(r, 1000));
    if (!hovering) {
      for (let i = 0; i < dragLayer.length; i++) {
        dragLayer[i].style.cursor = "default";
      }
      return;
    }
    let point = data.points[0];
    let gene = point.x;
    let score = point.y;
    console.log(gene);
    // find gene in var_names
    let idx = var_names.indexOf(gene);
    // use gb_col as /X/idx
    let _dataset = Object.keys(dataUtils.data)[0];
    data_obj = dataUtils.data[_dataset];
    if (data_obj._processeddata[`/X;${idx}`] === undefined) {
      data_obj._processeddata[`/X;${idx}`] = await dataUtils._hdf5Api.getXRow(
        data_obj._csv_path,
        idx,
        "/X",
      );
    }
    if (!hovering) {
      for (let i = 0; i < dragLayer.length; i++) {
        dragLayer[i].style.cursor = "default";
      }
      return;
    }
    data_obj._cb_col = `/X;${idx}`;
    data_obj._cb_cmap = "interpolateViridis";
    data_obj._sortby_col = `/X;${idx}`;
    data_obj._sortby_desc = false;
    // change tab name
    interfaceUtils.getElementById(_dataset + "_marker-tab-name").textContent =
      gene;
    glUtils.loadMarkers(_dataset, true);
    glUtils.draw();
    for (let i = 0; i < dragLayer.length; i++) {
      dragLayer[i].style.cursor = "default";
    }
  }
  async function update_color_selection(eventData) {
    if (eventData === undefined) {
      reset_color();
      return;
    }
    // set plot cursor to wait
    dragLayer = document.getElementsByClassName("nsewdrag");
    for (let i = 0; i < dragLayer.length; i++) {
      dragLayer[i].style.cursor = "wait";
    }
    // get gene selected in the plot
    console.log(data);
    var selectedPoints = eventData.points;
    if (selectedPoints.length == 0) {
      reset_color();
      return;
    }
    var all_data = [];
    var selected_genes = [];
    for (var i = 0; i < selectedPoints.length; i++) {
      let point = selectedPoints[i];
      let gene = point.x;
      let score = point.y;
      selected_genes.push(gene);
      // find gene in var_names
      let idx = var_names.indexOf(gene);
      // use gb_col as /X/idx
      let _dataset = Object.keys(dataUtils.data)[0];
      data_obj = dataUtils.data[_dataset];
      if (data_obj._processeddata[`/X;${idx}`] === undefined) {
        data_obj._processeddata[`/X;${idx}`] = await dataUtils._hdf5Api.getXRow(
          data_obj._csv_path,
          idx,
          "/X",
        );
      }
      all_data.push(data_obj._processeddata[`/X;${idx}`]);
    }
    // sum all_data arrays point wise into one array
    let sum = all_data.reduce((a, b) => a.map((x, i) => x + b[i]));
    data_obj._processeddata[`selection`] = sum;
    data_obj._cb_col = `selection`;
    data_obj._cb_cmap = "interpolateViridis";
    data_obj._sortby_col = `selection`;
    data_obj._sortby_desc = false;
    // change tab name
    interfaceUtils.getElementById(_dataset + "_marker-tab-name").textContent =
      selected_genes.join(" + ");
    glUtils.loadMarkers(_dataset, true);
    glUtils.draw();
    for (let i = 0; i < dragLayer.length; i++) {
      dragLayer[i].style.cursor = "default";
    }
  }
  function reset_color() {
    hovering = false;
    dragLayer = document.getElementsByClassName("nsewdrag");
    for (let i = 0; i < dragLayer.length; i++) {
      dragLayer[i].style.cursor = "default";
    }
    let _dataset = Object.keys(dataUtils.data)[0];
    data_obj = dataUtils.data[_dataset];
    if (data_obj._cb_col === null) return;
    data_obj._cb_col = null;
    data_obj._cb_cmap = null;
    interfaceUtils.getElementById(_dataset + "_marker-tab-name").textContent =
      tabName;
    glUtils.loadMarkers(_dataset, true);
    glUtils.draw();
  }
  // plot the DGE with plotly
  // with names on the x-axis and p-values on the y-axis
  let x = [];
  let y = [];
  let text = [];
  let colors = [];
  let size = [];
  for (let i = 0; i < 25; i++) {
    x.push(names[i]);
    y.push(scores[i]); //-Math.log10(scores[i]));
    text.push(names[i]);
    colors.push("rgba(0,0,0,0.5)");
    size.push(10);
  }
  let trace = {
    x: x,
    y: y,
    mode: "markers",
    type: "scatter",
    text: text,
    marker: {
      color: colors,
      size: size,
    },
  };
  let layout = {
    dragmode: "select",
    autosize: true,
    title: "<b>Cluster " + cluster_name + "</b> DGE (25 top)",
    height: 400,
    xaxis: {
      title: "Genes",
      dtick: 1,
    },
    yaxis: {
      title: "scores",
    },
    margin: {
      l: 60,
      r: 20,
      b: 100,
      t: 60,
      pad: 5,
    },
  };
  let data = [trace];
  let plot = document.createElement("div");
  plot.id = "plot";
  DGE.container.appendChild(plot);
  let options = {
    responsive: true,
    displayModeBar: false,
  };
  Plotly.newPlot(plot, data, layout, options);
  //plot.on('plotly_click', update_color);
  //plot.on('plotly_unhover', reset_color);
  plot.on("plotly_selected", update_color_selection);

  // plot the DGE with plotly
  // with names on the x-axis and p-values on the y-axis
  x = [];
  y = [];
  text = [];
  colors = [];
  size = [];
  for (let i = names.length - 25; i < names.length; i++) {
    x.push(names[i]);
    y.push(scores[i]); //-Math.log10(scores[i]));
    text.push(names[i]);
    colors.push("rgba(0,0,0,0.5)");
    size.push(10);
  }
  trace = {
    x: x,
    y: y,
    mode: "markers",
    type: "scatter",
    text: text,
    marker: {
      color: colors,
      size: size,
    },
  };
  layout = {
    dragmode: "select",
    title: "<b>Cluster " + cluster_name + "</b> DGE (25 bottom)",
    height: 400,
    xaxis: {
      title: "Genes",
      dtick: 1,
    },
    yaxis: {
      title: "scores",
    },
    margin: {
      l: 60,
      r: 20,
      b: 100,
      t: 60,
      pad: 5,
    },
  };
  data = [trace];
  plot = document.createElement("div");
  plot.id = "plot";
  DGE.container.appendChild(plot);
  Plotly.newPlot(plot, data, layout, options);
  // add listener on mouse over points of the plot
  //plot.on('plotly_click', update_color);
  //plot.on('plotly_unhover', reset_color);
  // add listener on horizontal selection
  plot.on("plotly_selected", update_color_selection);
};
