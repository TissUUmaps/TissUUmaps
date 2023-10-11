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
  _dataset: null,
  _clusterKey: false,
  _nclusters: 8,
  _expression_threshold: 1,
  _normalize_order: 1,
  _sigma: 15,
  _stride: 5,
  _region_name: "Clusters",
};

/**
 * @summary */
Points2Regions.init = function (container) {
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
    id: "Points2Regions_dataset",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label212 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "Points2Regions_dataset" },
  });
  label212.innerText = "Select marker dataset";

  row7 = HTMLElementUtils.createRow({});
  col71 = HTMLElementUtils.createColumn({ width: 12 });
  select711 = HTMLElementUtils.createElement({
    kind: "select",
    id: "Points2Regions_clusterKey",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label712 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "Points2Regions_clusterKey" },
  });
  label712.innerText = "Select Points2Regions Key";

  row_nclusters = HTMLElementUtils.createRow({});
  col_nclusters1 = HTMLElementUtils.createColumn({ width: 12 });
  label_nclusters12 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: {
      class: "form-check-label",
      for: "Points2Regions_bboxSize",
    },
  });
  label_nclusters12.innerHTML = "Number of clusters:";
  var input_nclusters12 = HTMLElementUtils.createElement({
    kind: "input",
    id: "Points2Regions_nclusters",
    extraAttributes: {
      class: "form-text-input form-control",
      type: "number",
      value: Points2Regions._nclusters,
    },
  });

  input_nclusters12.addEventListener("change", (event) => {
    Points2Regions._nclusters = parseInt(input_nclusters12.value);
  });

  row_expression_threshold = HTMLElementUtils.createRow({});
  col_expression_threshold1 = HTMLElementUtils.createColumn({ width: 12 });
  label_expression_threshold12 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: {
      class: "form-check-label",
      for: "Points2Regions_bboxSize",
    },
  });
  label_expression_threshold12.innerHTML = "Expression threshold:";
  var input_expression_threshold12 = HTMLElementUtils.createElement({
    kind: "input",
    id: "Points2Regions_expression_threshold",
    extraAttributes: {
      class: "form-text-input form-control",
      type: "number",
      value: Points2Regions._expression_threshold,
    },
  });

  input_expression_threshold12.addEventListener("change", (event) => {
    Points2Regions._expression_threshold = parseInt(
      input_expression_threshold12.value,
    );
  });

  row_normalize_order = HTMLElementUtils.createRow({});
  col_normalize_order1 = HTMLElementUtils.createColumn({ width: 12 });
  label_normalize_order12 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: {
      class: "form-check-label",
      for: "Points2Regions_bboxSize",
    },
  });
  label_normalize_order12.innerHTML = "Normalize order:";
  var input_normalize_order12 = HTMLElementUtils.createElement({
    kind: "input",
    id: "Points2Regions_normalize_order",
    extraAttributes: {
      class: "form-text-input form-control",
      type: "number",
      value: Points2Regions._normalize_order,
    },
  });

  input_normalize_order12.addEventListener("change", (event) => {
    Points2Regions._normalize_order = parseInt(input_normalize_order12.value);
  });

  row_sigma = HTMLElementUtils.createRow({});
  col_sigma1 = HTMLElementUtils.createColumn({ width: 12 });
  label_sigma12 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: {
      class: "form-check-label",
      for: "Points2Regions_bboxSize",
    },
  });
  label_sigma12.innerHTML = "Sigma:";
  var input_sigma12 = HTMLElementUtils.createElement({
    kind: "input",
    id: "Points2Regions_sigma",
    extraAttributes: {
      class: "form-text-input form-control",
      type: "number",
      value: Points2Regions._sigma,
    },
  });

  input_sigma12.addEventListener("change", (event) => {
    Points2Regions._sigma = parseInt(input_sigma12.value);
  });

  row_stride = HTMLElementUtils.createRow({});
  col_stride1 = HTMLElementUtils.createColumn({ width: 12 });
  label_stride12 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: {
      class: "form-check-label",
      for: "Points2Regions_bboxSize",
    },
  });
  label_stride12.innerHTML = "Bin width:";
  var input_stride12 = HTMLElementUtils.createElement({
    kind: "input",
    id: "Points2Regions_stride",
    extraAttributes: {
      class: "form-text-input form-control",
      type: "number",
      value: Points2Regions._stride,
    },
  });

  input_stride12.addEventListener("change", (event) => {
    Points2Regions._stride = parseInt(input_stride12.value);
  });

  button111.addEventListener("click", (event) => {
    interfaceUtils.cleanSelect("Points2Regions_dataset");
    interfaceUtils.cleanSelect("Points2Regions_clusterKey");

    var datasets = Object.keys(dataUtils.data).map(function (e, i) {
      return {
        value: e,
        innerHTML: document.getElementById(e + "_tab-name").value,
      };
    });
    interfaceUtils.addObjectsToSelect("Points2Regions_dataset", datasets);
    var event = new Event("change");
    interfaceUtils
      .getElementById("Points2Regions_dataset")
      .dispatchEvent(event);
  });
  select211.addEventListener("change", (event) => {
    Points2Regions._dataset = select211.value;
    if (!dataUtils.data[Points2Regions._dataset]) return;
    interfaceUtils.cleanSelect("Points2Regions_clusterKey");
    interfaceUtils.addElementsToSelect(
      "Points2Regions_clusterKey",
      dataUtils.data[Points2Regions._dataset]._csv_header,
    );
    if (
      dataUtils.data[Points2Regions._dataset]._csv_header.indexOf(
        dataUtils.data[Points2Regions._dataset]._gb_col,
      ) > 0
    ) {
      interfaceUtils.getElementById("Points2Regions_clusterKey").value =
        dataUtils.data[Points2Regions._dataset]._gb_col;
      var event = new Event("change");
      interfaceUtils
        .getElementById("Points2Regions_clusterKey")
        .dispatchEvent(event);
    }
  });
  select711.addEventListener("change", (event) => {
    alert(select711.value);
    Points2Regions._clusterKey = select711.value;
  });

  row_submit = HTMLElementUtils.createRow({});
  col_submit1 = HTMLElementUtils.createColumn({ width: 12 });
  button_submit11 = HTMLElementUtils.createButton({
    extraAttributes: { class: "btn btn-primary mx-2" },
  });
  button_submit11.innerText = "Run Points2Regions";

  button_submit11.addEventListener("click", (event) => {
    Points2Regions.run();
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
  container.appendChild(row_nclusters);
  row_nclusters.appendChild(col_nclusters1);
  col_nclusters1.appendChild(label_nclusters12);
  col_nclusters1.appendChild(input_nclusters12);
  container.appendChild(row_expression_threshold);
  row_expression_threshold.appendChild(col_expression_threshold1);
  col_expression_threshold1.appendChild(label_expression_threshold12);
  col_expression_threshold1.appendChild(input_expression_threshold12);
  container.appendChild(row_normalize_order);
  row_normalize_order.appendChild(col_normalize_order1);
  col_normalize_order1.appendChild(label_normalize_order12);
  col_normalize_order1.appendChild(input_normalize_order12);
  container.appendChild(row_sigma);
  row_sigma.appendChild(col_sigma1);
  col_sigma1.appendChild(label_sigma12);
  col_sigma1.appendChild(input_sigma12);
  container.appendChild(row_stride);
  row_stride.appendChild(col_stride1);
  col_stride1.appendChild(label_stride12);
  col_stride1.appendChild(input_stride12);
  container.appendChild(row_submit);
  row_submit.appendChild(col_submit1);
  col_submit1.appendChild(button_submit11);
  var event = new Event("click");
  button111.dispatchEvent(event);
};

Points2Regions.run = function () {
  // Get csv filename from dataset
  var csvFile = dataUtils.data[Points2Regions._dataset]._csv_path;
  if (typeof csvFile === "object") {
    interfaceUtils.alert(
      "This plugin can only run on datasets generated from buttons. Please convert your dataset to a button (Markers > Advanced Options > Generate button from tab)",
    );
    return;
  }
  // Get the path from url:
  const queryString = window.location.search;
  const urlParams = new URLSearchParams(queryString);
  const path = urlParams.get("path");
  if (path != null) {
    csvFile = path + "/" + csvFile;
  }
  loadingModal = interfaceUtils.loadingModal("Points2Regions... Please wait.");
  $.ajax({
    type: "post",
    url: "/plugins/Points2Regions/Points2Regions",
    contentType: "application/json; charset=utf-8",
    data: JSON.stringify({
      xKey: dataUtils.data[Points2Regions._dataset]._X,
      yKey: dataUtils.data[Points2Regions._dataset]._Y,
      clusterKey: Points2Regions._clusterKey,
      nclusters: Points2Regions._nclusters,
      expression_threshold: Points2Regions._expression_threshold,
      normalize_order: Points2Regions._normalize_order,
      sigma: Points2Regions._sigma,
      stride: Points2Regions._stride,
      region_name: Points2Regions._region_name,
      csv_path: csvFile,
    }),
    success: function (data) {
      Points2Regions.loadRegions(data);
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
  regionUtils.JSONValToRegions(regionsobj);
  $("#title-tab-regions").tab("show");
  $(
    document.getElementById("regionClass-" + Points2Regions._region_name),
  ).collapse("show");
  $("#" + Points2Regions._region_name + "_group_fill_ta").click();
};
