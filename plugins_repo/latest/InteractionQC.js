/**
 * @file InteractionQC.js
 * @author Andrea Behanova, Christophe Avenel
 */

/**
 * @namespace InteractionQC
 * @classdesc The root namespace for InteractionQC.
 */
var InteractionQC;
InteractionQC = {
  name: "InteractionQC Plugin",
  _dataset: null,
  _matrix: null,
  _matrix_header: null,
  _region: null,
  _regionPixels: null,
  _regionWin: null,
  _newwin: null,
};

/**
 * @summary */
InteractionQC.init = function (container) {
  var script = document.createElement("script");
  script.src = "https://cdn.plot.ly/plotly-latest.min.js";
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
    id: "InteractionQC_dataset",
    extraAttributes: {
      class: "form-select form-select-sm",
      "aria-label": ".form-select-sm",
    },
  });
  label212 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "InteractionQC_dataset" },
  });
  label212.innerText = "Select marker dataset";

  row4 = HTMLElementUtils.createRow({});
  col41 = HTMLElementUtils.createColumn({ width: 12 });
  input411 = HTMLElementUtils.createElement({
    kind: "input",
    id: "InteractionQC_csv",
    extraAttributes: {
      name: "InteractionQC_csv",
      class: "form-control-file form-control form-control-sm",
      type: "file",
      accept: ".csv,.tsv,.txt",
    },
  });
  label412 = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: { for: "matrix" },
  });
  label412.innerText = "Select file";

  row5 = HTMLElementUtils.createRow({});
  col51 = HTMLElementUtils.createColumn({ width: 12 });
  button511 = HTMLElementUtils.createButton({
    extraAttributes: { class: "btn btn-primary mx-2" },
  });
  button511.innerText = "Display Neighborhood Enrichment Test";

  row12 = HTMLElementUtils.createRow({});
  col121 = HTMLElementUtils.createElement({
    kind: "div",
    id: "InteractionQC_matrix",
  });
  col131 = HTMLElementUtils.createElement({
    kind: "div",
    id: "InteractionQC_legend",
  });

  // Refresh button Refresh drop-down lists based on loaded markers
  button111.addEventListener("click", (event) => {
    interfaceUtils.cleanSelect("InteractionQC_dataset");
    interfaceUtils.cleanSelect("Column");
    interfaceUtils.cleanSelect("matrix");

    var datasets = Object.keys(dataUtils.data).map(function (e, i) {
      return {
        value: e,
        innerHTML: document.getElementById(e + "_tab-name").value,
      };
    });
    interfaceUtils.addObjectsToSelect("InteractionQC_dataset", datasets);
    var event = new Event("change");
    interfaceUtils.getElementById("InteractionQC_dataset").dispatchEvent(event);
  });

  // Marker dataset selector
  select211.addEventListener("change", (event) => {
    InteractionQC._dataset = select211.value;
    if (!dataUtils.data[InteractionQC._dataset]) return;
    interfaceUtils.cleanSelect("Column");
    interfaceUtils.addElementsToSelect(
      "Column",
      dataUtils.data[InteractionQC._dataset]._csv_header,
    );
    interfaceUtils.cleanSelect("matrix");
    interfaceUtils.addElementsToSelect(
      "matrix",
      dataUtils.data[InteractionQC._dataset]._csv_header,
    );
    if (
      dataUtils.data[InteractionQC._dataset]._csv_header.indexOf("Column") > 0
    ) {
      interfaceUtils.getElementById("Column").value = "Column";
      var event = new Event("change");
      interfaceUtils.getElementById("Column").dispatchEvent(event);
    }
    if (
      dataUtils.data[InteractionQC._dataset]._csv_header.indexOf("matrix") > 0
    ) {
      interfaceUtils.getElementById("matrix").value = "matrix";
      var event = new Event("change");
      interfaceUtils.getElementById("matrix").dispatchEvent(event);
    }
  });

  input411.addEventListener("change", (event) => {
    var reader = new FileReader();

    function loadFile() {
      var file = input411.files[0];
      reader.addEventListener("load", parseFile, false);
      if (file) {
        reader.readAsText(file);
      }
    }

    function parseFile() {
      rows = Plotly.d3.csv.parseRows(reader.result);
      /*function unpack(rows, key) {
        return rows.map(function(row) { return row[key]; });
      }*/
      console.log(rows);
      rows = rows.map(function (row) {
        row.shift();
        return row;
      });
      InteractionQC._matrix_header = rows.shift();
      rows = rows.map(function (row) {
        return row.map(Number);
      });
      InteractionQC._matrix = rows.reverse();
    }
    loadFile();
    parseFile();
  });

  button511.addEventListener("click", (event) => {
    InteractionQC.run();
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
  container.appendChild(row4);
  row4.appendChild(col41);
  col41.appendChild(label412);
  col41.appendChild(input411);
  container.appendChild(row5);
  row5.appendChild(col51);
  col51.appendChild(button511);
  container.appendChild(row12);
  row12.appendChild(col121);
  row12.appendChild(col131);

  var event = new Event("click");
  button111.dispatchEvent(event);
};

InteractionQC.run = function () {
  var op = tmapp["object_prefix"];
  var vname = op + "_viewer";
  var InteractionQC_Control = document.getElementById("InteractionQC_Control");

  var data = [
    {
      z: InteractionQC._matrix,
      x: InteractionQC._matrix_header,
      y: InteractionQC._matrix_header.slice().reverse(),
      type: "heatmap",
      hoverongaps: false,
      colorscale: "Hot",
    },
  ];

  var layout = {
    autosize: true,
    automargin: true,
    showlegend: true,
    yaxis: {
      side: "top",
      tickmode: "array",
      tickvals: InteractionQC._matrix_header,
      ticktext: InteractionQC._matrix_header.map(function (text) {
        color = document.getElementById(
          InteractionQC._dataset + "_" + text + "_color",
        ).value;
        return "<span style='font-weight:bold;color:" + color + "'>███</span>";
      }),
      ticks: "",
      tickangle: 90,
      title: {
        text: "Cell class 2",
        font: {
          size: 25,
          color: "black",
        },
      },
    },
    xaxis: {
      // "scaleanchor":"x",
      tickvals: InteractionQC._matrix_header.slice().reverse(),
      ticktext: InteractionQC._matrix_header
        .slice()
        .reverse()
        .map(function (text) {
          color = document.getElementById(
            InteractionQC._dataset + "_" + text + "_color",
          ).value;
          return (
            "<span style='font-weight:bold;color:" + color + "'>███</span>"
          );
        }),
      ticks: "",
      tickangle: 0,
      title: {
        text: "Cell class 1",
        font: {
          size: 25,
          color: "black",
        },
      },
      ticklabelposition: "top",
      side: "top",
    },
    annotations: [],
    title: null,
  };

  Plotly.newPlot(
    document.getElementById("InteractionQC_matrix"),
    data,
    layout,
    {
      responsive: true,
      displayModeBar: false,
    },
  );

  legend = "";
  for (type of InteractionQC._matrix_header) {
    typecolor = document.getElementById(
      InteractionQC._dataset + "_" + type + "_color",
    ).value;
    legend +=
      "<div style='display:inline-block;margin-right:10px;'><span style='width:15px;color:" +
      typecolor +
      "'>█</span><span style='min-width:150px;margin: 0px 5px;'>" +
      type +
      "</span></div>";
  }
  document.getElementById("InteractionQC_legend").innerHTML = legend;

  // InteractionQC_Control.on('plotly_click', function(data){
  document
    .getElementById("InteractionQC_matrix")
    .on("plotly_click", function (data) {
      console.log(data.points[0].x, data.points[0].y);
      var clicked_x = data.points[0].x.replace(/ /g, "_");
      var clicked_y = data.points[0].y.replace(/ /g, "_");
      var uid = InteractionQC._dataset;
      console.log(uid + "_" + clicked_x);
      document.getElementById(uid + "_all_check").checked = true;
      document.getElementById(uid + "_all_check").click();
      document.getElementById(uid + "_" + clicked_x + "_check").click();
      if (clicked_x != clicked_y)
        document.getElementById(uid + "_" + clicked_y + "_check").click();

      /*var pts = '';
      for(var i=0; i < data.points.length; i++){
          pts = 'x = '+data.points[i].x +'\ny = '+
              data.points[i].y.toPrecision(4) + '\n\n';
      }
      alert('Closest point clicked:\n\n'+pts);*/
    });
};

InteractionQC.getData = function () {};
