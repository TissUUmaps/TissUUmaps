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
  parameters: {
    _run: {
      label: "Display Neighborhood Enrichment Test",
      type: "button",
    },
    _AdvancedSection: {
      label:
        "If you have a precomputed matrix in csv format, you can upload it here. Headers must be the same as the cell class names.",
      title: "ADVANCED OPTIONS",
      type: "section",
      collapsed: true,
    },
  },
  _matrix: null,
  _matrix_header: null,
};

/**
 * @summary */
InteractionQC.init = function (container) {
  var script = document.createElement("script");
  script.src = "https://cdn.plot.ly/plotly-latest.min.js";
  document.head.appendChild(script);
  let advanced_section = document.getElementById(
    "collapsedSection_InteractionQC__AdvancedSection",
  );

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

  row12 = HTMLElementUtils.createRow({});
  col121 = HTMLElementUtils.createElement({
    kind: "div",
    id: "InteractionQC_matrix",
  });
  col131 = HTMLElementUtils.createElement({
    kind: "div",
    id: "InteractionQC_legend",
  });
  advanced_section.appendChild(row4);
  row4.appendChild(col41);
  col41.appendChild(label412);
  col41.appendChild(input411);

  container.appendChild(row12);
  row12.appendChild(col121);
  row12.appendChild(col131);

  /*container.innerHTML = "";
  // container.appendChild(row0);
  container.appendChild(row1);
  row1.appendChild(col11);
  col11.appendChild(button111);
  container.appendChild(row2);
  row2.appendChild(col21);
  col21.appendChild(label212);
  col21.appendChild(select211);

  container.appendChild(row5);
  row5.appendChild(col51);
  col51.appendChild(button511);
  container.appendChild(row12);
  row12.appendChild(col121);
  row12.appendChild(col131);

  var event = new Event("click");
  button111.dispatchEvent(event);*/
};

InteractionQC.inputTrigger = function (parameterName) {
  if (parameterName == "_run") {
    InteractionQC.run();
  }
};

InteractionQC.loadFromH5AD = async function () {
  let _dataset = Object.keys(dataUtils.data)[0];
  data_obj = dataUtils.data[_dataset];
  if (!data_obj._filetype == "h5") return;
  try {
    let obs = data_obj._gb_col.replace(/\/obs/g, "");
    let matrix = await dataUtils._hdf5Api.get(data_obj._csv_path, {
      path: "/uns/" + obs + "_nhood_enrichment/zscore",
    });
    console.log(matrix);
    let _matrix_header = Object.keys(data_obj._groupgarden);
    // convert matrix from 1D typed array of shape NxN to array of arrays

    InteractionQC._matrix = [];
    for (let i = 0; i < _matrix_header.length; i += 1) {
      InteractionQC._matrix.push(
        matrix.value.slice(
          i * _matrix_header.length,
          (i + 1) * _matrix_header.length,
        ),
      );
    }
    InteractionQC._matrix = InteractionQC._matrix.reverse();
    console.log(InteractionQC._matrix);
    return _matrix_header;
  } catch (error) {
    interfaceUtils.alert(
      "No precomputed matrix found for the current dataset.",
    );
    return null;
  }
};

InteractionQC.run = async function () {
  let _dataset = Object.keys(dataUtils.data)[0];
  if (!_dataset) {
    interfaceUtils.alert("No marker dataset loaded");
    return;
  }
  let _matrix_header = InteractionQC._matrix_header;
  if (!InteractionQC._matrix_header) {
    _matrix_header = await InteractionQC.loadFromH5AD();
  }
  var op = tmapp["object_prefix"];

  var data = [
    {
      z: InteractionQC._matrix,
      x: _matrix_header,
      y: _matrix_header.slice().reverse(),
      type: "heatmap",
      hoverongaps: false,
      colorscale: "Viridis",
    },
  ];

  var layout = {
    autosize: true,
    automargin: true,
    showlegend: true,
    yaxis: {
      side: "top",
      tickmode: "array",
      tickvals: _matrix_header,
      ticktext: _matrix_header.map(function (text) {
        let color = document.getElementById(
          _dataset + "_" + text + "_color",
        )?.value;
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
      tickvals: _matrix_header.slice().reverse(),
      ticktext: _matrix_header
        .slice()
        .reverse()
        .map(function (text) {
          let color = document.getElementById(
            _dataset + "_" + text + "_color",
          )?.value;
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

  let legend = "";
  for (type of _matrix_header) {
    let typecolor = document.getElementById(
      _dataset + "_" + type + "_color",
    )?.value;
    legend +=
      "<div style='display:inline-block;margin-right:10px;'><span style='width:15px;color:" +
      typecolor +
      "'>█</span><span style='min-width:150px;margin: 0px 5px;'>" +
      type +
      "</span></div>";
  }
  document.getElementById("InteractionQC_legend").innerHTML = legend;

  document
    .getElementById("InteractionQC_matrix")
    .on("plotly_click", function (data) {
      console.log(data.points[0].x, data.points[0].y);
      var clicked_x = data.points[0].x.toString().replace(/ /g, "_");
      var clicked_y = data.points[0].y.toString().replace(/ /g, "_");
      var uid = _dataset;
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
