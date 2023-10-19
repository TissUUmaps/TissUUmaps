/**
 * @file DEPICTER.js
 * @author Christophe Avenel
 */

/**
 * @namespace DEPICTER
 * @classdesc The root namespace for DEPICTER.
 */
var DEPICTER;
var url = window.location.pathname;
var filename = url.substring(url.lastIndexOf("/") + 1).replace(".tmap", "");

DEPICTER = {
  name: "DEPICTER",
  _numberOfClasses: 0,
  _seeds: {},
  _classes: undefined,
  _history: [],
  parameters: {
    _refresh: {
      label: "Refresh plugin",
      type: "button",
    },
    _info: {
      label: "Hold shift to draw a region around markers",
      type: "label",
    },
    _run_sic: {
      label: "Run Seeded Iterative Clustering",
      type: "button",
    },
    /*"_run_crf":{
       "label":"Run CRF Postprocessing",
       "type":"button"
     },*/
    _undo: {
      label: "Undo",
      type: "button",
    },
    _fs_annotation: {
      label: "Feature space annotation",
      type: "button",
    },
    _downloadCSV: {
      label: "Download data as CSV",
      type: "button",
    },
    _opacity: {
      label: "Marker opacity",
      type: "number",
      default: 0.5,
      attributes: { step: 0.1, min: 0, max: 1 },
    },
  },
  _dataset: null,
  _UMAP1: "/obsm/X_umap;0",
  _UMAP2: "/obsm/X_umap;1",
  _region: null,
  _regionPixels: null,
  _regionWin: null,
  _newwin: null,
};

/**
 * This method is called when the document is loaded.
 * The container element is a div where the plugin options will be displayed.
 * @summary After setting up the tmapp object, initialize it*/
DEPICTER.init = async function (container) {
  while (Object.keys(dataUtils.data).length == 0) {
    await new Promise((r) => setTimeout(r, 2000));
  }
  DEPICTER.container = container;
  if (glUtils.temp_pick === undefined) {
    glUtils.temp_pick = glUtils.pick;
    tmapp["ISS_viewer"].removeHandler("canvas-click", glUtils.pick);
    glUtils.pick = function (event) {
      glUtils.temp_pick(event);
      if (glUtils._pickedMarker[0] != -1) {
        classSelected = document.querySelector(
          'input[name="radio_name"]:checked',
        ).value;
        if (classSelected) {
          DEPICTER.add_seed(classSelected, glUtils._pickedMarker[1]);
          let newData = { seeds: DEPICTER._seeds, classes: DEPICTER._classes };
          DEPICTER._history.push(JSON.parse(JSON.stringify(newData)));
          DEPICTER.updateClasses(newData);
        }
        glUtils._pickedMarker[0] = -1;
      }
    };
    tmapp["ISS_viewer"].addHandler("canvas-click", glUtils.pick);
  }

  DEPICTER.add_row(DEPICTER.container, "#00FF00", "Negative class");
  DEPICTER.add_row(DEPICTER.container, "#FF0000", "Positive class");

  var label_row = HTMLElementUtils.createRow({});
  var label_col = HTMLElementUtils.createColumn({ width: 12 });
  var label = HTMLElementUtils.createElement({ kind: "p", id: "_message" });
  label.setAttribute("class", "badge bg-warning text-dark");
  label_row.appendChild(label_col);
  label_col.appendChild(label);
  DEPICTER.container.appendChild(label_row);
  DEPICTER.setMessage("Initializing plugin...");

  let newData = { seeds: DEPICTER._seeds, classes: DEPICTER._classes };
  DEPICTER._history.push(JSON.parse(JSON.stringify(newData)));
  DEPICTER.updateClasses(newData);

  if (DEPICTER._dataset === null) {
    DEPICTER._dataset = Object.keys(dataUtils.data)[0];
  }

  $("#" + DEPICTER._dataset + "_shape-fixed-value")[0].value = "square";
  $("#" + DEPICTER._dataset + "_shape-fixed")[0].checked = false;
  $("#" + DEPICTER._dataset + "_scale-factor").val(2.5);
  $("#" + DEPICTER._dataset + "_shape-fixed")[0].click();
  dataUtils.updateViewOptions(DEPICTER._dataset);

  // Pre-download features:
  data_obj = dataUtils.data[DEPICTER._dataset];
  await dataUtils._hdf5Api.get(data_obj._csv_path, { path: "/obsm/features" });
  let allinputs = {
    umap1: { value: DEPICTER._UMAP1 },
    umap2: { value: DEPICTER._UMAP2 },
  };
  await dataUtils.getAllH5Data(DEPICTER._dataset, allinputs);
  await dataUtils._hdf5Api.get(data_obj._csv_path, { path: DEPICTER._UMAP2 });
  DEPICTER.showFeatureSpace();
  DEPICTER.initPython();
};

DEPICTER.inputTrigger = function (parameterName) {
  if (parameterName == "_run_sic") {
    DEPICTER.runSIC();
  }
  if (parameterName == "_run_crf") {
    DEPICTER.runCRF();
  }
  if (parameterName == "_undo") {
    DEPICTER.undo();
  } else if (parameterName == "_refresh") {
    pluginUtils.startPlugin("DEPICTER");
  } else if (parameterName == "_fs_annotation") {
    DEPICTER.runFSannotation();
  } else if (parameterName == "_downloadCSV") {
    DEPICTER.downloadCSV();
  } else if (parameterName == "_opacity") {
    DEPICTER.updateClasses({
      seeds: DEPICTER._seeds,
      classes: DEPICTER._classes,
    });
  }
};

DEPICTER.runFSannotation = function () {
  let data_obj = dataUtils.data[DEPICTER._dataset];
  let LEN = data_obj._processeddata[data_obj._X].length;
  let arr = new Array(LEN).fill(0);
  for (s of DEPICTER._seeds[2]) {
    arr[s] = 1;
  }
  let newData = { seeds: DEPICTER._seeds, classes: arr };
  DEPICTER._history.push(JSON.parse(JSON.stringify(newData)));
  DEPICTER.updateClasses(newData);
};

DEPICTER.runCRF = function () {
  // TODO run on server?
};

DEPICTER.undo = function () {
  console.log("undo", DEPICTER._history.length, DEPICTER._history);
  if (DEPICTER._history.length < 2) {
    return;
  }
  actualState = JSON.stringify(DEPICTER._history.pop());
  let last_data = DEPICTER._history[DEPICTER._history.length - 1];
  while (JSON.stringify(last_data) == actualState) {
    if (DEPICTER._history.length < 2) {
      return;
    }
    DEPICTER._history.pop();
    last_data = DEPICTER._history[DEPICTER._history.length - 1];
  }
  last_data = DEPICTER._history[DEPICTER._history.length - 1];
  DEPICTER._seeds = JSON.parse(JSON.stringify(last_data["seeds"]));
  DEPICTER.updateClasses(last_data);
};

DEPICTER.runSIC = function () {
  var pythonScript = `
    DEPICTER.setMessage ("Run failed") # Printed only if Python fails
    from js import dataUtils
    from js import DEPICTER
    from pyodide.ffi import to_js
    import numpy as np
    import json
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score


    embeddings = np.asarray(DEPICTER._features.to_py())
    embeddings = np.reshape(embeddings, (-1, DEPICTER._num_features))
    print (embeddings.shape)
    jsonParam = {}
    jsonParam["seeds"] = DEPICTER._seeds.to_py()

    seeds_0 = []
    seeds_1 = []

    # for i in range(len(jsonParam["seeds"]['1'])):
    #  seeds_0.append(embeddings[jsonParam["seeds"]['1'][i]])
    # for i in range(len(jsonParam["seeds"]['2'])):
    #  seeds_1.append(embeddings[jsonParam["seeds"]['2'][i]])

    # seeds = np.vstack([np.mean(seeds_0,axis=0),np.mean(seeds_1,axis=0)])

    seeds = -np.ones((embeddings.shape[0],))
    seeds[jsonParam["seeds"]['1']] = 0
    seeds[jsonParam["seeds"]['2']] = 1

    print(np.unique(seeds))

    def calculate_seeds(X, sparse_annot, n_clusters):
      seed_list = []
      clusters = np.array(range(n_clusters))
      for i in clusters:
          if i in np.unique(sparse_annot):
              seed = np.mean(X[sparse_annot == i], axis=0)
              seed_list.append(seed)
          else:
              missing = np.array(range(n_clusters))[(np.unique(sparse_annot)!=-1)==True]
              X_missing = X[clusters[(np.unique(sparse_annot)!=-1)^1==True]]
              seed_list.append(X_missing[np.random.choice(X_missing.shape[0], 1, replace=False)].squeeze())
      seeds = np.array(seed_list)
      return seeds


    def sic(x,y):
        ckmeans_initial = KMeans(n_clusters=2, init=calculate_seeds(x,y,2)).fit(x)
        labels = ckmeans_initial.labels_
        score_prev = 0
        while True:
            try:
                ckmeans_iter = KMeans(n_clusters=2, init=calculate_seeds(x[labels==1],y[labels==1],2)).fit(x[labels==1])
                labels_temp = np.zeros(len(x))
                labels_temp[labels==1] = ckmeans_iter.labels_
            except(FloatingPointError, IndexError):
                if (y[labels==1]==1).any() and x[labels==1].shape[0]>1:
                    kmeans = KMeans(n_clusters=2).fit(x[labels==1])
                    kmeans_labels = kmeans.labels_^1 if np.mean(kmeans.labels_^1==y[labels==1])>np.mean(kmeans.labels_==y[labels==1]) else kmeans.labels_
                    labels_temp = np.zeros(len(x))
                    labels_temp[labels==1] = kmeans_labels
                break
            score = accuracy_score(y[y>-1], labels_temp[y>-1])
            if score<=score_prev:
                break
            score_prev, labels = score, labels_temp
        return labels

    # kmeans = KMeans(n_clusters=2, init=seeds).fit(embeddings)

    labels = sic(embeddings, seeds)

    jsonParam["classes"] = [int(x) for x in list(labels)]

    DEPICTER._history.append(to_js(json.dumps(jsonParam)));
    DEPICTER.updateClasses(to_js(json.dumps(jsonParam)))
  `;
  data_obj = dataUtils.data[DEPICTER._dataset];
  //DEPICTER.loadingmodal = interfaceUtils.loadingModal("Computing seeded interactive clustering...")
  DEPICTER.setMessage("Loading Embeddings...");
  dataUtils._hdf5Api
    .get(data_obj._csv_path, { path: "/obsm/features" })
    .then(function (data) {
      DEPICTER._features = data.value;
      let nb_spots = data_obj._processeddata[data_obj._X].length;
      DEPICTER._num_features = DEPICTER._features.length / nb_spots;
      DEPICTER.setMessage("Running SIC...");
      setTimeout(() => {
        DEPICTER.executePythonString(pythonScript);
      }, 10);
    });
};
/*
 DEPICTER.demo = function (message) {
   console.log(
     JSON.stringify({
       seeds: DEPICTER._seeds,
     })
   );
   var loadingmodal = interfaceUtils.loadingModal("Computing seeded interactive clustering...")
   $.ajax({
     type: "post",
     url: "/plugins/DEPICTER/server_demo",
     contentType: "application/json; charset=utf-8",
     data: JSON.stringify({
      seeds: DEPICTER._seeds,
      classes: []
     }),
     success: function (data) {
      console.log("Success:", data);
       DEPICTER.updateClasses (JSON.parse(data));
     },
     complete: function (data) {
       // do something, not critical.
       $(loadingmodal).modal('hide');
     },
     error: function (data) {
       alert("Error:", data);
     },
   });
 };
*/

DEPICTER.updateClasses = function (data) {
  if (typeof data === "string" || data instanceof String) {
    data = JSON.parse(data);
  }
  if (DEPICTER._dataset === null) {
    DEPICTER._dataset = Object.keys(dataUtils.data)[0];
    if (DEPICTER._dataset === undefined) {
      interfaceUtils.alert("Please load dataset before starting plugin.");
      return;
    }
  }
  console.log("data", data);
  DEPICTER._classes = data["classes"];
  var markerData = dataUtils.data[DEPICTER._dataset]["_processeddata"];

  var opacityPropertyName = "DEPICTER_opacity";
  var colorPropertyName = "DEPICTER_color";
  dataUtils.data[DEPICTER._dataset]["_opacity_col"] = opacityPropertyName;
  dataUtils.data[DEPICTER._dataset]["_cb_col"] = colorPropertyName;

  markerData[opacityPropertyName] = new Float64Array(
    markerData[dataUtils.data[DEPICTER._dataset]["_X"]].length,
  );
  markerData[opacityPropertyName] = markerData[opacityPropertyName].map(
    function () {
      return DEPICTER.get("_opacity");
    },
  );
  if (DEPICTER._classes !== undefined) {
    var classPropertyName = "DEPICTER_class";
    markerData[classPropertyName] = DEPICTER._classes;
    markerData[colorPropertyName] = DEPICTER._classes;

    markerData[colorPropertyName] = markerData[colorPropertyName].map(
      function (v) {
        return $("#class_id_color_" + (v - -1)).val();
      },
    );
  } else {
    markerData[colorPropertyName] = Array.apply(
      null,
      Array(markerData[dataUtils.data[DEPICTER._dataset]["_X"]].length),
    );
    markerData[colorPropertyName] = markerData[colorPropertyName].map(
      function () {
        return "#FFFFFF";
      },
    );
  }
  for (var seeds in data["seeds"]) {
    for (var seed of data["seeds"][seeds]) {
      markerData[opacityPropertyName][seed] = 0.85;
      markerData[colorPropertyName][seed] = $("#class_id_color_" + seeds).val();
    }
  }
  /*setTimeout(() => {
    $(DEPICTER.loadingmodal).modal('hide'),
    500
  })*/

  glUtils.loadMarkers(DEPICTER._dataset, true);
  glUtils.draw();
  if (DEPICTER._newwin) {
    DEPICTER._newwin.glUtils.loadMarkers(DEPICTER._dataset, true);
    DEPICTER._newwin.glUtils.draw();
  }
  DEPICTER.setMessage("");
};

DEPICTER.add_row = function (container, color, name) {
  DEPICTER._numberOfClasses += 1;
  rowx = HTMLElementUtils.createRow({});
  colx1 = HTMLElementUtils.createColumn({ width: 12 });
  inputx11 = HTMLElementUtils.inputTypeColor({
    id: "class_id_color_" + DEPICTER._numberOfClasses,
  });
  inputradio1cb = HTMLElementUtils.createElement({
    kind: "input",
    id: "class_id_radio_" + DEPICTER._numberOfClasses,
    extraAttributes: {
      name: "radio_name",
      class: "form-check-input",
      type: "radio",
      checked: true,
      value: DEPICTER._numberOfClasses,
      style: "margin-right: 5px; margin-left: 5px",
    },
  });
  labelcbgroup = HTMLElementUtils.createElement({
    kind: "label",
    extraAttributes: {
      class: "form-check-label",
      for: "class_id_radio_" + DEPICTER._numberOfClasses,
      style: "vertical-align: top;  display: inline-block;",
    },
  });
  // labelcbgroup.innerText="Class " + DEPICTER._numberOfClasses;
  labelcbgroup.innerText = name;

  container.appendChild(rowx);
  rowx.appendChild(colx1);
  colx1.appendChild(inputx11);
  colx1.appendChild(inputradio1cb);
  colx1.appendChild(labelcbgroup);
  DEPICTER._seeds[DEPICTER._numberOfClasses] = [];
  inputx11.value = color;
};

DEPICTER.downloadCSV = function () {
  var csvRows = [];
  let alldata = dataUtils.data[DEPICTER._dataset]._processeddata;
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

  DEPICTER.downloadPointsInRegionsCSV(csv);
};

DEPICTER.downloadPointsInRegionsCSV = function (data) {
  var blob = new Blob([data], { kind: "text/csv" });
  var url = window.URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.setAttribute("hidden", "");
  a.setAttribute("href", url);
  a.setAttribute("download", filename + ".csv");
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
};

/*
 * Only helper functions below
 *
 */
DEPICTER.executePythonString = function (text) {
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
  if (DEPICTER.myPyScript) {
    DEPICTER.myPyScript.remove();
  }
  // now remember the new script
  DEPICTER.myPyScript = div.firstElementChild;
  try {
    // add it to the body - this will already augment the tag in certain ways
    document.body.appendChild(DEPICTER.myPyScript);
    // execute the code / evaluate the expression
    //DEPICTER.myPyScript.evaluate();
  } catch (error) {
    console.error("Python error:");
    console.error(error);
  }
};

DEPICTER.initPython = function () {
  DEPICTER.setMessage("Loading Python interpreter...");
  if (!document.getElementById("pyScript")) {
    var link = document.createElement("link");
    link.src = "https://pyscript.net/latest/pyscript.css";
    link.id = "pyScript";
    link.rel = "stylesheet";
    document.head.appendChild(link);

    var script = document.createElement("script");
    script.src = "https://pyscript.net/latest/pyscript.js";
    script.defer = true;
    document.head.appendChild(script);

    var pyconfig = document.createElement("py-config");
    pyconfig.innerHTML = "packages=['scikit-learn']";
    document.head.appendChild(pyconfig);
  }

  DEPICTER.executePythonString(`
    from js import DEPICTER
    DEPICTER.pythonLoaded()
  `);
};

DEPICTER.pythonLoaded = function () {
  DEPICTER.setMessage("");
};

DEPICTER.setMessage = function (text) {
  document.getElementById("_message").innerText = text;
};

/*
 * Feature space stuff
 */

async function copyDataset(dataIn, dataOut) {
  var headers = interfaceUtils._mGenUIFuncs.getTabDropDowns(DEPICTER._dataset);
  dataOut["expectedHeader"] = Object.assign(
    {},
    ...Object.keys(headers).map((k) => ({ [k]: headers[k].value })),
  );
  var radios = interfaceUtils._mGenUIFuncs.getTabRadiosAndChecks(
    DEPICTER._dataset,
  );
  dataOut["expectedRadios"] = Object.assign(
    {},
    ...Object.keys(radios).map((k) => ({ [k]: radios[k].checked })),
  );
  dataOut["expectedHeader"]["X"] = DEPICTER._UMAP1;
  dataOut["expectedHeader"]["Y"] = DEPICTER._UMAP2;
  dataOut["expectedHeader"]["coord_factor"] = 1.0;
  dataOut["expectedRadios"]["collectionItem_col"] = false;
  dataOut["expectedRadios"]["collectionItem_fixed"] = true;
  for (var key of Object.keys(dataIn)) {
    if (
      [
        "_X",
        "_Y",
        "expectedHeader",
        "expectedRadios",
        "_groupgarden",
        "_coord_factor",
      ].indexOf(key) == -1
    ) {
      dataOut[key] = dataIn[key];
    } else if (key == "_X") {
      dataOut[key] = DEPICTER._UMAP1;
    } else if (key == "_Y") {
      dataOut[key] = DEPICTER._UMAP2;
    }
  }
  dataOut["_collectionItem_col"] = null;
  dataOut["_collectionItem_fixed"] = 0;
}

DEPICTER.showFeatureSpace = async function () {
  var op = tmapp["object_prefix"];
  var vname = op + "_viewer";
  var DEPICTER_Control = document.getElementById("DEPICTER_Control");
  if (DEPICTER_Control) {
    DEPICTER.clear();
  }

  DEPICTER_Control = document.createElement("iframe");
  DEPICTER_Control.id = "DEPICTER_Control";
  DEPICTER_Control.style.width = "100%";
  DEPICTER_Control.style.height = "100%";
  DEPICTER_Control.style.borderLeft = "1px solid #aaa";
  DEPICTER_Control.style.display = "inline-block";
  var elt = document.createElement("div");
  elt.style.width = "40%";
  elt.style.height = "100%";
  elt.style.display = "inline-block";
  elt.style.verticalAlign = "top";
  elt.id = "DEPICTER_MainDiv";
  elt.appendChild(DEPICTER_Control);
  document.getElementById("ISS_viewer").appendChild(elt);
  $(".openseadragon-container")[0].style.display = "inline-flex";
  $(".openseadragon-container")[0].style.width = "60%";

  DEPICTER_Control.addEventListener("load", (ev) => {
    DEPICTER_Control.classList.add("d-none");
    var timeout = setInterval(function () {
      var newwin = DEPICTER_Control.contentWindow;
      DEPICTER._newwin = newwin;
      //OSD handlers are not registered manually they have to be registered
      //using MouseTracker OSD objects
      if (newwin.tmapp.ISS_viewer) {
        clearInterval(timeout);
      } else {
        return;
      }
      DEPICTER._newwin.tmapp.ISS_viewer.canvas.style.backgroundColor =
        "#999999";
      DEPICTER._newwin.tmapp[vname].viewport.preserveImageSizeOnResize = false;
      DEPICTER._newwin.tmapp[vname].viewport.visibilityRatio = 1.0;
      new DEPICTER._newwin.OpenSeadragon.MouseTracker({
        element: DEPICTER._newwin.tmapp[vname].canvas,
        moveHandler: (event) =>
          DEPICTER.moveHandler(event, DEPICTER._newwin, window),
      }).setTracking(true);
      DEPICTER._newwin.tmapp["ISS_viewer"].addHandler(
        "canvas-press",
        (event) => {
          DEPICTER.pressHandler(event, DEPICTER._newwin, window);
        },
      );
      DEPICTER._newwin.tmapp["ISS_viewer"].addHandler(
        "canvas-release",
        (event) => {
          DEPICTER.releaseHandler(event, DEPICTER._newwin, window);
        },
      );
      DEPICTER._newwin.tmapp["ISS_viewer"].addHandler(
        "canvas-drag",
        (event) => {
          if (event.originalEvent.shiftKey) event.preventDefaultAction = true;
        },
      );
      DEPICTER._newwin.tmapp["ISS_viewer"].addHandler(
        "animation-finish",
        function animationFinishHandler(event) {
          DEPICTER._newwin.d3
            .selectAll(".region_UMAP")
            .selectAll("polyline")
            .each(function (el) {
              $(this).attr(
                "stroke-width",
                (2 * regionUtils._polygonStrokeWidth) /
                  DEPICTER._newwin.tmapp["ISS_viewer"].viewport.getZoom(),
              );
            });
          DEPICTER._newwin.d3
            .selectAll(".region_UMAP")
            .selectAll("circle")
            .each(function (el) {
              $(this).attr(
                "r",
                (10 * regionUtils._handleRadius) /
                  DEPICTER._newwin.tmapp["ISS_viewer"].viewport.getZoom(),
              );
            });
          DEPICTER._newwin.d3.selectAll(".region_UMAP").each(function (el) {
            $(this).attr(
              "stroke-width",
              (2 * regionUtils._polygonStrokeWidth) /
                DEPICTER._newwin.tmapp["ISS_viewer"].viewport.getZoom(),
            );
          });
        },
      );

      new OpenSeadragon.MouseTracker({
        element: tmapp[vname].canvas,
        moveHandler: (event) =>
          DEPICTER.moveHandler(event, window, DEPICTER._newwin),
      }).setTracking(true);

      tmapp["ISS_viewer"].addHandler("canvas-press", (event) => {
        DEPICTER.pressHandler(event, window, DEPICTER._newwin);
      });
      tmapp["ISS_viewer"].addHandler("canvas-release", (event) => {
        DEPICTER.releaseHandler(event, window, DEPICTER._newwin);
      });
      tmapp["ISS_viewer"].addHandler("canvas-drag", (event) => {
        if (event.originalEvent.shiftKey) event.preventDefaultAction = true;
      });
      tmapp["ISS_viewer"].addHandler(
        "animation-finish",
        function animationFinishHandler(event) {
          d3.selectAll(".region_UMAP")
            .selectAll("polyline")
            .each(function (el) {
              $(this).attr(
                "stroke-width",
                (2 * regionUtils._polygonStrokeWidth) /
                  tmapp["ISS_viewer"].viewport.getZoom(),
              );
            });
          d3.selectAll(".region_UMAP")
            .selectAll("circle")
            .each(function (el) {
              $(this).attr(
                "r",
                (10 * regionUtils._handleRadius) /
                  tmapp["ISS_viewer"].viewport.getZoom(),
              );
            });
          d3.selectAll(".region_UMAP").each(function (el) {
            $(this).attr(
              "stroke-width",
              (2 * regionUtils._polygonStrokeWidth) /
                tmapp["ISS_viewer"].viewport.getZoom(),
            );
          });
        },
      );

      /*newwin.projectUtils._activeState = JSON.parse(
        JSON.stringify(projectUtils._activeState)
      );*/
      newwin.projectUtils._activeState["markerFiles"] = JSON.parse(
        JSON.stringify(projectUtils._activeState["markerFiles"]),
      );
      newwin.tmapp["ISS_viewer"].close();

      newwin.filterUtils._compositeMode = filterUtils._compositeMode;
      newwin.interfaceUtils.generateDataTabUI({
        uid: DEPICTER._dataset,
        name: "UMAP",
      });
      /*try {
        newwin.interfaceUtils.generateDataTabUI({
          uid: DEPICTER._dataset,
          name: "UMAP",
        });
      } catch (error) {}*/
      newwin.dataUtils.data[DEPICTER._dataset] = {};
      copyDataset(
        dataUtils.data[DEPICTER._dataset],
        newwin.dataUtils.data[DEPICTER._dataset],
      ).then(() => {
        newwin.dataUtils.createMenuFromCSV(
          DEPICTER._dataset,
          newwin.dataUtils.data[DEPICTER._dataset]["_processeddata"].columns,
        );
        let main_button = newwin.document.getElementById("ISS_collapse_btn");
        main_button.classList.add("d-none");
        newwin.interfaceUtils.toggleRightPanel();
        newwin.document.getElementById("main-navbar").classList.add("d-none");
        newwin.document
          .getElementById("floating-navbar-toggler")
          .classList.add("d-none");
        newwin.document
          .getElementById("powered_by_tissuumaps")
          .classList.add("d-none");
        let elt = document.createElement("div");
        elt.className = "closeDEPICTER px-1 mx-1 viewer-layer";
        elt.id = "closeDEPICTER";
        elt.style.zIndex = "100";
        elt.style.cursor = "pointer";
        elt.innerHTML = "<i class='bi bi-x-lg'></i>";
        elt.addEventListener("click", function (event) {
          DEPICTER.clear();
        });
        newwin.tmapp.ISS_viewer.addControl(elt, {
          anchor: OpenSeadragon.ControlAnchor.TOP_LEFT,
        });
        newwin.tmapp.ISS_viewer.close();
        DEPICTER_Control.classList.remove("d-none");
        newwin.document
          .getElementsByClassName("navigator ")[0]
          .classList.add("d-none");
        setTimeout(function () {
          var copySettings = function () {
            setTimeout(function () {
              newwin = DEPICTER._newwin;
              copyDataset(
                dataUtils.data[DEPICTER._dataset],
                newwin.dataUtils.data[DEPICTER._dataset],
              ).then(() => {
                $(
                  "." +
                    DEPICTER._dataset +
                    "-marker-input, ." +
                    DEPICTER._dataset +
                    "-marker-hidden, ." +
                    DEPICTER._dataset +
                    "-marker-color, ." +
                    DEPICTER._dataset +
                    "-marker-shape",
                )
                  .each(function (i, elt) {
                    newwin.document.getElementById(elt.id).value = elt.value;
                    newwin.document.getElementById(elt.id).checked =
                      elt.checked;
                  })
                  .promise()
                  .done(function () {
                    newwin.glUtils.loadMarkers(DEPICTER._dataset);
                    newwin.glUtils.draw();
                  });
              });
            }, 100);
          };
          if (glUtils.temp_draw === undefined) {
            glUtils.temp_draw = glUtils.draw;
            glUtils.draw = function () {
              glUtils.temp_draw();
              copySettings();
            };
            glUtils.temp_updateColorLUTTexture = glUtils._updateColorLUTTexture;
            glUtils._updateColorLUTTexture = function (gl, uid, texture) {
              glUtils.temp_updateColorLUTTexture(gl, uid, texture);
              copySettings();
            };
            dataUtils.temp_updateViewOptions = dataUtils.updateViewOptions;
            dataUtils.updateViewOptions = function (
              data_id,
              force_reload_all,
              reloadH5,
            ) {
              newwin.tmapp["ISS_viewer"].world.removeAll();
              dataUtils.temp_updateViewOptions(
                data_id,
                force_reload_all,
                reloadH5,
              );
              copyDataset(
                dataUtils.data[DEPICTER._dataset],
                newwin.dataUtils.data[DEPICTER._dataset],
              ).then(() => {
                newwin.dataUtils.createMenuFromCSV(
                  DEPICTER._dataset,
                  newwin.dataUtils.data[DEPICTER._dataset]["_processeddata"]
                    .columns,
                );
                newwin.dataUtils.updateViewOptions(
                  data_id,
                  force_reload_all,
                  reloadH5,
                );
              });
            };
          }
          if (interfaceUtils.temp_toggleRightPanel === undefined) {
            interfaceUtils.temp_toggleRightPanel =
              interfaceUtils.toggleRightPanel;
            interfaceUtils.toggleRightPanel = function () {
              interfaceUtils.temp_toggleRightPanel();
            };
          }
          DEPICTER.updateClasses({
            seeds: DEPICTER._seeds,
            classes: DEPICTER._classes,
          });
        }, 200);
      });
    }, 200);
  });

  DEPICTER_Control.classList.add("d-none");
  DEPICTER_Control.setAttribute(
    "src",
    window.location.href.replace(/#.*$/, "") + "&tmap=null",
  );
};

DEPICTER.clear = function () {
  DEPICTER_Control = document.getElementById("DEPICTER_Control");
  DEPICTER_Control.parentNode.remove();
  $(".openseadragon-container")[0].style.display = "block";
  $(".openseadragon-container")[0].style.width = "100%";
};

DEPICTER.pressHandler = function (event, win, mainwin) {
  var OSDviewer = win.tmapp[tmapp["object_prefix"] + "_viewer"];

  if (event.originalEvent.shiftKey) {
    win.tmapp.ISS_viewer.gestureSettingsMouse.dragToPan = false;
    var normCoords = OSDviewer.viewport.pointFromPixel(event.position);
    var nextpoint = [normCoords.x, normCoords.y];
    DEPICTER._region = [normCoords];
    DEPICTER._regionPixels = [event.position];
    DEPICTER._regionWin = win;
  } else {
    win.tmapp.ISS_viewer.gestureSettingsMouse.dragToPan = true;
    DEPICTER._region == [];
  }
  return;
};

DEPICTER.releaseHandler = function (event, win, mainwin) {
  if (DEPICTER._region == []) {
    return;
  }
  if (!event.originalEvent.shiftKey) {
    return;
  }
  var OSDviewer = win.tmapp[tmapp["object_prefix"] + "_viewer"];

  var canvas =
    win.overlayUtils._d3nodes[
      win.tmapp["object_prefix"] + "_regions_svgnode"
    ].node();
  var regionobj = d3.select(canvas).append("g").attr("class", "_UMAP_region");
  var elements = win.document.getElementsByClassName("region_UMAP");
  for (var element of elements) element.parentNode.removeChild(element);
  var elements = mainwin.document.getElementsByClassName("region_UMAP");
  for (var element of elements) element.parentNode.removeChild(element);
  DEPICTER._region.push(DEPICTER._region[0]);

  regionobj
    .append("path")
    .attr("d", win.regionUtils.pointsToPath([[DEPICTER._region]]))
    .attr("id", "path_UMAP")
    .attr("class", "region_UMAP")
    .attr(
      "stroke-width",
      (2 * regionUtils._polygonStrokeWidth) /
        win.tmapp["ISS_viewer"].viewport.getZoom(),
    )
    .style("stroke", "#ff0000")
    .style("fill", "none");

  var pointsIn = DEPICTER.analyzeRegion(DEPICTER._region, win);
  classSelected = document.querySelector(
    'input[name="radio_name"]:checked',
  ).value;
  if (classSelected) {
    for (var d of pointsIn) {
      DEPICTER.add_seed(classSelected, d);
    }
  }

  let newData = { seeds: DEPICTER._seeds, classes: DEPICTER._classes };
  DEPICTER._history.push(JSON.parse(JSON.stringify(newData)));
  DEPICTER.updateClasses(newData);
  /*
  var scalePropertyName = "UMAP_Region_scale";
  win.dataUtils.data[DEPICTER._dataset]["_scale_col"] = scalePropertyName;
  dataUtils.data[DEPICTER._dataset]["_scale_col"] = scalePropertyName;
  var markerData = win.dataUtils.data[DEPICTER._dataset]["_processeddata"];
  markerData[scalePropertyName] = new Float64Array(
    markerData[win.dataUtils.data[DEPICTER._dataset]["_X"]].length
  );
  var opacityPropertyName = "UMAP_Region_opacity";
  win.dataUtils.data[DEPICTER._dataset]["_opacity_col"] =
    opacityPropertyName;
  dataUtils.data[DEPICTER._dataset]["_opacity_col"] = opacityPropertyName;
  markerData[opacityPropertyName] = new Float64Array(
    markerData[win.dataUtils.data[DEPICTER._dataset]["_X"]].length
  );
  markerData[opacityPropertyName] = markerData[opacityPropertyName].map(
    function () {
      return 0.15;
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

  DEPICTER_Control.style.height = "100%";
  */
  mainwin.glUtils.loadMarkers(DEPICTER._dataset, true);
  mainwin.glUtils.draw();
  win.glUtils.loadMarkers(DEPICTER._dataset, true);
  win.glUtils.draw();

  var regionobj = d3.select(canvas).append("g").attr("class", "_UMAP_region");
  var elements = win.document.getElementsByClassName("region_UMAP");
  for (var element of elements) element.parentNode.removeChild(element);
  var elements = mainwin.document.getElementsByClassName("region_UMAP");
  for (var element of elements) element.parentNode.removeChild(element);
  return;
};

DEPICTER.add_seed = function (classSelected, seed) {
  if (DEPICTER._seeds["1"].indexOf(seed) != -1) {
    DEPICTER._seeds["1"].splice(DEPICTER._seeds["1"].indexOf(seed), 1);
  }
  if (DEPICTER._seeds["2"].indexOf(seed) != -1) {
    DEPICTER._seeds["2"].splice(DEPICTER._seeds["2"].indexOf(seed), 1);
  }
  DEPICTER._seeds[classSelected].push(seed);
};

DEPICTER.moveHandler = function (event, win, mainwin) {
  if (event.buttons != 1 || DEPICTER._region == []) {
    //|| !event.shift) {
    //DEPICTER._region = [];
    //DEPICTER._regionPixels = [];
    //win.tmapp.ISS_viewer.setMouseNavEnabled(true);
    return;
  }
  if (win !== DEPICTER._regionWin) {
    return;
  }
  if (!event.originalEvent.shiftKey) {
    return;
  }
  var OSDviewer = win.tmapp[tmapp["object_prefix"] + "_viewer"];

  var normCoords = OSDviewer.viewport.pointFromPixel(event.position);

  var nextpoint = normCoords; //[normCoords.x, normCoords.y];
  DEPICTER._regionPixels.push(event.position);
  function distance(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }
  if (DEPICTER._regionPixels.length > 1) {
    dis = distance(
      DEPICTER._regionPixels[DEPICTER._regionPixels.length - 1],
      DEPICTER._regionPixels[DEPICTER._regionPixels.length - 2],
    );
    if (dis < 5) {
      DEPICTER._regionPixels.pop();
      return;
    }
  }
  DEPICTER._region.push(nextpoint);
  var canvas =
    win.overlayUtils._d3nodes[
      win.tmapp["object_prefix"] + "_regions_svgnode"
    ].node();
  var regionobj = d3.select(canvas).append("g").attr("class", "_UMAP_region");
  var elements = win.document.getElementsByClassName("region_UMAP");
  for (var element of elements) element.parentNode.removeChild(element);
  var elements = mainwin.document.getElementsByClassName("region_UMAP");
  for (var element of elements) element.parentNode.removeChild(element);

  var polyline = regionobj
    .append("polyline")
    .attr(
      "points",
      DEPICTER._region.map(function (x) {
        return [x.x, x.y];
      }),
    )
    .style("fill", "none")
    .attr(
      "stroke-width",
      (2 * regionUtils._polygonStrokeWidth) /
        win.tmapp["ISS_viewer"].viewport.getZoom(),
    )
    .attr("stroke", "#ff0000")
    .attr("class", "region_UMAP");
  return;
};

DEPICTER.analyzeRegion = function (points, win) {
  var pointsInside = [];
  var dataset = DEPICTER._dataset;
  var countsInsideRegion = {};
  var options = {
    globalCoords: true,
    xselector: win.dataUtils.data[dataset]["_X"],
    yselector: win.dataUtils.data[dataset]["_Y"],
    dataset: dataset,
  };
  var imageWidth = win.OSDViewerUtils.getImageWidth();
  var x0 = Math.min(
    ...points.map(function (x) {
      return x.x;
    }),
  );
  var y0 = Math.min(
    ...points.map(function (x) {
      return x.y;
    }),
  );
  var x3 = Math.max(
    ...points.map(function (x) {
      return x.x;
    }),
  );
  var y3 = Math.max(
    ...points.map(function (x) {
      return x.y;
    }),
  );
  var xselector = options.xselector;
  var yselector = options.yselector;
  var regionPath = win.document.getElementById("path_UMAP");
  var svgovname = win.tmapp["object_prefix"] + "_svgov";
  var svg = win.tmapp[svgovname]._svg;
  var tmpPoint = svg.createSVGPoint();

  var pointInBbox = [
    ...Array(
      win.dataUtils.data[dataset]["_processeddata"][xselector].length,
    ).keys(),
  ];
  var markerData = win.dataUtils.data[dataset]["_processeddata"];
  var collectionItemIndex = win.glUtils._collectionItemIndex[dataset];
  const collectionItemPropertyName =
    win.dataUtils.data[dataset]["_collectionItem_col"];
  const useCollectionItemFromMarker =
    win.dataUtils.data[dataset]["_collectionItem_col"] != null;
  const worldCount = win.tmapp["ISS_viewer"].world.getItemCount();
  for (var d of pointInBbox) {
    if (useCollectionItemFromMarker) {
      LUTindex = markerData[collectionItemPropertyName][d];
    } else {
      LUTindex = collectionItemIndex;
    }
    LUTindex = LUTindex % worldCount;
    const image = win.tmapp["ISS_viewer"].world.getItemAt(LUTindex);
    var viewportCoord = image.imageToViewportCoordinates(
      markerData[xselector][d] * win.dataUtils.data[dataset]._coord_factor,
      markerData[yselector][d] * win.dataUtils.data[dataset]._coord_factor,
    );
    if (
      viewportCoord.x < x0 ||
      viewportCoord.x > x3 ||
      viewportCoord.y < y0 ||
      viewportCoord.y > y3
    ) {
      continue;
    }
    var key;
    if (
      win.regionUtils.globalPointInPath(
        viewportCoord.x,
        viewportCoord.y,
        regionPath,
        tmpPoint,
      )
    ) {
      countsInsideRegion[key] += 1;
      pointsInside.push(d);
    }
  }
  function compare(a, b) {
    if (a.count > b.count) return -1;
    if (a.count < b.count) return 1;
    return 0;
  }
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
DEPICTER.searchTreeForPointsInBbox = function (
  quadtree,
  x0,
  y0,
  x3,
  y3,
  options,
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
