/**
 * @file Spot_Inspector.js
 * @author Christophe Avenel, Axel Andersson
 */

/**
 * @namespace Spot_Inspector
 * @classdesc The root namespace for Spot_Inspector.
 */

let cmap = ["None"].concat(
  dataUtils._d3LUTs.map(function (str, index) {
    return { value: index, innerHTML: str.replace("interpolate", "") };
  }),
);

var Spot_Inspector;
Spot_Inspector = {
  name: "Spot Inspector Plugin",
  _bboxSize: 11,
  _figureSize: 7,
  _layer_format: null,
  _marker_row: "channels",
  _marker_col: "rounds",
  _cmap: "33",
  parameters: {
    _ImageSection: {
      label: "Image layer options",
      title: "IMAGE LAYER OPTIONS",
      type: "section",
    },
    _layer_format: {
      label:
        "Layer name format - use <b>{row}</b> and <b>{col}</b> to define dimensions:",
      type: "text",
      default: "",
    },
    _cmap: {
      label: "Select colormap",
      type: "select",
      default: 33,
    },
    _gamma: {
      label: "Use gamma transform on intensities",
      type: "number",
      default: 1,
      attributes: {
        min: 0,
        max: 5,
        step: 0.1,
      },
    },
    _MarkersSection: {
      label:
        'If you want to visualize markers on top of the image, you need to have a csv column for successive {row} values separated by semi-colons, and a csv column for successive {col} values separated by semi-colons (e.g. "1;2;3;4" and "C;T;G;A" for layers in the format Round1_C.tif',
      title: "MARKER OPTIONS",
      type: "section",
    },
    _marker_row: {
      label: "Select {row} column of markers",
      type: "select",
      default: "rounds",
    },
    _marker_col: {
      label: "Select {col} column of markers",
      type: "select",
      default: "channels",
    },
    _layername: {
      label: "Or select column with corresponding layer name",
      type: "select",
      default: "",
    },
    _AdvancedSection: {
      label: "Only use these settings if you know what you are doing!",
      title: "ADVANCED SETTINGS",
      type: "section",
    },
    _importImages: {
      label: "Import folder of images into layers <i>(optional)</i>",
      type: "button",
    },
    _max_width: {
      label: "Maximum width of frame, in pixels",
      type: "number",
      default: 800,
    },
    _line_width: {
      label: "Line width",
      type: "number",
      default: 5,
    },
  },
};

// Log Scale
const expScale = d3.scalePow().exponent(Math.E).domain([0, 1]);
const colorScaleExp = d3.scaleSequential((d) =>
  d3.interpolateGreys(expScale(1 - d)),
);

d3["LogGreys"] = colorScaleExp;
dataUtils._d3LUTs.push("LogGreys");

Spot_Inspector.inputTrigger = function (parameterName) {
  if (parameterName == "_layer_format") {
    $(".Spot_Inspector_overlay").remove();
    $("#ISS_Spot_Inspector_viewer").remove();
    Spot_Inspector.getMatrix();
  }
  if (parameterName == "_gamma") {
    Spot_Inspector.setFilters();
  } else if (parameterName == "_cmap") {
    Spot_Inspector.setFilters();
  } else if (parameterName == "_importImages") {
    $(".Spot_Inspector_overlay").remove();
    $("#ISS_Spot_Inspector_viewer").remove();
    interfaceUtils
      .prompt(
        "<i>This will replace all layers of the current project.</i><br/>Give the path format of your images, use * for numbers:",
        "R*_C*.tif",
        "Import images into layers",
      )
      .then((pathFormat) => {
        Spot_Inspector.loadImages(pathFormat);
      });
  }
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
  if (Spot_Inspector.get("_layer_format") == "")
    Spot_Inspector.updateLayerFormat(false);

  cmap = ["None"].concat(
    dataUtils._d3LUTs.map(function (str, index) {
      return { value: index, innerHTML: str.replace("interpolate", "") };
    }),
  );
  interfaceUtils.addObjectsToSelect(Spot_Inspector.getInputID("_cmap"), cmap);
  Spot_Inspector.set("_cmap", Spot_Inspector.get("_cmap"));

  if (Object.keys(dataUtils.data).length > 0) {
    let marker_row = Spot_Inspector.getInputID("_marker_row");
    interfaceUtils.cleanSelect(marker_row);
    interfaceUtils.addElementsToSelect(
      marker_row,
      [0].concat(Object.values(dataUtils.data)[0]._csv_header),
    );

    let marker_col = Spot_Inspector.getInputID("_marker_col");
    interfaceUtils.cleanSelect(marker_col);
    interfaceUtils.addElementsToSelect(
      marker_col,
      [0].concat(Object.values(dataUtils.data)[0]._csv_header),
    );

    let layername = Spot_Inspector.getInputID("_layername");
    interfaceUtils.cleanSelect(layername);
    interfaceUtils.addElementsToSelect(
      layername,
      [null].concat(Object.values(dataUtils.data)[0]._csv_header),
    );

    if (
      Object.values(dataUtils.data)[0]._csv_header.indexOf(
        Spot_Inspector.get("_layername"),
      ) > 0
    ) {
      Spot_Inspector.set("_layername", Spot_Inspector.get("_layername"));
    }
    if (
      Object.values(dataUtils.data)[0]._csv_header.indexOf(
        Spot_Inspector.get("_marker_row"),
      ) > 0
    ) {
      Spot_Inspector.set("_marker_row", Spot_Inspector.get("_marker_row"));
    }
    if (
      Object.values(dataUtils.data)[0]._csv_header.indexOf(
        Spot_Inspector.get("_marker_col"),
      ) > 0
    ) {
      Spot_Inspector.set("_marker_col", Spot_Inspector.get("_marker_col"));
    }
  }

  let advancedSectionIndex = 9;

  let advancedSectionElement = document.querySelector(
    `#plugin-Spot_Inspector div:nth-child(${advancedSectionIndex}) div h6`,
  );
  advancedSectionElement?.setAttribute("data-bs-toggle", "collapse");
  advancedSectionElement?.setAttribute("data-bs-target", "#collapse_advanced");
  advancedSectionElement?.setAttribute("aria-expanded", "false");
  advancedSectionElement?.setAttribute("aria-controls", "collapse_advanced");
  advancedSectionElement?.setAttribute(
    "class",
    "collapse_button_transform border-bottom-0 p-1 collapsed",
  );
  advancedSectionElement?.setAttribute("style", "cursor: pointer;");
  advancedSectionElement?.setAttribute("title", "Click to expand");
  let newDiv = document.createElement("div");
  newDiv.setAttribute("id", "collapse_advanced");
  newDiv.setAttribute("class", "collapse");
  $("#plugin-Spot_Inspector").append(newDiv);
  let advancedSectionSubtitle = document.querySelector(
    `#plugin-Spot_Inspector div:nth-child(${advancedSectionIndex}) div p`,
  );
  newDiv.appendChild(advancedSectionSubtitle);
  for (
    let indexElement = advancedSectionIndex + 1;
    indexElement < Object.keys(Spot_Inspector.parameters).length + 1;
    indexElement++
  ) {
    let element = document.querySelector(
      `#plugin-Spot_Inspector div:nth-child(${advancedSectionIndex + 1})`,
    );
    newDiv.appendChild(element);
  }

  //Spot_Inspector.run();
  setTimeout(() => Spot_Inspector.getMatrix(), 2000);
};

Spot_Inspector.loadImages = function (pathFormat) {
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
        Spot_Inspector.getMatrix();
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
    return;
    if (event.quick) {
      /*var OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];
      var tiledImage = OSDviewer.world.getItemAt(0);
      var viewportCoords = OSDviewer.viewport.pointFromPixel(event.position);
      var normCoords = tiledImage.viewportToImageCoordinates(viewportCoords);

      var bbox = [
        Math.round(normCoords.x - Spot_Inspector._bboxSize / 2),
        Math.round(normCoords.y - Spot_Inspector._bboxSize / 2),
        Spot_Inspector._bboxSize,
        Spot_Inspector._bboxSize,
      ];*/
      var markers = Spot_Inspector.getMarkers(bbox);
      if (markers.length > 0) {
        if (!Spot_Inspector._layer_format)
          Spot_Inspector.updateLayerFormat(false);
      }
      let img = document.getElementById("ISS_Spot_Inspector_img");
      Spot_Inspector.getMatrix();

      /*color = "red";
      var boundBoxOverlay = document.getElementById("overlay-Spot_Inspector");
      if (boundBoxOverlay) {
        OSDviewer.removeOverlay(boundBoxOverlay);
      }
      var boundBoxRect = tiledImage.imageToViewportRectangle(
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3]
      );
      boundBoxOverlay = $('<div id="overlay-Spot_Inspector"></div>');
      boundBoxOverlay.css({
        border: "2px solid " + color,
      });
      OSDviewer.addOverlay(boundBoxOverlay.get(0), boundBoxRect);*/
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
  function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); // $& means the whole matched string
  }
  if (doPrompt == undefined) doPrompt = true;
  rounds = [];
  channels = [];

  var difference = patienceDiff(
    escapeRegExp(tmapp.layers[0].name).split(""),
    escapeRegExp(tmapp.layers[tmapp.layers.length - 1].name).split(""),
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
  Spot_Inspector.set("_layer_format", format);
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

Spot_Inspector.getMatrix = function () {
  var op = tmapp["object_prefix"];
  var vname = op + "_viewer";
  let img = document.getElementById("ISS_Spot_Inspector_viewer");
  let getCoord = Spot_Inspector.getCoordinates();
  let layers = getCoord[0];
  console.log("LAYERS:", layers);
  if (!img) {
    img = document.createElement("div");
    img.id = "ISS_Spot_Inspector_viewer";
    img.style.width = "50px";
    img.style.height = "50px";

    var elt = document.createElement("div");
    elt.classList.add("viewer-layer");
    //elt.classList.add("px-1");
    //elt.classList.add("mx-1");
    elt.style.overflow = "auto";
    elt.style.background = "#000000";
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
    eltClose.style.right = "5px";
    eltClose.style.top = "5px";
    eltClose.innerHTML = "<i class='bi bi-x-lg'></i>";
    eltClose.addEventListener("click", function (event) {
      img.parentElement.remove();
    });
    elt.appendChild(eltClose);

    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    const path = urlParams.get("path");
    var _url_suffix = "";
    if (path != null) {
      _url_suffix = path + "/";
    }

    let options_osd = {
      id: "ISS_Spot_Inspector_viewer",
      showNavigator: false,
      animationTime: 0.0,
      blendTime: 0,
      minZoomImageRatio: 1,
      maxZoomPixelRatio: 30,
      immediateRender: true,
      showNavigationControl: false,
      imageLoaderLimit: 50,
      preload: false,
      imageSmoothingEnabled: false,
      mouseNavEnabled: false,
      preserveImageSizeOnResize: true,
    };
    Spot_Inspector.osd_viewer = OpenSeadragon(options_osd);
    Spot_Inspector.svgOverlay = Spot_Inspector.osd_viewer.svgOverlay();
    Spot_Inspector.d3Node = d3.select(Spot_Inspector.svgOverlay.node());
    Spot_Inspector.pathNode = Spot_Inspector.d3Node.append("g");
    for (let layer of layers) {
      Spot_Inspector.osd_viewer.addTiledImage({
        tileSource: _url_suffix + layer.tileSource,
        x: 0,
        y: 0,
        opacity: 0,
      });
    }
    Spot_Inspector.setFilters();
    new OpenSeadragon.MouseTracker({
      element: tmapp.ISS_viewer.canvas,
      moveHandler: (event) => Spot_Inspector.moveHandler(event),
      scrollHandler: (event) => Spot_Inspector.moveHandler(event),
    }).setTracking(true);
    setTimeout(document.getElementById("ISS_viewer").click(), 2000);
    setTimeout(document.getElementById("ISS_viewer").click(), 3000);

    tmapp.ISS_viewer.addHandler("animation", function animationHandler(event) {
      Spot_Inspector.moveHandler(event);
    });
    Spot_Inspector.osd_viewer.addHandler(
      "resize",
      function animationFinishHandler(event) {
        Spot_Inspector.moveHandler(event);
      },
    );
    Spot_Inspector.animationTimeout = null;
    Spot_Inspector.osd_viewer.addHandler(
      "animation-finish",
      function animationFinishHandler(event) {
        Spot_Inspector.animationTimeout = null;
        var count = Spot_Inspector.osd_viewer.world.getItemCount();
        for (var i = 0; i < count; i++) {
          var tiledImage = Spot_Inspector.osd_viewer.world.getItemAt(i);
          tiledImage.immediateRender = true;
        }
        Spot_Inspector.osd_viewer.imageLoaderLimit = 0;
      },
    );
    Spot_Inspector.osd_viewer.addHandler(
      "animation-start",
      function animationStartHandler(event) {
        if (Spot_Inspector.animationTimeout == null) {
          Spot_Inspector.animationTimeout = setTimeout(function () {
            if (Spot_Inspector.animationTimeout == null) return;
            var count = Spot_Inspector.osd_viewer.world.getItemCount();
            for (var i = 0; i < count; i++) {
              var tiledImage = Spot_Inspector.osd_viewer.world.getItemAt(i);
              tiledImage.immediateRender = false;
            }
            Spot_Inspector.osd_viewer.imageLoaderLimit = 2;
          }, 200);
        }
      },
    );
  }
};

Caman.Filter.register("log", function (channelValue) {
  this.process("log", function (rgba) {
    //console.log(rgba.r, rgba.g, rgba.b)
    let c = 255; // / (Math.log(1+255))
    rgba.r = c * Math.exp(1 - rgba.r / 255);
    rgba.g = c * Math.exp(1 - rgba.g / 255);
    rgba.b = c * Math.exp(1 - rgba.b / 255);
    //console.log(rgba.r, rgba.g, rgba.b)

    // Return the modified RGB values
    return rgba;
  });
});

Spot_Inspector.setFilters = function () {
  if (!filterUtils._filters["Log"]) {
    filterUtils._filters["Log"] = {
      params: {
        type: "checkbox",
      },
      filterFunction: function (value) {
        if (value == false) {
          return function (context, callback) {
            callback();
          };
        } else {
          return function (context, callback) {
            Caman(context.canvas, function () {
              this.log(value);
              this.render(callback);
            });
          };
        }
      },
    };
  }

  if (false) {
    Spot_Inspector.osd_viewer.setFilterOptions({
      filters: [],
      loadMode: "async",
    });
    return;
  }
  Spot_Inspector.waitLayersReady().then(() => {
    console.log("Add Items!");
    filters = [];
    for (const layer in filterUtils._filterItems) {
      processors = [];
      if (Spot_Inspector.get("_gamma") != 1) {
        processors.push(
          filterUtils._filters["Gamma"]["filterFunction"](
            Spot_Inspector.get("_gamma"),
          ),
        );
      }
      if (Spot_Inspector.get("_cmap") != "undefined") {
        processors.push(
          filterUtils._filters["Colormap"]["filterFunction"](
            parseInt(Spot_Inspector.get("_cmap")) + 1,
          ),
        );
      }
      for (
        var filterIndex = 0;
        filterIndex < filterUtils._filterItems[layer].length;
        filterIndex++
      ) {
        if (filterUtils._filterItems[layer][filterIndex].name != "Color") {
          processors.push(
            filterUtils._filterItems[layer][filterIndex].filterFunction(
              filterUtils._filterItems[layer][filterIndex].value,
            ),
          );
        }
      }
      filters.push({
        items: Spot_Inspector.osd_viewer.world.getItemAt(layer),
        processors: processors,
        toReset: true,
      });
    }
    Spot_Inspector.osd_viewer.setFilterOptions({
      filters: filters,
      loadMode: "async",
    });
    for (var i = 0; i < Spot_Inspector.osd_viewer.world._items.length; i++) {
      Spot_Inspector.osd_viewer.world._items[i].tilesMatrix = {};
      Spot_Inspector.osd_viewer.world._items[i]._needsDraw = true;
    }
  });
};

Spot_Inspector.waitLayersReady = async function () {
  await new Promise((r) => setTimeout(r, 200));
  while (
    !(
      !Spot_Inspector.osd_viewer.world ||
      !Spot_Inspector.osd_viewer.world.getItemCount() != tmapp.layers.length
    )
  ) {
    await new Promise((r) => setTimeout(r, 200));
  }
};

Spot_Inspector.getCoordinates = function () {
  let layers = tmapp.layers,
    layer_format = Spot_Inspector.get("_layer_format");

  const invert_row_col =
    layer_format.indexOf("{row}") > layer_format.indexOf("{col}");
  const regexp_format = new RegExp(
    layer_format.replace(/\{row\}|\{col\}/g, "(.*)"),
  );

  const outputFields = [];
  const kept_layers = [];
  for (const layer of layers) {
    const tileCoord = layer["name"].match(regexp_format);
    if (tileCoord === null) {
      continue;
    }
    kept_layers.push(layer);
    let tileCoordList;
    if (invert_row_col) {
      tileCoordList = Array.from(tileCoord).slice(1);
    } else {
      tileCoordList = Array.from(tileCoord).slice(1).reverse(); // Remove the full match (index 0)
    }
    outputFields.push(tileCoordList);
  }
  let rowNames = Array.from(new Set(outputFields.map((fields) => fields[0]))); //.sort();
  let colNames = Array.from(new Set(outputFields.map((fields) => fields[1]))); //.sort();
  console.log(regexp_format, rowNames, colNames, outputFields);
  outputFields.map((fields) => {
    fields[0] = rowNames.indexOf(fields[0]);
    fields[1] = colNames.indexOf(fields[1]);
  });
  // We check if two coordinates are similar, and add one if it the case:
  for (let i = 0; i < outputFields.length; i++) {
    for (let j = i + 1; j < outputFields.length; j++) {
      if (
        outputFields[i][0] == outputFields[j][0] &&
        outputFields[i][1] == outputFields[j][1]
      ) {
        outputFields[j][1] += 1;
      }
    }
  }

  console.log(outputFields);
  layers = kept_layers;
  return [kept_layers, outputFields];
};

Spot_Inspector.moveHandler = function (event) {
  let img = document.getElementById("ISS_Spot_Inspector_viewer");
  if (!img) return;
  if (Spot_Inspector.osd_viewer.world.getItemCount() == 0) return;
  if (!event.position) event.position = Spot_Inspector.lastEventPosition;
  Spot_Inspector.lastEventPosition = event.position;
  let getCoord = Spot_Inspector.getCoordinates();
  let layers = getCoord[0];
  let layerCoordinates = getCoord[1];
  var patch_width = 200;
  var font_height = 23;
  var margin = 5;
  var zoom;
  if (false) {
    zoom = 1;
  } else {
    zoom = tmapp.ISS_viewer.world
      .getItemAt(0)
      .viewportToImageZoom(tmapp.ISS_viewer.viewport.getZoom());
  }
  patch_width /= zoom;
  font_height /= zoom;
  margin /= zoom;
  if (event.position) {
    Spot_Inspector.position = event.position;
  } else if (!Spot_Inspector.position) {
    return;
  }
  var normCoords = tmapp.ISS_viewer.viewport.pointFromPixel(
    Spot_Inspector.position,
  );
  var imagePoint = tmapp.ISS_viewer.world
    .getItemAt(0)
    .viewportToImageCoordinates(normCoords);

  //var targetZoom = tmapp.ISS_viewer.world.getItemAt(0).source.dimensions.x / Spot_Inspector.osd_viewer.viewport.getContainerSize().x;
  //Spot_Inspector.osd_viewer.viewport.panTo(normCoords, true);
  //Spot_Inspector.osd_viewer.viewport.zoomTo(targetZoom/2, normCoords, true);
  let x_point_offset_max = 0;
  let y_point_offset_max = 0;
  for (let layer in layers) {
    let layerCoordinate = layerCoordinates[layer];

    layer = parseInt(layer);
    if (Spot_Inspector.osd_viewer.world.getItemCount() - 1 < layer) break;

    let overlay = Spot_Inspector.osd_viewer.getOverlayById(
      "Spot_Inspector_overlay_" + layer,
    );
    if (!overlay) {
      var elt = document.createElement("div");
      elt.id = "Spot_Inspector_overlay_" + layer;
      elt.className = "Spot_Inspector_overlay";
      elt.innerText = layers[layer].name.substring(0, 20);
      elt.style.fontSize = "12px";
      elt.style.color = "#FFFFFF";
      document.body.appendChild(elt);
      position = new OpenSeadragon.Point(0, 0);
      Spot_Inspector.osd_viewer.addOverlay(
        elt.id,
        position,
        OpenSeadragon.Placement.TOP,
      );
      overlay = Spot_Inspector.osd_viewer.getOverlayById(
        "Spot_Inspector_overlay_" + layer,
      );
    }

    var tiledImage = Spot_Inspector.osd_viewer.world.getItemAt(layer); // Assuming you just have a single image in the viewer
    tiledImage.setClip(
      new OpenSeadragon.Rect(
        imagePoint.x - patch_width / 2,
        imagePoint.y - patch_width / 2,
        patch_width,
        patch_width,
      ),
    );
    let x_point_offset = layerCoordinate[0] * (patch_width + margin);
    x_point_offset_max = Math.max(
      x_point_offset_max,
      x_point_offset + patch_width,
    );
    let y_point_offset = layerCoordinate[1] * (patch_width + font_height);
    y_point_offset_max = Math.max(
      y_point_offset_max,
      y_point_offset + patch_width,
    );
    var point = new OpenSeadragon.Point(x_point_offset, y_point_offset);
    tiledImage.setPosition(
      Spot_Inspector.osd_viewer.world
        .getItemAt(0)
        .imageToViewportCoordinates(point),
      true,
    );

    let label_position = new OpenSeadragon.Rect(
      imagePoint.x - patch_width / 2 + x_point_offset,
      imagePoint.y - patch_width / 2 + y_point_offset - font_height,
      patch_width,
      font_height,
    );
    overlay.update(
      Spot_Inspector.osd_viewer.world
        .getItemAt(0)
        .imageToViewportRectangle(label_position),
      OpenSeadragon.Placement.TOP_LEFT,
    );
    tiledImage.setOpacity(100);
  }
  Spot_Inspector.osd_viewer.viewport.fitBounds(
    Spot_Inspector.osd_viewer.world
      .getItemAt(0)
      .imageToViewportRectangle(
        new OpenSeadragon.Rect(
          imagePoint.x - patch_width / 2,
          imagePoint.y - patch_width / 2 - font_height,
          x_point_offset_max,
          y_point_offset_max + font_height,
        ),
      ),
    true,
  );

  // Change window size:
  let ratio = x_point_offset_max / (y_point_offset_max + font_height);
  if (ratio > 1) {
    let max_width = Math.min(
      Spot_Inspector.get("_max_width"),
      document.getElementById("ISS_viewer")?.offsetWidth || 0,
    );
    document.getElementById("ISS_Spot_Inspector_viewer").style.width =
      max_width + "px";
    document.getElementById("ISS_Spot_Inspector_viewer").style.height =
      max_width / ratio + "px";
  } else {
    let max_height = Math.min(
      Spot_Inspector.get("_max_width"),
      document.getElementById("ISS_viewer")?.offsetHeight || 0,
    );
    document.getElementById("ISS_Spot_Inspector_viewer").style.width =
      max_height * ratio + "px";
    document.getElementById("ISS_Spot_Inspector_viewer").style.height =
      max_height + "px";
  }

  let overlay = Spot_Inspector.osd_viewer.getOverlayById(
    "Spot_Inspector_overlay_0",
  );
  let real_font_size = overlay.size.y / 1.3;
  for (let layer in layers) {
    let overlay = Spot_Inspector.osd_viewer.getOverlayById(
      "Spot_Inspector_overlay_" + layer,
    );
    if (overlay) overlay.style.fontSize = real_font_size + "px";
  }

  var OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];
  var tiledImage = OSDviewer.world.getItemAt(0);
  var viewportCoords = OSDviewer.viewport.pointFromPixel(event.position);
  var normCoords = tiledImage.viewportToImageCoordinates(viewportCoords);
  console.log(patch_width, Spot_Inspector._bboxSize);
  let _bboxSize = Math.min(25, patch_width);
  var bbox = [
    Math.round(normCoords.x - _bboxSize / 2),
    Math.round(normCoords.y - _bboxSize / 2),
    _bboxSize,
    _bboxSize,
  ];
  let markers = Spot_Inspector.getMarkers(bbox),
    marker_row = Spot_Inspector._marker_row,
    marker_col = Spot_Inspector._marker_col;

  var canvas = Spot_Inspector.pathNode.node();
  let regionobj = d3.select(canvas);
  regionobj.selectAll("polyline").remove();
  for (let layer in layers) {
    let layerCoordinate = layerCoordinates[layer];
    let x_point_offset = layerCoordinate[0] * (patch_width + margin);
    let y_point_offset = layerCoordinate[1] * (patch_width + font_height);
    for (let edges of [
      [-1, 0, 1, 0],
      [0, -1, 0, 1],
    ]) {
      let p1 = Spot_Inspector.osd_viewer.world
          .getItemAt(0)
          .imageToViewportCoordinates(
            normCoords.x +
              x_point_offset +
              edges[0] *
                0.005 *
                patch_width *
                Spot_Inspector.get("_line_width"),
            normCoords.y +
              y_point_offset +
              edges[1] *
                0.005 *
                patch_width *
                Spot_Inspector.get("_line_width"),
          ),
        p2 = Spot_Inspector.osd_viewer.world
          .getItemAt(0)
          .imageToViewportCoordinates(
            normCoords.x +
              x_point_offset +
              edges[2] *
                0.005 *
                patch_width *
                Spot_Inspector.get("_line_width"),
            normCoords.y +
              y_point_offset +
              edges[3] *
                0.005 *
                patch_width *
                Spot_Inspector.get("_line_width"),
          );
      let strokeWstr =
        (Spot_Inspector.get("_line_width") * 0.0005) /
        Spot_Inspector.osd_viewer.viewport.getZoom();
      var polyline = regionobj
        .append("polyline")
        .attr("points", [
          [p1.x, p1.y],
          [p2.x, p2.y],
        ])
        .style("fill", "none")
        .attr("stroke-width", strokeWstr)
        .attr("stroke", "#FFFFFF");
    }
  }
  if (!Spot_Inspector.get("_marker_col")) return;

  for (let marker of markers) {
    let marker_col = marker[Spot_Inspector.get("_marker_col")]
      ? marker[Spot_Inspector.get("_marker_col")]
      : 0;
    let marker_row = marker[Spot_Inspector.get("_marker_row")]
      ? marker[Spot_Inspector.get("_marker_row")]
      : 0;
    let layername = marker[Spot_Inspector.get("_layername")]
      ? marker[Spot_Inspector.get("_layername")]
      : null;
    let x = marker["global_X_pos"],
      y = marker["global_Y_pos"];
    // Compute the opacity so that it is 1 in the center of patch_width and 0 at the border
    let opacity =
      1 -
      1.5 *
        Math.max(
          0,
          Math.min(
            1,
            (Math.abs(x - normCoords.x) + Math.abs(y - normCoords.y)) /
              patch_width,
          ),
        );
    if (
      marker_col.toString().indexOf(";") > -1 &&
      marker_row.toString().indexOf(";") > -1
    ) {
      let channels = marker_col.split(";"),
        rounds = marker_row.split(";");
      let pathPoints = [];
      for (let i = 0; i < channels.length; i++) {
        let channel = parseInt(channels[i]),
          round = parseInt(rounds[i]);
        let point_x = x + channel * (patch_width + margin),
          point_y = y + round * (patch_width + font_height);
        let viewportPoint = Spot_Inspector.osd_viewer.world
          .getItemAt(0)
          .imageToViewportCoordinates(point_x, point_y);
        pathPoints.push([viewportPoint.x, viewportPoint.y]);
      }
      //var OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];
      //var normCoords = OSDviewer.viewport.pointFromPixel(event.position);
      let idregion = 0;

      let strokeWstr =
        (Spot_Inspector.get("_line_width") * 0.001) /
        Spot_Inspector.osd_viewer.viewport.getZoom();
      var polyline = regionobj
        .append("polyline")
        .attr("points", pathPoints)
        .style("fill", "none")
        .attr("stroke-width", strokeWstr)
        .attr("stroke", marker["color"])
        .attr("class", "region" + marker[""])
        .attr("opacity", opacity);
    } else if (layername) {
      console.log("Layer names: ", layername, layers);
      let layerIndex = layers
        .map((x) => {
          return x.name;
        })
        .indexOf(layername + ".dzi");
      let coord = layerCoordinates[layerIndex];
      console.log("Coordinates: ", coord);
      let channel = coord[0],
        round = coord[1];
      console.log(channel, round);
      let point_x = channel ? x + channel * (patch_width + margin) : x,
        point_y = round ? y + round * (patch_width + font_height) : y;
      for (let edges of [
        [-1, -1, 1, 1],
        [-1, 1, 1, -1],
      ]) {
        let p1 = Spot_Inspector.osd_viewer.world
            .getItemAt(0)
            .imageToViewportCoordinates(
              point_x +
                edges[0] *
                  0.01 *
                  patch_width *
                  Spot_Inspector.get("_line_width"),
              point_y +
                edges[1] *
                  0.01 *
                  patch_width *
                  Spot_Inspector.get("_line_width"),
            ),
          p2 = Spot_Inspector.osd_viewer.world
            .getItemAt(0)
            .imageToViewportCoordinates(
              point_x +
                edges[2] *
                  0.01 *
                  patch_width *
                  Spot_Inspector.get("_line_width"),
              point_y +
                edges[3] *
                  0.01 *
                  patch_width *
                  Spot_Inspector.get("_line_width"),
            );
        let strokeWstr =
          (Spot_Inspector.get("_line_width") * 0.001) /
          Spot_Inspector.osd_viewer.viewport.getZoom();
        var polyline = regionobj
          .append("polyline")
          .attr("points", [
            [p1.x, p1.y],
            [p2.x, p2.y],
          ])
          .style("fill", "none")
          .attr("stroke-width", strokeWstr)
          .attr("stroke", marker["color"])
          .attr("class", "region" + marker[""])
          .attr("opacity", opacity);
      }
    }
  }
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
