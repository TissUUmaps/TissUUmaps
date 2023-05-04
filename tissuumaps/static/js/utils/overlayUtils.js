/**
 * @file overlayUtils.js Interface to ask information to OSD
 * @author Leslie Solorzano
 * @see {@link overlayUtils}
 */

/**
 * @namespace overlayUtils
 * @property {Bool}   overlayUtils._drawRegions - If false then disable the drawing of regions
 * @property {Object} overlayUtils._d3nodes - Main group or container of all d3 svg groups of overlays corresponding to the 3 main marker data groups
 * @property {Number} overlayUtils._percentageForSubsample - Take this percentage of each barcode when downsamplir for lower resolutions
 * @property {Number}  overlayUtils._zoomForSubsample - When the zoom is bigger than this, display all the checked genes 
 */
overlayUtils = {
    _drawRegions: false,
    _d3nodes: {},
    _percentageForSubsample: 0.25,
    _zoomForSubsample:5.15,
    _layerOpacities:{},
    _linkMarkersToChannels:false,
    _collectionMode:false
}

/**
 * This method is used to add all layers from tmapp */
overlayUtils.addAllLayers = function() {
    /* For backward compatibility with tmapp.fixed_file, but converted to a layer */
    if (tmapp.fixed_file && tmapp.fixed_file != "") {
        tmapp.layers.unshift({"name":tmapp.slideFilename, "tileSource":tmapp.fixed_file})
        /*overlayUtils.addLayer(tmapp.slideFilename, tmapp._url_suffix +  tmapp.fixed_file, -1)*/
        tmapp.fixed_file = "";
    }
    tmapp.layers.forEach(function(layer, i) {
        overlayUtils.addLayer(layer, i-1);
    });
    overlayUtils.addAllLayersSettings();
    setTimeout(overlayUtils.setCollectionMode,500);
}

/**
 * This method is used to add all layer settings */
overlayUtils.addAllLayersSettings = function() {
    var settingsPanel = document.getElementById("image-overlay-panel");
    settingsPanel.innerHTML = "";
    tmapp.layers.forEach(function(layer, i) {
        overlayUtils.addLayerSettings(layer.name, layer.tileSource, i-1);
    });
    filterUtils.setRangesFromFilterItems();
    
    // Add collection mode checkbox:
    if (document.getElementById("setCollectionModeRow")) {
        document.getElementById("setCollectionModeRow").remove();
    }
    var extraAttributes = {
        class: "form-check-input",
        type: "checkbox"
    };
    if (projectUtils._activeState.collectionMode) {
        extraAttributes.checked = true;
    }
    var input11 = HTMLElementUtils.createElement({
        kind: "input",
        id: "setCollectionMode",
        extraAttributes: extraAttributes,
    });
    var label11 = HTMLElementUtils.createElement({
        kind: "label",
        extraAttributes: { for: "setCollectionMode" },
    });
    label11.innerHTML = "&nbsp;Collection mode";
    var row = HTMLElementUtils.createRow({ id: "setCollectionModeRow"});
    var col1 = HTMLElementUtils.createColumn({ width: 6 });
    col1.appendChild(input11);
    col1.appendChild(label11);
    row.appendChild(col1);
    settingsPanel.after(row);
    input11.addEventListener("change", (event) => {
        projectUtils._activeState.collectionMode = event.target.checked;
        overlayUtils.setCollectionMode();
    });
    
    // Add background color input:
    var extraAttributes = {
        "style":"width:50px;",
        "class":"form-control form-control-sm"
    };
    if (projectUtils._activeState.backgroundColor) {
        extraAttributes.value = projectUtils._activeState.backgroundColor;
    }
    var input11 = HTMLElementUtils.inputTypeColor(
        {
            "id": "setBackgroundColor", 
            extraAttributes: extraAttributes}
    );

    /*<div class="input-group" style="
    display: flex;
    width: 125px;
">
        <span class="input-group-text">
            x
        </span>
        <input type="text" class="form-control" placeholder="x">
    </div>*/
    var label11 = HTMLElementUtils.createElement({
        kind: "label",
        extraAttributes: { for: "setBackgroundColor", class:"input-group-text py-1 px-2 small" },
    });
    label11.innerHTML = "Background color";
    var col1 = HTMLElementUtils.createColumn({ width: "auto" });
    col1.classList.add("input-group");
    col1.appendChild(label11);
    col1.appendChild(input11);
    row.appendChild(col1);
    //settingsPanel.after(row);
    input11.addEventListener("input", (event) => {
        projectUtils._activeState.backgroundColor = event.target.value;
        $(".openseadragon-canvas")[0].style.backgroundColor=event.target.value;
    });
}

/**
 * Update position, scale, rotation and flip for a given layer */
overlayUtils.updateTransform = function (layerIndex) {
    var OSDviewer = tmapp[tmapp["object_prefix"] + "_viewer"];
    const layer = tmapp.layers[layerIndex];
    
    const x = layer.x || 0;
    const y = layer.y || 0;
    const scale = layer.scale || 1;
    const flip = layer.flip || false;
    const rotation = layer.rotation || 0;
  
    const tiledImage1 = OSDviewer.world.getItemAt(0);
    const tiledImage2 = OSDviewer.world.getItemAt(layerIndex);
    
    const layer0X = tiledImage1.getContentSize().x;
    const layerNX = tiledImage2.getContentSize().x;
    tiledImage2.setWidth(scale*layerNX/layer0X);
    var point = new OpenSeadragon.Point(x/layer0X, y/layer0X);
    tiledImage2.setPosition(point);
    tiledImage2.setRotation(rotation);
    tiledImage2.setFlip(flip);
    glUtils.draw();
}
  
/**
 * This method is used to add a layer */
overlayUtils.addLayerSettings = function(layerName, tileSource, layerIndex, checked) {
    var settingsPanel = document.getElementById("image-overlay-panel");
    var layerTable = document.getElementById("image-overlay-tbody");
    if (!layerTable) {
        layerTable = document.createElement("table");
        layerTable.id = "image-overlay-table";
        layerTable.className += "table table-striped"
        layerTable.style.marginBottom = "0px";
        filterHeaders = "";
        for (filterIndex = 0; filterIndex < filterUtils._filtersUsed.length; filterIndex++) {
            filterHeaders += "<th class='text-center'>" + filterUtils._filtersUsed[filterIndex] + "</th>";
        }
        layerTable.innerHTML = `<thead>
            <th class='text-center'>Name</th>
            <th class='text-center'>Visible</th>
            <th class='text-center'>Opacity</th>` +
            filterHeaders +
            "</thead><tbody id='image-overlay-tbody'></tbody>"
        settingsPanel.appendChild(layerTable);
    }
    layerTable = document.getElementById("image-overlay-tbody");
    var tr = document.createElement("tr");

    var visible = document.createElement("input");
    visible.type = "checkbox";
    if (layerIndex < 0 || checked)
        visible.checked = true; 
    visible.id = "visible-layer-" + (layerIndex + 1);
    visible.classList.add("visible-layers");
    visible.classList.add("form-check-input")
    visible.setAttribute("layer", (layerIndex + 1));
    var td_visible = document.createElement("td");
    td_visible.appendChild(visible);
    td_visible.classList.add("text-center");
    td_visible.classList.add("border-bottom-0");
    td_visible.classList.add("p-1");

    var opacity = document.createElement("input");
    opacity.classList.add("overlay-slider");
    opacity.classList.add("form-range");
    opacity.type = "range";
    opacity.setAttribute("min", "0");
    opacity.setAttribute("max", "1");
    opacity.setAttribute("step", "0.1");
    opacity.setAttribute("layer", (layerIndex + 1));
    opacity.id = "opacity-layer-" + (layerIndex + 1);
    var td_opacity = document.createElement("td");
    td_opacity.appendChild(opacity);
    td_opacity.classList.add("text-center");
    td_opacity.classList.add("border-bottom-0");
    td_opacity.classList.add("p-1");
    tileSource = tileSource.replace(/\\/g, '\\\\');
    var td_name = HTMLElementUtils.createElement(
        {kind:"td", extraAttributes:{
            "data-bs-toggle":"collapse",
            "data-bs-target":"#collapse_tranform_" + (layerIndex + 1),
            "aria-expanded":"false",
            "aria-controls":"collapse_tranform_" + (layerIndex + 1),
            "class":"collapse_button_transform collapsed"
        }});
    td_name.innerHTML = layerName;
    td_name.classList.add("border-bottom-0");
    td_name.classList.add("p-1");
    tr.appendChild(td_name);
    tr.appendChild(td_visible);
    tr.appendChild(td_opacity);

    for (filterIndex = 0; filterIndex < filterUtils._filtersUsed.length; filterIndex++) {
        filterName = filterUtils._filtersUsed[filterIndex];
        filterParams = filterUtils.getFilterParams(filterName)
        filterParams.layer = layerIndex + 1;

        filterInput = filterUtils.createHTMLFilter(filterParams);
        if (filterParams.type == "range") {
            filterInput.classList.add("overlay-slider");
            filterInput.classList.add("form-range");
        }
        else if (filterParams.type == "checkbox") {
            filterInput.classList.add("form-check-input");
        }
        else if (filterParams.type == "color") {
            filterInput.classList.add("form-range");
        }
        var td_filterInput = document.createElement("td");
        td_filterInput.classList.add("text-center");
        td_filterInput.classList.add("border-bottom-0");
        td_filterInput.classList.add("p-1");
        td_filterInput.appendChild(filterInput);

        tr.appendChild(td_filterInput);
    }

    visible.addEventListener("change", function(ev) {
        var layer = ev.srcElement.getAttribute("layer")
        var slider = document.querySelectorAll('[layer="' + layer + '"][type="range"]')[0];
        var checkbox = ev.srcElement;
        if (checkbox.checked) {
            overlayUtils._layerOpacities[layer] = slider.value;
        } else {
            overlayUtils._layerOpacities[layer] = 0;
        }
        overlayUtils.setItemOpacity(layer);
    });
    opacity.addEventListener("input", function(ev) {
        var layer = ev.srcElement.getAttribute("layer")
        var slider = ev.srcElement;
        var checkbox = document.querySelectorAll('[layer="' + layer + '"][type="checkbox"]')[0];
        if (checkbox.checked) {
            overlayUtils._layerOpacities[layer] = slider.value;
        } else {
            overlayUtils._layerOpacities[layer] = 0;
        }
        overlayUtils.setItemOpacity(layer);
    });

    // Add layer transformations:
    
    layerIndex = layerIndex + 1;
    var layer = tmapp.layers[layerIndex];

    var field_type = {
      "x":"number",
      "y":"number",
      "scale":"number",
      "rotation":"number",
      "flip":"checkbox",
    }
    var field_default = {
      "x":0,
      "y":0,
      "scale":1,
      "rotation":0,
      "flip":false,
    }
    
    var tr_transform = document.createElement("tr");
    var td_transform = document.createElement("td");
    td_transform.classList.add("p-0")
    td_transform.setAttribute("colspan","100");
    collapse_div = document.createElement("div");
    collapse_div.id = "collapse_tranform_" + layerIndex;
    collapse_div.classList.add("collapse")
    collapse_div.classList.add("container")
    collapse_div.classList.add("p-0")
    
    row = HTMLElementUtils.createRow({});
    row.classList.add("row-cols-auto")
    col1 = HTMLElementUtils.createColumn({
        "width": "auto", 
        "extraAttributes": {"data-source":tileSource, "class":"layerSettingButton"}
    });
    collapse_div.appendChild(row)
    row.appendChild(col1);
    for (var field in field_type) {
      if (field_type[field] == "number") {
        var form_class = "form-control form-text-input form-control-sm";
      }
      else {
        var form_class = "form-control form-check-input ";
      }
      var input11 = HTMLElementUtils.createElement({
        kind: "input",
        id: "layer_" + layerIndex + "_" + field,
        extraAttributes: {
          class: form_class + " me-1 small",
          type: field_type[field],
          value: layer[field] || field_default[field],
          style: "max-width:60px;",
          data_field: field,
          data_layerIndex: layerIndex,
        },
      });
      if (field_type[field] == "checkbox") {
        input11.checked = layer[field] || field_default[field];
        input11.classList.add("p-0")
      }
      input11.addEventListener("change", (event) => {
        var layerIndex = parseInt(event.target.getAttribute("data_layerIndex"));
        var field = event.target.getAttribute("data_field");
        if (field_type[field] == "checkbox") {
          tmapp.layers[layerIndex][field] = event.target.checked;
        }
        else {
          tmapp.layers[layerIndex][field] = event.target.value;
        }
        overlayUtils.updateTransform(layerIndex);
        glUtils.draw();
      });
      label12 = HTMLElementUtils.createElement({
        kind: "span",
        extraAttributes: { for: "layer_" + layerIndex + "_" + field, class:"input-group-text py-1 px-2 small"},
      });   
      label12.innerHTML = field.replace("rotation", "rot.");
    
      col11 = HTMLElementUtils.createElement({ kind: "div" });
      col11.classList.add("col");
      col11.classList.add("input-group");
      col11.classList.add("p-0");
      col11.appendChild(label12);
      if (field_type[field] == "checkbox") {
        let input11_div = HTMLElementUtils.createElement({
            kind: "div",
            extraAttributes: { class:"input-group-text bg-white"},
        });   
        input11_div.appendChild(input11);
        col11.appendChild(input11_div);
      }
      else {
        col11.appendChild(input11);
      }
      
      row.appendChild(col11);
    }
    tr_transform.appendChild(td_transform);
    td_transform.appendChild(collapse_div);
    row.classList.remove("p-2")
    row.classList.add("ms-0")
    layerTable.prepend(tr_transform);
    // Empty tr to keep stripes in table:
    var tr_empty = document.createElement("tr");
    tr_empty.classList.add("d-none");
    layerTable.prepend(tr_empty);
    layerTable.prepend(tr);

    overlayUtils.addLayerSlider();
}

/**
 * This method is used to add a layer */
 overlayUtils.addLayerSlider = function() {
    if (document.getElementById("channelRangeInput") == undefined) {
        var elt = document.createElement('div');
        elt.className = "channelRange px-1 mx-1 viewer-layer";
        elt.id = "channelRangeDiv"
        elt.style.zIndex = "100";
        var span = document.createElement('div');
        span.innerHTML = "Channel 1"
        span.id = "channelValue"
        span.style.maxWidth="200px";
        span.style.overflow="hidden";
        var channelRange = document.createElement("input");
        channelRange.classList.add("form-range");
        channelRange.type = "range";
        channelRange.style.width = "200px";
        channelRange.id = "channelRangeInput";
        elt.appendChild(span);
        elt.appendChild(channelRange);
        changeFun = function(ev) {
            var slider = $(channelRange)[0];
            channel = slider.value;
            $(".visible-layers").prop("checked",true);$(".visible-layers").click();$("#visible-layer-"+(channel- -1)).click();
            channelName = tmapp.layers[channel- -1].name
            channelId = channelName.replace(".dzi","");
            document.getElementById("channelValue").innerHTML = "Channel " + (channel - -2) + ": " + channelName;
            if (overlayUtils._linkMarkersToChannels) {
                $(".uniquetab-marker-input").prop("checked",false);
                if (document.getElementById("uniquetab_"+channelId+"_check")) {
                    $(document.getElementById("uniquetab_"+channelId+"_check")).click();
                }
                else {
                    $("#uniquetab_all_check").prop("checked",true);
                    $("#uniquetab_all_check").click();
                }
            }
        };
        channelRange.addEventListener("input", changeFun);

        var mousewheelevt = (/Firefox/i.test(navigator.userAgent)) ? "DOMMouseScroll" : "mousewheel";
        $(elt).bind(mousewheelevt, moveSlider);
        function moveSlider(e){
            var zoomLevel = parseInt($(channelRange).val()); 
            // detect positive or negative scrolling
            if ( e.originalEvent.wheelDelta < 0 ) {
                //scroll down
                $(channelRange).val(zoomLevel+1);
            } else {
                //scroll up
                $(channelRange).val(zoomLevel-1);
            }

            // trigger the change event
            changeFun(e.originalEvent);

            //prevent page fom scrolling
            return false;
        }
        tmapp['ISS_viewer'].addControl(elt,{anchor: OpenSeadragon.ControlAnchor.BOTTOM_LEFT});
    }
    channelRange = document.getElementById("channelRangeInput");
    var op = tmapp["object_prefix"];
    var nLayers = tmapp.layers.length;
    if (tmapp.fixed_file && tmapp.fixed_file != "") {
        nLayers += 1
    }
    channelRange.setAttribute("min", -1);
    channelRange.setAttribute("max", nLayers - 2);
    channelRange.setAttribute("step", "1");
    channelRange.setAttribute("value", "-1");
    if (nLayers <= 1) {
        document.getElementById("channelRangeDiv").style.display = "none";
    }
    else {
        document.getElementById("channelRangeDiv").style.display = "table";
    }
}

/**
 * This method is used to add a layer from select input */
overlayUtils.addLayerFromSelect = function() {
    var e = document.getElementById("layerSelect");
    var layerName = e.options[e.selectedIndex].text;
    var tileSource = e.options[e.selectedIndex].value;
    var layer = {
        name: layerName,
        tileSource: tileSource
    }
    tmapp.layers.push(layer);
    i = tmapp.layers.length - 1;
    overlayUtils.addLayer(layer, i);
    overlayUtils.addAllLayersSettings();
}

/**
 * This method is used to add a layer */
overlayUtils.addLayer = function(layer, i, visible) {
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    const path = urlParams.get('path')
    var layerName = layer.layerName
    var tileSource = layer.tileSource
    if (path != null) {
        tileSource = path + "/" + tileSource
    }
    var op = tmapp["object_prefix"];
    var vname = op + "_viewer";
    var opacity = 1.0;
    if (i >= 0 && !visible) {
        opacity = 0.0;
    }
    var showModal = true;
    var loadingModal = null;
    setTimeout(function(){
        if (showModal)
            loadingModal = interfaceUtils.loadingModal("Converting image, please wait...");
    },800);
    if (tmapp["ISS_viewer"].world.getItemCount() != 0) {
        if (tmapp["ISS_viewer"].world.getItemAt(0).source.getTileUrl(0,0,0) == null) {
            tmapp["ISS_viewer"].close();
        }
    }
    var x = layer.x || 0;
    var y = layer.y || 0;
    var scale = layer.scale || 1;
    var flip = layer.flip || false;
    var rotation = layer.rotation || 0;
    if (layer.transform_matrix) {
        var transform_matrix = layer.transform_matrix.map(Number) 
        rotation = Math.atan2(transform_matrix[1], transform_matrix[0]) * 360 / (2*Math.PI)
        var shear_y = Math.atan2(transform_matrix[4], transform_matrix[1]) - Math.PI/2 - (2*Math.PI*rotation / 360)
        var scale_x = Math.sqrt(transform_matrix[0] * transform_matrix[0] + transform_matrix[3] * transform_matrix[3])
        var scale_y = Math.sqrt(transform_matrix[1] * transform_matrix[1] + transform_matrix[4] * transform_matrix[4]) * Math.cos(shear_y)
        if (scale_x < 0) {
            scale_x = -scale_x;
            flip = true;
        }
        if (scale_y < 0) {
            scale_y = -scale_y;
            flip = true;
            rotation += 180;
        }
        scale = (scale_x + scale_y) / 2.;
        x = transform_matrix[2]
        y = transform_matrix[5]
        layer.x = x;
        layer.y = y;
        layer.scale = scale;
        layer.rotation = rotation;
        layer.flip = flip;
    }
    
    tmapp[vname].addTiledImage({
        index: i + 1,
        x: 0,
        y: 0,
        tileSource: tmapp._url_suffix + tileSource,
        opacity: opacity,
        success: function(i) {
            layer0X = tmapp[op + "_viewer"].world.getItemAt(0).getContentSize().x;
            layerNX = tmapp[op + "_viewer"].world.getItemAt(tmapp[op + "_viewer"].world.getItemCount()-1).getContentSize().x;
            tmapp[op + "_viewer"].world.getItemAt(tmapp[op + "_viewer"].world.getItemCount()-1).setWidth(scale*layerNX/layer0X);
            var point = new OpenSeadragon.Point(x/layer0X, y/layer0X);
            tmapp[op + "_viewer"].world.getItemAt(tmapp[op + "_viewer"].world.getItemCount()-1).setPosition(point);
            tmapp[op + "_viewer"].world.getItemAt(tmapp[op + "_viewer"].world.getItemCount()-1).setRotation(rotation);
            tmapp[op + "_viewer"].world.getItemAt(tmapp[op + "_viewer"].world.getItemCount()-1).setFlip(flip);
            if(layer.clip) {
                tmapp[op + "_viewer"].world.getItemAt(tmapp[op + "_viewer"].world.getItemCount()-1).setClip(
                    new OpenSeadragon.Rect(layer.clip.x,layer.clip.y,layer.clip.w,layer.clip.h,layer.clip.degrees)
                );
            }
            if (loadingModal) {
                setTimeout(function(){$(loadingModal).modal("hide");}, 500);
            }
            showModal = false;
            overlayUtils.waitLayersReady().then(()=>{
                filterUtils.setCompositeOperation();
                filterUtils.getFilterItems();
                if (overlayUtils._collectionMode) {
                    filterUtils.setCollectionMode();
                }
            })
        },
        error: function(i) {
            if (loadingModal) {
                setTimeout(function(){$(loadingModal).modal("hide");}, 500);
            }
            interfaceUtils.alert("Impossible to load file.")
            showModal = false;
        } 
    });
}

/** 
 * @summary Set collection mode of layers */
 overlayUtils.setCollectionMode = function() {
    var op = tmapp["object_prefix"];
    overlayUtils.waitLayersReady().then(() => {
        if (projectUtils._activeState.collectionMode) {
            overlayUtils._collectionMode = true;
            var collectionLayout = {
                tileSize: 1, tileMargin: 0.1,
                columns: Math.ceil(Math.sqrt(tmapp.layers.length))
            }
            if (projectUtils._activeState["collectionLayout"] !== undefined) {
                collectionLayout = projectUtils._activeState["collectionLayout"];
            }
            tmapp["ISS_viewer"].world.arrange(collectionLayout);
            var inputs = document.querySelectorAll(".visible-layers");
            for(var i = 0; i < inputs.length; i++) {
                inputs[i].checked = false;
                inputs[i].click();
            }
            tmapp["ISS_viewer"].viewport.goHome();
            $(".channelRange").hide();
        }
        else if (overlayUtils._collectionMode){
            overlayUtils._collectionMode = false;
            tmapp["ISS_viewer"].world.setAutoRefigureSizes(false);
            for (var i = 0; i < tmapp["ISS_viewer"].world._items.length; i++) {
                layer = tmapp.layers[i];
                var x = layer.x || 0;
                var y = layer.y || 0;
                var scale = layer.scale || 1;
                var  item = tmapp["ISS_viewer"].world._items[i];
                var layer0X = tmapp[op + "_viewer"].world.getItemAt(0).getContentSize().x;
                var layerNX = item.getContentSize().x;
                item.setWidth(scale*layerNX/layer0X);
                var point = new OpenSeadragon.Point(x/layer0X, y/layer0X);
                item.setPosition(point);
            }
            tmapp["ISS_viewer"].world.setAutoRefigureSizes(true);
            tmapp["ISS_viewer"].viewport.goHome();
            $(".channelRange").show();
        }
    });
}


/** 
 * @param {Number} item Index of an OSD tile source
 * @summary Set the opacity of a tile source */
overlayUtils.setItemOpacity = function(item) {
    opacity = overlayUtils._layerOpacities[item];

    var op = tmapp["object_prefix"];
    if (!tmapp[op + "_viewer"].world.getItemAt(item)) {
        setTimeout(function() {
            overlayUtils.setItemOpacity(item);
        }, 100);
        return;
    }
    tmapp[op + "_viewer"].world.getItemAt(item).setOpacity(opacity);
}

overlayUtils.areAllFullyLoaded = function () {
    var tiledImage;
    var op = tmapp["object_prefix"];
    var count = tmapp[op + "_viewer"].world.getItemCount();
    for (var i = 0; i < count; i++) {
      tiledImage = tmapp[op + "_viewer"].world.getItemAt(i);
      if (Object.keys(tiledImage.loadingCoverage).length > 0) {
        if (!tiledImage.getFullyLoaded() && tiledImage.getOpacity() != 0) {
            return false;
        }
      }
    }
    return true;
}

overlayUtils.waitLayersReady = async function () {
    var op = tmapp["object_prefix"];
    await new Promise(r => setTimeout(r, 200));
    while (!(!tmapp[op + "_viewer"].world || !tmapp[op + "_viewer"].world.getItemCount() != tmapp.layers.length)) {
        await new Promise(r => setTimeout(r, 200));
    }
}

overlayUtils.waitFullyLoaded = async function () {
    await new Promise(r => setTimeout(r, 200));
    while (!overlayUtils.areAllFullyLoaded()) {
        await new Promise(r => setTimeout(r, 200));
    }
}

/** 
 * @param {String} layerName name of an existing d3 node
 * @param {Number} opacity desired opacity
 * @summary Set the opacity of a tile source */
overlayUtils.setLayerOpacity= function(layerName,opacity){
    if(layerName in overlayUtils._d3nodes){
        var layer = overlayUtils._d3nodes[layerName];
        layer._groups[0][0].style.opacity=opacity;
    }else{
        console.log("layer does not exist or is not a D3 node");
    }
}

/**
 * @param {String} colortype A string from [hex,hsl,rgb]
 * @summary Get a random color in the desired format
 */
overlayUtils.randomColor = function (colortype) {
    if (!colortype) {
        colortype = "hex";
    }
    //I need random colors that are far away from the palette in the image
    //in this case Hematoxilyn and DAB so far away from brown and light blue
    //and avoid light colors because of the white  background 
    //in HSL color space this means L from 0.2 to 0.75
    //H [60,190],[220,360], S[0.3, 1.0]
    var rh1 = Math.floor(Math.random() * (190 - 60 + 1)) + 60;
    var rh2 = Math.floor(Math.random() * (360 - 220 + 1)) + 220;
    var H = 0.0;

    if (Math.random() > 0.5) { H = rh1; } else { H = rh2; }

    var L = Math.floor(Math.random() * (75 - 20 + 1)) + 20 + '%';
    var S = Math.floor(Math.random() * (100 - 40 + 1)) + 40 + '%';

    var hslstring = 'hsl(' + H.toString() + ',' + S.toString() + ',' + L.toString() + ')';

    var d3color = d3.hsl(hslstring);
    if (colortype == "hsl") return hslstring;
    if (colortype == "rgb") {
        return d3color.rgb().toString();
    }
    if (colortype == "hex") {
        var hex = function (value) {
            value = Math.max(0, Math.min(255, Math.round(value) || 0));
            return (value < 16 ? "0" : "") + value.toString(16);
        }
        var rgbcolor = d3color.rgb();
        return "#" + hex(rgbcolor.r) + hex(rgbcolor.g) + hex(rgbcolor.b);
    }
}

/**
 * Main function to update the view if there has been a reason for it. 
 * It computes all the elements that have to be drawn.
 */
overlayUtils.modifyDisplayIfAny = function () {
    //get four corners of view
    var op = tmapp["object_prefix"];
    var bounds = tmapp[op + "_viewer"].viewport.getBounds();
    var currentZoom = tmapp[op + "_viewer"].viewport.getZoom();

    var xmin, xmax, ymin, ymax;
    xmin = bounds.x; ymin = bounds.y;
    xmax = xmin + bounds.width; ymax = ymin + bounds.height;

    var imageWidth = OSDViewerUtils.getImageWidth();
    var imageHeight = OSDViewerUtils.getImageHeight();

    if (xmin < 0) { xmin = 0; }; if (xmax > 1.0) { xmax = 1.0; };
    if (ymin < 0) { ymin = 0; }; if (ymax > imageHeight / imageWidth) { ymax = imageHeight / imageWidth; };

    var total = imageWidth * imageHeight;

    //convert to global image coords
    xmin *= imageWidth; xmax *= imageWidth; ymin *= imageWidth; ymax *= imageWidth;

    var portion = (xmax - xmin) * (ymax - ymin);
    var percentage = portion / total;
}

/**
 * Save the current SVG overlay to open in a vector graphics editor for a figure for example
 */
overlayUtils.saveSVG=function(){
    var svg = d3.select("svg");
    var svgData = svg._groups[0][0].outerHTML;
    var svgBlob = new Blob([svgData], {type:"image/svg+xml;charset=utf-8"});
    var svgUrl = URL.createObjectURL(svgBlob);
    var downloadLink = document.createElement("a");
    downloadLink.href = svgUrl;
    downloadLink.download = "currentview.svg";
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink); 
}

/**
 * Save the current canvas as a PNG image
 */
overlayUtils.savePNG=function() {
    interfaceUtils.prompt("Resolution for export (1 = screen resolution):<br/>High resolution can take time to load!</small>","4","Capture viewport","number")
    .then((resolution) => {
        var bounds = tmapp.ISS_viewer.viewport.getBounds();
        var loading=interfaceUtils.loadingModal();
        tmapp.ISS_viewer.world.getItemAt(0).immediateRender = true
        var strokeWidth = regionUtils._polygonStrokeWidth
        regionUtils._polygonStrokeWidth *= resolution
        overlayUtils.waitFullyLoaded().then(() => {
            overlayUtils.getCanvasPNG(resolution)
            .then (() => {
                // We go back to original size:
                regionUtils._polygonStrokeWidth = strokeWidth;
                tmapp.ISS_viewer.world.getItemAt(0).immediateRender = false
                tmapp.ISS_viewer.viewport.fitBounds(bounds, true);
                setTimeout(()=>{$(loading).modal("hide");}, 300);
                
                document.getElementById("ISS_viewer").style.setProperty("visibility", "unset");
            })
        });
    })
}

/**
 * Get the current canvas as a PNG image
 */
 overlayUtils.getCanvasPNG=function(tiling) {
    tiling = tiling ? Math.ceil(tiling) : 1;
    function sleep (time) {
        return new Promise((resolve) => setTimeout(resolve, time));
    }
    function getCanvasCtx_aux (index, size, ctx, bounds) {
        return new Promise((resolve, reject) => {
            if (index == size*size) {
                resolve(ctx);
                return;
            }
            var index_x = index % size;
            var index_y = Math.floor(index/size);
            var newBounds = new OpenSeadragon.Rect(
                bounds.x+index_x*bounds.width/size,
                bounds.y+index_y*bounds.height/size,
                bounds.width/size,
                bounds.height/size,
                0
            );
            console.log(bounds, newBounds);
            tmapp.ISS_viewer.viewport.fitBounds(newBounds, true);
            overlayUtils.waitFullyLoaded().then(() => {
                overlayUtils.getCanvasCtx().then ((ctx_offset) => {
                    ctx.drawImage(
                        ctx_offset.canvas, 
                        ctx_offset.canvas.width * index_x, 
                        ctx_offset.canvas.height * index_y, 
                        ctx_offset.canvas.width,
                        ctx_offset.canvas.height
                    );
                    setTimeout(() => {
                        getCanvasCtx_aux(index+1, size, ctx, bounds).then(
                            (ctx)=>{
                                resolve(ctx);
                                return;
                            }
                            
                        )
                    },200);
                });
            });
        });
    }
    function add_colorbar (ctx, resolution) {
        var ctx_colorbar = document.querySelector("#colorbar_canvas").getContext("2d");
        if (ctx_colorbar.canvas.classList.contains("d-none")) return;
        // Set up CSS size.
        ctx_colorbar.canvas.style.width = ctx_colorbar.canvas.style.width || ctx_colorbar.canvas.width + 'px';
        ctx_colorbar.canvas.style.height = ctx_colorbar.canvas.style.height || ctx_colorbar.canvas.height + 'px';

        // Resize canvas and scale future draws.
        glUtils._updateColorbarCanvas(resolution)
        var ctx_colorbar_width = ctx_colorbar.canvas.width;
        var ctx_colorbar_height = ctx_colorbar.canvas.height;

        var width = ctx.canvas.width;
        var height = ctx.canvas.height;
        ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
        ctx.beginPath();
        ctx.fillRect(
            width - ctx_colorbar_width * 1 - 10,
            height - ctx_colorbar_height * 1 - 10,
            ctx_colorbar_width * 1,
            ctx_colorbar_height * 1,
            5*resolution
        );
        
        ctx.drawImage(
            ctx_colorbar.canvas, 
            width - ctx_colorbar_width * 1 - 10,
            height - ctx_colorbar_height * 1 - 10,
            ctx_colorbar_width * 1,
            ctx_colorbar_height * 1
        );
        glUtils._updateColorbarCanvas(1);
    }
    return new Promise((resolve, reject) => {
        if (tiling > 1) {
            var canvas = document.createElement("canvas");
            var ctx = canvas.getContext("2d");
            var ctx_osd = document.querySelector(".openseadragon-canvas canvas").getContext("2d");
            var ctx_webgl = document.querySelector("#gl_canvas").getContext("webgl2", glUtils._options);
            canvas.width = tiling * Math.min(ctx_osd.canvas.width, ctx_webgl.canvas.width);
            canvas.height = tiling * Math.min(ctx_osd.canvas.height, ctx_webgl.canvas.height);
            var bounds = tmapp.ISS_viewer.viewport.getBounds();
            getCanvasCtx_aux(0, tiling, ctx, bounds).then((ctx_tiling) => {
                ctx = add_colorbar (ctx, tiling);
                var png = ctx_tiling.canvas.toDataURL("image/png");
                
                var a = document.createElement("a"); //Create <a>
                a.href = png; //Image Base64 Goes here
                a.download = "TissUUmaps_capture.png"; //File name Here
                a.click(); //Downloaded file
                resolve(png);
            })
        }
        else {
            overlayUtils.getCanvasCtx().then((ctx)  =>{
                ctx = add_colorbar (ctx, 1);
                var png = ctx.canvas.toDataURL("image/png");
                
                var a = document.createElement("a"); //Create <a>
                a.href = png; //Image Base64 Goes here
                a.download = "TissUUmaps_capture.png"; //File name Here
                a.click(); //Downloaded file
                resolve(png);
            })
        }
    })
}

/**
 * Get the current canvas as a 2d context
 */
 overlayUtils.getCanvasCtx=function() {
    return new Promise((resolve, reject) => {
        // Create an empty canvas element
        var canvas = document.createElement("canvas");
        var ctx_osd = document.querySelector(".openseadragon-canvas canvas").getContext("2d");
        var ctx_webgl = document.querySelector("#gl_canvas").getContext("webgl2", glUtils._options);
        canvas.width = Math.min(ctx_osd.canvas.width, ctx_webgl.canvas.width);
        canvas.height = Math.min(ctx_osd.canvas.height, ctx_webgl.canvas.height);

        // Copy the image contents to the canvas
        var ctx = canvas.getContext("2d");
        if (projectUtils._activeState.backgroundColor) {
            ctx.fillStyle = projectUtils._activeState.backgroundColor;
            ctx.fillRect(
                0,
                0,
                canvas.width,
                canvas.height
            );
        }
        ctx.drawImage(ctx_osd.canvas, 0, 0, canvas.width, canvas.height);
        ctx.drawImage(ctx_webgl.canvas, 0, 0, canvas.width, canvas.height);
        
        var svgString = new XMLSerializer().serializeToString(document.querySelector('.openseadragon-canvas svg'));

        var DOMURL = self.URL || self.webkitURL || self;
        var img = new Image();
        var svg = new Blob([svgString], {type: "image/svg+xml;charset=utf-8"});
        var url = DOMURL.createObjectURL(svg);
        img.onload = function() {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            resolve(ctx);
            DOMURL.revokeObjectURL(url);
        };
        img.src = url;
    })
}
