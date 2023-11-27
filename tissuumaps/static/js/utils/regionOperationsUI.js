/**
 * @summary Adds the regions toolbar in UI
 */
regionUtils.addRegionToolbarUI = function () {
  let buttonsContainer = document.getElementById(
    "region-toolbar-buttons"
  );
  if (!buttonsContainer) {
    buttonsContainer = document.createElement("div");
    buttonsContainer.classList.add(
      "viewer-layer",
      "d-flex",
      "flex-wrap",
      "mt-2"
    );
    buttonsContainer.id = "region-toolbar-buttons";
    buttonsContainer.style.zIndex = "13";
    buttonsContainer.style.top = "2px";
    buttonsContainer.style.backgroundColor = "#fadfe199";
    buttonsContainer.style.marginRight = "165px";
    buttonsContainer.style.marginLeft = "14px";

    
    const multipleRegionOperations = [
      {"name": "union", "icon": `<img style="width: 30px; height: 20px;" src="static/misc/union.svg" />`, "title": "Merge selected regions"},
      {"name": "xor", "icon": `<img style="width: 30px; height: 20px;" src="static/misc/difference.svg" />`, "title": "XOR selected regions"},
      {"name": "intersect", "icon": `<img style="width: 30px; height: 20px;" src="static/misc/intersection.svg" />`, "title": "Intersect selected regions"},
    ];


    // Create dropdown button for region operations
    const regionOperationsDropdownButton = HTMLElementUtils.createElement({
      kind: "div",
      extraAttributes: { class: "btn-group" }
    });
    
    const mainRegionOperationsButton = HTMLElementUtils.createElement({
      kind: "button",
      extraAttributes: { class: "btn lh-1 btn-primary m-1 px-2 py-0 dropdown-toggle only-selected only-two-selected", "type": "button", "data-bs-toggle": "dropdown", "aria-expanded": "false" },
    });
    mainRegionOperationsButton.id = "region_operations_dropdown_button";
    const mainRegionOperationsButtonSpan = HTMLElementUtils.createElement({
      kind: "span",
      extraAttributes: { "title": "Boolean operations" }
    });
    mainRegionOperationsButtonSpan.innerHTML = multipleRegionOperations[1].icon;
    mainRegionOperationsButton.appendChild(mainRegionOperationsButtonSpan);
    var tooltip = new bootstrap.Tooltip(mainRegionOperationsButtonSpan, {
      placement: "bottom", trigger : 'hover',offset: [0, 9]
    });
    tooltip.enable();
    $(mainRegionOperationsButtonSpan).on('click', function () {
      $(this).tooltip('hide');
    });

    const regionOperationsDropdownMenu = HTMLElementUtils.createElement({
      kind: "ul",
      extraAttributes: { class: "dropdown-menu dropdown-menu-dark" }
    });
    regionOperationsDropdownMenu.style.backgroundColor = "var(--bs-primary)"

    // Create dropdown menu items for each region operation
    multipleRegionOperations.forEach(operation => {
      const menuItem = HTMLElementUtils.createElement({ kind: "li" });
      const anchor = HTMLElementUtils.createElement({
          kind: "a",
          extraAttributes: { class: "dropdown-item", href: "#" },
          innerHTML: operation.icon + "&nbsp;-&nbsp;" + operation.title
      });
      anchor.addEventListener("click", () => {
          regionUtils.regionsClipper(Object.values(regionUtils._selectedRegions), operation.name);
          //regionUtils.resetSelection();
      });
      menuItem.appendChild(anchor);
      regionOperationsDropdownMenu.appendChild(menuItem);
    });

    // Add the dropdown button and dropdown menu to the document
    regionOperationsDropdownButton.appendChild(mainRegionOperationsButton);
    regionOperationsDropdownButton.appendChild(regionOperationsDropdownMenu);



    const duplicateButton = HTMLElementUtils.createButton({
      extraAttributes: { class: "btn lh-1 btn-primary m-1 p-2 only-selected", "title": "Duplicate" },
    });
    duplicateButton.onclick = function () {
      const regions = Object.values(regionUtils._selectedRegions);
      regionUtils.duplicateRegions(regions);
      regionUtils.resetSelection();
    };
    var tooltip = new bootstrap.Tooltip(duplicateButton, {
      placement: "bottom", trigger : 'hover'
    });
    tooltip.enable();
    duplicateButton.innerHTML = '<i class="bi bi-back"></i>';
    const scaleButton = HTMLElementUtils.createButton({
      extraAttributes: { class: "btn lh-1 btn-primary m-1 p-2 only-selected", "title": "Scale" },
    });
    scaleButton.onclick = function () {
      const regions = Object.values(regionUtils._selectedRegions);
      regionUtils.resizeRegionsModal(regions);
      //regionUtils.resetSelection();
    };
    var tooltip = new bootstrap.Tooltip(scaleButton, {
      placement: "bottom", trigger : 'hover'
    });
    tooltip.enable();
    scaleButton.innerHTML = '<i class="bi bi-aspect-ratio"></i>';
    const dilateButton = HTMLElementUtils.createButton({
      extraAttributes: { class: "btn lh-1 btn-primary m-1 p-2 only-selected", "title": "Erode / Dilate" },
    });
    dilateButton.onclick = function () {
      const regions = Object.values(regionUtils._selectedRegions);
      regionUtils.dilateRegionsModal(regions);
      //regionUtils.resetSelection();
    };
    var tooltip = new bootstrap.Tooltip(dilateButton, {
      placement: "bottom", trigger : 'hover'
    });
    tooltip.enable();
    dilateButton.innerHTML = '<i class="bi bi-record-circle"></i>';
    
    const deleteButton = HTMLElementUtils.createButton({
      extraAttributes: { class: "btn lh-1 btn-primary m-1 p-2 only-selected", "title": "Delete" },
    });
    deleteButton.onclick = function () {
      const regions = Object.values(regionUtils._selectedRegions);
      regionUtils.deleteRegions(
        regions.map((region) => region.id)
      );
      regionUtils.resetSelection();
    };
    var tooltip = new bootstrap.Tooltip(deleteButton, {
      placement: "bottom", trigger : 'hover'
    });
    tooltip.enable();
    deleteButton.innerHTML = '<i class="bi bi-trash"></i>';
    
    const splitButton = HTMLElementUtils.createButton({
      extraAttributes: { class: "btn lh-1 btn-primary m-1 p-2 only-selected", "title": "Split multipolygons into multiple regions" },
    });
    splitButton.onclick = function () {
      const regions = Object.values(regionUtils._selectedRegions);
      regionUtils.splitRegions(
        regions.map((region) => region.id)
      );
      regionUtils.resetSelection();
    };
    var tooltip = new bootstrap.Tooltip(splitButton, {
      placement: "bottom", trigger : 'hover'
    });
    tooltip.enable();
    splitButton.innerHTML = '<i class="bi bi-percent"></i>';

    const fillHolesButton = HTMLElementUtils.createButton({
      extraAttributes: { class: "btn lh-1 btn-primary m-1 p-2 only-selected", "title": "Fill holes in regions" },
    });
    fillHolesButton.onclick = function () {
      const regions = Object.values(regionUtils._selectedRegions);
      regionUtils.fillHolesRegions(
        regions.map((region) => region.id)
      );
    };
    var tooltip = new bootstrap.Tooltip(fillHolesButton, {
      placement: "bottom", trigger : 'hover'
    });
    tooltip.enable();
    fillHolesButton.innerHTML = '<i class="bi bi-egg-fried"></i>';

    const unselectButton = HTMLElementUtils.createButton({
      extraAttributes: { class: "btn lh-1 btn-primary m-1 p-2 only-selected", "title": "Unselect all regions (escape)" },
    });
    unselectButton.onclick = function () {
      regionUtils.resetSelection();
    };
    var tooltip = new bootstrap.Tooltip(unselectButton, {
      placement: "bottom", trigger : 'hover'
    });
    tooltip.enable();
    unselectButton.innerHTML = '<i class="bi bi-dash-circle-dotted"></i>';
    
    const separator = HTMLElementUtils.createElement({
      kind: "div",
      extraAttributes: { class: "only-selected mx-1"},
      });
    separator.style.borderLeft = "2px solid var(--bs-primary)";
    separator.style.height = "22px";
    separator.style.margin = "auto 10px";
    separator.style.width = "0px";

    const paintingTools = [
        {"name": "free", "icon": "bi-pencil-square", "title": "Free hand drawing"},
        {"name": "points", "icon": "bi-pentagon", "title": "Point based drawing"},
        {"name": "brush", "icon": "bi-brush", "title": "Brush based drawing"},
        {"name": "rectangle", "icon": "bi-bounding-box-circles", "title": "Rectangle drawing"},
        {"name": "ellipse", "icon": "bi-circle", "title": "Ellipse drawing"},
    ];
    
    const dropdownButton = HTMLElementUtils.createElement({
        kind: "div",
        extraAttributes: { class: "btn-group" }
    });
    
    const mainButton = HTMLElementUtils.createElement({
        kind: "button",
        extraAttributes: { class: "btn lh-1 btn-light m-1 p-2 me-0", "type": "button" , "data-Selected": "0", "title": "Drawing tool"},
        innerHTML: "<i class='bi "+paintingTools[0].icon+"'></i>"
    });
    mainButton.id = "region_drawing_button"
    var tooltip = new bootstrap.Tooltip(mainButton, {
      placement: "bottom",trigger : 'hover',offset: [20, 0]
    });
    tooltip.enable();
    $(mainButton).on('click', function () {
      $(this).tooltip('hide');
    });
    mainButton.addEventListener("click", () => {
        const selected = parseInt(mainButton.dataset.selected);
        regionUtils.setMode(paintingTools[parseInt(selected)].name);
    });
    dropdownButton.appendChild(mainButton);
    
    const splitDropdownButton = HTMLElementUtils.createElement({
        kind: "button",
        extraAttributes: { class: "btn lh-1 btn-light m-1 p-2 ms-0 dropdown-toggle dropdown-toggle-split", "type": "button", "data-bs-toggle": "dropdown", "aria-expanded": "false",  "data-bs-reference": "parent" },
        innerHTML: '<span class="visually-hidden">Toggle Dropdown</span>'
    });
    splitDropdownButton.id = "region_drawing_button_dropdown"
    dropdownButton.appendChild(splitDropdownButton);
    
    const dropdownMenu = HTMLElementUtils.createElement({
        kind: "ul",
        extraAttributes: { class: "dropdown-menu" }
    });
    
    paintingTools.forEach(tool => {
        const menuItem = HTMLElementUtils.createElement({ kind: "li" });
        const anchor = HTMLElementUtils.createElement({
            kind: "a",
            extraAttributes: { class: "dropdown-item", href: "#" },
            innerHTML: "<i class='bi "+tool.icon+"'></i>&nbsp;-&nbsp;" + tool.title
        });
        anchor.addEventListener("click", () => {
            regionUtils._regionMode = null;
            mainButton.innerHTML = "<i class='bi "+tool.icon+"'></i>";
            mainButton.dataset.selected = paintingTools.indexOf(tool);
            regionUtils.setMode(tool.name);
        });
        menuItem.appendChild(anchor);
        dropdownMenu.appendChild(menuItem); 
    });
    
    dropdownButton.appendChild(dropdownMenu);
      

    const selectionButton = HTMLElementUtils.createButton({
      extraAttributes: { class: "btn lh-1 btn-light m-1 p-2", "title": "Select regions (press shift to select multiple regions)" },
    });
    selectionButton.id = "region_selection_button"
    selectionButton.onclick = function () {
      regionUtils.setMode("select");
    };
    var tooltip = new bootstrap.Tooltip(selectionButton, {
      placement: "bottom", trigger : 'hover'
    });
    tooltip.enable();
    selectionButton.innerHTML = '<i class="bi bi-cursor"></i>';

    const separator2 = HTMLElementUtils.createElement({
      kind: "div",
      extraAttributes: { class: "mx-1"},
      });
    separator2.style.borderLeft = "2px solid var(--bs-primary)";
    separator2.style.height = "22px";
    separator2.style.margin = "auto 10px";
    separator2.style.width = "0px";

    const showInstancesButton = HTMLElementUtils.createButton({
      extraAttributes: { class: "btn lh-1 btn-light m-1 p-2", "title": "Show instances" },
    });
    showInstancesButton.id = "region_show_instances_button";
    showInstancesButton.onclick = function () {
      regionUtils.showInstances();
    };
    var tooltip = new bootstrap.Tooltip(showInstancesButton, {
      placement: "bottom", trigger : 'hover'
    });
    tooltip.enable();
    showInstancesButton.innerHTML = '<i class="bi bi-layout-wtf"></i>';
        
    const zoomSelectedButton = HTMLElementUtils.createButton({
      extraAttributes: { class: "btn lh-1 btn-primary m-1 p-2 only-selected", "title": "Zoom to selected regions" },
    });
    zoomSelectedButton.onclick = function () {
      const regions = Object.values(regionUtils._selectedRegions);
      regionUtils.zoomToRegions(regions);
    };
    var tooltip = new bootstrap.Tooltip(zoomSelectedButton, {
      placement: "bottom", trigger : 'hover'
    });
    tooltip.enable();
    zoomSelectedButton.innerHTML = '<i class="bi bi-box-arrow-in-down-right"></i>';

    // New code for "Line width" dropdown button
    const lineWidthDropdownButton = HTMLElementUtils.createElement({
      kind: "div",
      extraAttributes: { class: "btn-group m-1" }
    });

    const lineWidthButton = HTMLElementUtils.createElement({
      kind: "button",
      extraAttributes: { class: "btn lh-1 btn-light px-2 py-0 dropdown-toggle", "type": "button", "data-bs-toggle": "dropdown", "aria-expanded": "false" },
    });
    lineWidthButton.id = "line_width_dropdown_button";
    const lineWidthButtonSpan = HTMLElementUtils.createElement({
      kind: "span",
      extraAttributes: { "title": "Line width" }
    });
    lineWidthButtonSpan.innerHTML = '<i class="bi bi-border-width"></i>';

    lineWidthButton.appendChild(lineWidthButtonSpan);

    const lineWidthDropdownMenu = HTMLElementUtils.createElement({
      kind: "div",
      extraAttributes: { class: "dropdown-menu p-2" }
    });

    // Range input for line width
    const lineWidthRangeInput = HTMLElementUtils.createElement({
      kind: "input",
      extraAttributes: { type: "range", min: "0", max: "2", step: "0.1", value: "1", class: "form-range" },
    });
    lineWidthDropdownMenu.appendChild(lineWidthRangeInput);
    lineWidthRangeInput.addEventListener("input", () => {
      // raise zoom event on tmapp["ISS_viewer"]:
      regionUtils._regionStrokeWidth = lineWidthRangeInput.value;
      tmapp["ISS_viewer"].raiseEvent("zoom", { zoom: tmapp["ISS_viewer"].viewport.getZoom() });
      glUtils.draw();
    });
    // Checkbox for "Adapt on zoom"
    const adaptOnZoomCheckbox = HTMLElementUtils.createElement({
      kind: "div",
      extraAttributes: { class: "form-check" }
    });
    const checkboxLabel = HTMLElementUtils.createElement({
      kind: "label",
      extraAttributes: { class: "form-check-label" }
    });
    const checkboxInput = HTMLElementUtils.createElement({
      kind: "input",
      extraAttributes: { type: "checkbox", class: "form-check-input" },
    });
    checkboxLabel.appendChild(checkboxInput);
    checkboxLabel.innerHTML += " Adapt on zoom";
    adaptOnZoomCheckbox.appendChild(checkboxLabel);
    lineWidthDropdownMenu.appendChild(adaptOnZoomCheckbox);

    adaptOnZoomCheckbox.addEventListener("input", (event) => {
      regionUtils._regionStrokeAdaptOnZoom = event.target.checked;
      tmapp["ISS_viewer"].raiseEvent("zoom", { zoom: tmapp["ISS_viewer"].viewport.getZoom() });
      glUtils.draw();
    });

    // Add the "Line width" dropdown button and menu to the document
    lineWidthDropdownButton.appendChild(lineWidthButton);
    lineWidthDropdownButton.appendChild(lineWidthDropdownMenu);

    var tooltip = new bootstrap.Tooltip(lineWidthButtonSpan, {
      placement: "bottom", trigger : 'hover', offset: [0, 9]
    });
    tooltip.enable();

    // Add "Fill opacity" dropdown button
    let fillOpacityDropdownButton = undefined;
    {
      fillOpacityDropdownButton = HTMLElementUtils.createElement({
        kind: "div",
        extraAttributes: { class: "btn-group m-1" }
      });

      // Toggle button
      const fillOpacityToggleButton = HTMLElementUtils.createElement({
        kind: "button",
        extraAttributes: { class: "btn lh-1 btn-light px-2 py-0", "type": "button", "title": "Fill opacity" },
      });
      fillOpacityToggleButton.id = "fill_opacity_button";
      fillOpacityToggleButton.innerHTML = '<i class="bi bi-front"></i>';
      
      // Add tooltip
      var tooltip = new bootstrap.Tooltip(fillOpacityToggleButton, {
        placement: "bottom", trigger: 'hover', offset: [20, 0]
      });
      tooltip.enable();

      // Down arrow button
      const fillOpacityDownArrowButton = HTMLElementUtils.createElement({
        kind: "button",
        extraAttributes: { class: "btn lh-1 btn-light px-2 py-0 dropdown-toggle dropdown-toggle-split", "type": "button", "data-bs-toggle": "dropdown", "aria-expanded": "false" },
        innerHTML: '<span class="visually-hidden">Toggle Dropdown</span>'
      });
      fillOpacityDownArrowButton.id = "fill_opacity_dropdown_button"

      // Dropdown menu
      const fillOpacityDropdownMenu = HTMLElementUtils.createElement({
        kind: "div",
        extraAttributes: { class: "dropdown-menu p-2" }
      });

      // Range input for region opacity
      const fillOpacityRangeInput = HTMLElementUtils.createElement({
        kind: "input",
        extraAttributes: { type: "range", min: "0", max: "1", step: "0.05", value: "0.5", class: "form-range" },
      });
      fillOpacityDropdownMenu.appendChild(fillOpacityRangeInput);
      fillOpacityRangeInput.addEventListener("input", () => {
        glUtils._regionOpacity = fillOpacityRangeInput.value;
        glUtils.draw();
      });

      // Checkbox for "Fill regions"
      const fillRegionsCheckbox = HTMLElementUtils.createElement({
        kind: "div",
        extraAttributes: { class: "form-check" }
      });
      const checkboxLabel = HTMLElementUtils.createElement({
        kind: "label",
        extraAttributes: { class: "form-check-label" }
      });
      const checkboxInput = HTMLElementUtils.createElement({
        kind: "input",
        extraAttributes: { type: "checkbox", class: "form-check-input" },
      });
      checkboxInput.id="fill_regions_checkbox";
      checkboxLabel.appendChild(checkboxInput);
      checkboxLabel.innerHTML += " Fill regions";
      fillRegionsCheckbox.appendChild(checkboxLabel);
      fillOpacityDropdownMenu.appendChild(fillRegionsCheckbox);
      fillRegionsCheckbox.addEventListener("input", (event) => {
        regionUtils.fillAllRegions();
      });
      fillOpacityToggleButton.addEventListener("click", () => {
        document.getElementById("fill_regions_checkbox").click();
      });

      // Add the toggle and down arrow buttons to the dropdown
      fillOpacityDropdownButton.appendChild(fillOpacityToggleButton);
      fillOpacityDropdownButton.appendChild(fillOpacityDownArrowButton);
      fillOpacityDropdownButton.appendChild(fillOpacityDropdownMenu);

    }

    buttonsContainer.appendChild(dropdownButton);
    buttonsContainer.appendChild(selectionButton);
    buttonsContainer.appendChild(separator2);
    buttonsContainer.appendChild(showInstancesButton);
    buttonsContainer.appendChild(fillOpacityDropdownButton);
    buttonsContainer.appendChild(lineWidthDropdownButton);
    buttonsContainer.appendChild(zoomSelectedButton);
    buttonsContainer.appendChild(separator);
    buttonsContainer.appendChild(unselectButton);
    buttonsContainer.appendChild(deleteButton);
    buttonsContainer.appendChild(duplicateButton);
    buttonsContainer.appendChild(scaleButton);
    buttonsContainer.appendChild(dilateButton);
    buttonsContainer.appendChild(splitButton);
    buttonsContainer.appendChild(fillHolesButton);
    buttonsContainer.appendChild(regionOperationsDropdownButton);
    tmapp.ISS_viewer.addControl(buttonsContainer, { anchor: OpenSeadragon.ControlAnchor.TOP_LEFT, autoFade: false });
  }
  if (overlayUtils._regionToolbar) {
    buttonsContainer.classList.remove("d-none");
  } else {
    buttonsContainer.classList.add("d-none");
  }
  if (Object.keys(regionUtils._selectedRegions).length > 0) {
    $(".only-selected").removeClass("d-none");
  } else {
    $(".only-selected").addClass("d-none");
  }
  if (Object.keys(regionUtils._selectedRegions).length > 1) {
    $(".only-two-selected").removeClass("disabled");
  } else {
    $(".only-two-selected").addClass("disabled");
  }
};

/**
 * @summary Merges a collection of regions into one individual region
 * @param {*} regions Array of regions to be merged
 */
regionUtils.resizeRegionsModal = async function (regions) {
  if (regions.length < 1) {
    interfaceUtils.alert("Please select at least one region");
    return;
  }
  var modalUID = "scale_messagebox"
  let button1=HTMLElementUtils.createButton({"extraAttributes":{ "class":"btn btn-primary mx-2"}})
  button1.innerText = "Apply";
  let button2=HTMLElementUtils.createButton({"extraAttributes":{ "class":"btn btn-secondary mx-2", "data-bs-dismiss":"modal"}})
  button2.innerText = "Cancel";
  let buttons=divpane=HTMLElementUtils.createElement({"kind":"div"});
  buttons.appendChild(button1);
  buttons.appendChild(button2);
  button1.addEventListener("click",function(event) {
      $(`#${modalUID}_modal`).modal('hide');
      const scale = document.getElementById(`scale_value_${modalUID}`).value;
      for (let region of regions) {
        regionUtils.resizeRegion(region.id, scale, false);
      }
      glUtils.updateRegionDataTextures();
      glUtils.updateRegionLUTTextures();
      glUtils.draw();
  })
  button2.addEventListener("click",function(event) {
      d3.selectAll(".region_previewpoly").remove();
      $(`#${modalUID}_modal`).modal('hide');
  })
  function preview (event) {
    const preview = document.getElementById(`${modalUID}_preview`).checked;
    if (!preview) {
      d3.selectAll(".region_previewpoly").remove();
      return;
    };
    const scale = document.getElementById(`scale_value_${modalUID}`).value;
    for (let region of regions) {
      regionUtils.resizeRegion(region.id, scale, true);
    }
  }
  const scaleValue = regions[0].scale?regions[0].scale:100;
  let content=HTMLElementUtils.createElement({"kind":"div"});
    row1=HTMLElementUtils.createRow({});
        col11=HTMLElementUtils.createColumn({"width":12});
            label111=HTMLElementUtils.createElement({"kind":"label", "extraAttributes":{ "for":"scale_value_" + modalUID }});
            label111.innerText="Scale (in percents):"
            value112=HTMLElementUtils.createElement({"kind":"input", "id":"scale_value_" + modalUID, "extraAttributes":{ "class":"form-text-input form-control", "type":"number", "value":scaleValue}});

    row3=HTMLElementUtils.createRow({});
    col31=HTMLElementUtils.createColumn({"width":12});
        divformcheck310=HTMLElementUtils.createElement({ "kind":"div", "extraAttributes":{"class":"form-check"}});
            inputcheck310=HTMLElementUtils.createElement({"kind":"input", "id":modalUID+"_preview","extraAttributes":{"class":"form-check-input","type":"checkbox", "checked":true }});
            label311=HTMLElementUtils.createElement({"kind":"label", "id":modalUID+"_preview-label", "extraAttributes":{ "for":modalUID+"_preview" }});
            label311.innerText="Preview"
  inputcheck310.addEventListener("input", preview);
  value112.addEventListener("input", preview);
  content.appendChild(row1);
    row1.appendChild(col11);
        col11.appendChild(label111);
        col11.appendChild(value112);
  content.appendChild(row3);
    row3.appendChild(col31);
      col31.appendChild(divformcheck310);
        divformcheck310.appendChild(label311);
        divformcheck310.appendChild(inputcheck310);
  const title = "Region scale"
  const apply = await interfaceUtils.generateModal(
    title, content, buttons, modalUID, true, true
  );
};

/**
 * @summary Dilates/Erodes a collection of regions
 * @param {*} regions Array of regions to be merged
 */
regionUtils.dilateRegionsModal = async function (regions) {
  if (regions.length < 1) {
    interfaceUtils.alert("Please select at least one region");
    return;
  }
  var modalUID = "dilate_messagebox"
  let button1=HTMLElementUtils.createButton({"extraAttributes":{ "class":"btn btn-primary mx-2"}})
  button1.innerText = "Apply";
  let button2=HTMLElementUtils.createButton({"extraAttributes":{ "class":"btn btn-secondary mx-2", "data-bs-dismiss":"modal"}})
  button2.innerText = "Cancel";
  let buttons=divpane=HTMLElementUtils.createElement({"kind":"div"});
  buttons.appendChild(button1);
  buttons.appendChild(button2);
  button1.addEventListener("click",function(event) {
      $(`#${modalUID}_modal`).modal('hide');
      const offset = document.getElementById(`dilate_value_${modalUID}`).value;
      const border = document.getElementById(`${modalUID}_border`).checked;
      for (let region of regions) {
        regionUtils.dilateRegion(region.id, offset, false, border);
      }
      glUtils.updateRegionDataTextures();
      glUtils.updateRegionLUTTextures();
      glUtils.draw();
  })
  button2.addEventListener("click",function(event) {
      d3.selectAll(".region_previewpoly").remove();
      $(`#${modalUID}_modal`).modal('hide');
  })
  function preview (event) {
    const preview = document.getElementById(`${modalUID}_preview`).checked;
    if (!preview) {
      d3.selectAll(".region_previewpoly").remove();
      return;
    };
    const offset = document.getElementById(`dilate_value_${modalUID}`).value;
    const border = document.getElementById(`${modalUID}_border`).checked;
    for (let region of regions) {
      regionUtils.dilateRegion(region.id, offset, true, border);
    }
  }
  let content=HTMLElementUtils.createElement({"kind":"div"});
    row1=HTMLElementUtils.createRow({});
        col11=HTMLElementUtils.createColumn({"width":12});
            label111=HTMLElementUtils.createElement({"kind":"label", "extraAttributes":{ "for":"dilate_value_" + modalUID }});
            label111.innerText="Size in pixel (negative for erosion):"
            value112=HTMLElementUtils.createElement({"kind":"input", "id":"dilate_value_" + modalUID, "extraAttributes":{ "class":"form-text-input form-control", "type":"number", "value":0}});

    row2=HTMLElementUtils.createRow({});
        col21=HTMLElementUtils.createColumn({"width":12});
            divformcheck210=HTMLElementUtils.createElement({ "kind":"div", "extraAttributes":{"class":"form-check"}});
                inputcheck210=HTMLElementUtils.createElement({"kind":"input", "id":modalUID+"_border","extraAttributes":{"class":"form-check-input","type":"checkbox" }});
                label211=HTMLElementUtils.createElement({"kind":"label", "id":modalUID+"_border-label", "extraAttributes":{ "for":modalUID+"_border" }});
                label211.innerText="Border only"
    
    row3=HTMLElementUtils.createRow({});
    col31=HTMLElementUtils.createColumn({"width":12});
        divformcheck310=HTMLElementUtils.createElement({ "kind":"div", "extraAttributes":{"class":"form-check"}});
            inputcheck310=HTMLElementUtils.createElement({"kind":"input", "id":modalUID+"_preview","extraAttributes":{"class":"form-check-input","type":"checkbox", "checked":true }});
            label311=HTMLElementUtils.createElement({"kind":"label", "id":modalUID+"_preview-label", "extraAttributes":{ "for":modalUID+"_preview" }});
            label311.innerText="Preview"
  inputcheck210.addEventListener("input", preview);
  inputcheck310.addEventListener("input", preview);
  value112.addEventListener("input", preview);
  content.appendChild(row1);
    row1.appendChild(col11);
        col11.appendChild(label111);
        col11.appendChild(value112);
  content.appendChild(row2);
    row2.appendChild(col21);
      col21.appendChild(divformcheck210);
        divformcheck210.appendChild(label211);
        divformcheck210.appendChild(inputcheck210);
  content.appendChild(row3);
    row3.appendChild(col31);
      col31.appendChild(divformcheck310);
        divformcheck310.appendChild(label311);
        divformcheck310.appendChild(inputcheck310);
  const title = "Region erosion / dilation"
  const apply = await interfaceUtils.generateModal(
    title, content, buttons, modalUID, true, true
  );
};
