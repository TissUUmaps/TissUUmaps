/**
 * @summary Generates region class accordions and region rows when regions
 * operations mode is toggled on
 */
regionUtils.createRegionOperationsTable = function () {
  const allRegionClasses = Object.values(regionUtils._regions).map(function (
    e
  ) {
    return e.regionClass;
  });
  const regionClasses = allRegionClasses.filter(
    (v, i, a) => a.indexOf(v) === i
  );

  const tableContainer = HTMLElementUtils.createElement({
    kind: "div",
    extraAttributes: { id: "region-operations-list" },
  });
  const table = HTMLElementUtils.createElement({
    kind: "table",
    extraAttributes: { class: "table table-striped marker_table" },
  });
  const thead = HTMLElementUtils.createElement({ kind: "thead" });
  const theadrow = HTMLElementUtils.createElement({ kind: "tr" });
  const tbody = HTMLElementUtils.createElement({
    kind: "tbody",
    extraAttributes: { id: "operations_accordion_table" },
  });

  const headerLabels = ["Show", "Class", "Count"];

  const operationLabels = [
    "Show",
    "Select",
    "Name",
    "Scale (%)",
    "Dilation/Erosion (Pixels)",
  ];

  const sortable = {
    "": "sorttable_nosort",
    Counts: "sorttable_sort",
    Class: "sorttable_sort col-md-8",
    Color: "sorttable_nosort",
    Color: "sorttable_nosort",
  };
  headerLabels.forEach((opt) => {
    var th = HTMLElementUtils.createElement({
      kind: "th",
      extraAttributes: { scope: "col", class: sortable[opt] },
    });
    th.innerText = opt;
    theadrow.appendChild(th);
  });

  thead.appendChild(theadrow);

  const toggleRegionsVisibilityCol = HTMLElementUtils.createElement({
    kind: "th",
  });

  theadrow.insertBefore(toggleRegionsVisibilityCol, theadrow.firstChild);

  // Get previous "All" checkbox element so that we can re-use its old state
  const lastCheckAll = interfaceUtils.getElementById(
    "regionUI_all_check",
    false
  );

  const regionsVisibilityCheck = HTMLElementUtils.createElement({
    kind: "input",
    id: "regionUI_all_check",
    extraAttributes: { class: "form-check-input", type: "checkbox" },
  });
  regionsVisibilityCheck.checked =
    lastCheckAll != null ? lastCheckAll.checked : true;
  toggleRegionsVisibilityCol.appendChild(regionsVisibilityCheck);
  regionsVisibilityCheck.addEventListener("input", (event) => {
    visible = event.target.checked;
    clist = interfaceUtils.getElementsByClassName("regionUI-region-input");
    for (let i = 0; i < clist.length; ++i) {
      clist[i].checked = visible;
    }
    groupRegions = Object.values(regionUtils._regions);
    for (region of groupRegions) {
      region.visibility = visible;
    }
    glUtils.updateRegionLUTTextures();
    glUtils.draw();
  });

  for (i of Object.keys(regionClasses.sort())) {
    const regionClass = regionClasses[i];
    const regionClassID = HTMLElementUtils.stringToId("region_" + regionClass);
    const numberOfRegions = allRegionClasses.filter(
      (x) => x == regionClass
    ).length;

    const regionClassLabel = HTMLElementUtils.createElement({
      kind: "p",
      extraAttributes: {
        class: "mb-0",
        style: "font-weight: bold;",
        type: "checkbox",
      },
    });

    regionClassLabel.innerText = `${
      regionClasses[i] !== "" ? regionClasses[i] : "Unclassified"
    }`;

    const regionClassCountLabel = HTMLElementUtils.createElement({
      kind: "p",
      extraAttributes: {
        class: "mb-0",
        type: "checkbox",
        id: regionClasses[i] + "_operations_count",
      },
    });

    regionClassCountLabel.innerText = numberOfRegions;

    //row
    const regionClassRow = HTMLElementUtils.createElement({
      kind: "tr",
      extraAttributes: {
        "data-escapedID": regionClassID,
        id: `region_class_operations_accordion_${regionClass}`,
      },
    });

    const td0 = HTMLElementUtils.createElement({
      kind: "td",
      extraAttributes: {
        "data-bs-toggle": "collapse",
        "data-bs-target": "#operations_collapse_region_" + regionClass,
        "aria-expanded": "false",
        "aria-controls": "operations_collapse_region_" + regionClass,
        class: "collapse_button_transform collapsed",
      },
    });
    const td1 = HTMLElementUtils.createElement({ kind: "td" });
    const td2 = HTMLElementUtils.createElement({ kind: "td" });
    const countCol = HTMLElementUtils.createElement({ kind: "td" });

    td2.appendChild(regionClassLabel);
    countCol.appendChild(regionClassCountLabel);

    regionClassRow.appendChild(td0);
    regionClassRow.appendChild(td1);
    regionClassRow.appendChild(td2);
    regionClassRow.appendChild(countCol);

    var check0 = HTMLElementUtils.createElement({
      kind: "input",
      id: "regionUI_operations" + regionClassID + "_check",
      extraAttributes: {
        class: "form-check-input regionUI-region-input",
        type: "checkbox",
      },
    });
    check0.checked = true;
    td1.appendChild(check0);

    var check1 = HTMLElementUtils.createElement({
      kind: "input",
      id: "regionUI_operations" + regionClassID + "_hidden",
      extraAttributes: {
        class: "form-check-input region-hidden d-none regionUI-region-hidden",
        type: "checkbox",
      },
    });
    check1.checked = true;
    td1.appendChild(check1);

    check0.addEventListener("input", function (event) {
      var visible = event.target.checked;
      clist = interfaceUtils.getElementsByClassName(
        "regionUI-region-" + regionClassID + "-input"
      );
      for (var i = 0; i < clist.length; ++i) {
        clist[i].checked = visible;
      }
      groupRegions = Object.values(regionUtils._regions).filter(
        (x) => x.regionClass == regionClass
      );
      for (region of groupRegions) {
        region.visibility = visible;
      }
      glUtils.updateRegionLUTTextures();
      glUtils.draw();
    });

    const regionItemsRow = document.createElement("tr");
    const regionItemsCol = document.createElement("td");
    regionItemsCol.classList.add("p-0");
    regionItemsCol.setAttribute("colspan", "100");
    let table_subregions = HTMLElementUtils.createElement({
      kind: "table",
      extraAttributes: { class: "table marker_table" },
    });
    var tbody_subregions = HTMLElementUtils.createElement({
      kind: "tbody",
      id: "tbody_subregions_operations_" + regionClassID,
    });
    let collapse_div = HTMLElementUtils.createElement({ kind: "div" });
    collapse_div.id = "operations_collapse_region_" + regionClass;
    collapse_div.setAttribute("data-region-class", regionClass);
    collapse_div.setAttribute("data-region-classID", regionClassID);
    collapse_div.classList.add("collapse");
    collapse_div.classList.add("container");
    collapse_div.classList.add("p-0");
    $(collapse_div).on("show.bs.collapse", function () {
      let selectedRegionClass = this.getAttribute("data-region-class");
      let selectedRegionClassID = this.getAttribute("data-region-classID");
      let tbody_subregions = document.getElementById(
        "tbody_subregions_operations_" + selectedRegionClassID
      );

      if (tbody_subregions.innerHTML == "") {
        var groupContainer = this;
        var numberOfItemsPerPage = 1000;
        let groupRegions = Object.values(regionUtils._regions).filter(
          (x) => x.regionClass == selectedRegionClass
        );
        let subGroupRegions = groupRegions.slice(0, numberOfItemsPerPage);
        regionDetails = document.createDocumentFragment();
        const labelRow = HTMLElementUtils.createElement({
          kind: "tr",
          extraAttributes: {
            style:
              "font-weight: bold; position: sticky; top: 0; background-color: var(--bs-primary); color: white;",
          },
        });
        operationLabels.forEach((opt) => {
          const col = HTMLElementUtils.createElement({
            kind: "td",
          });
          col.innerText = opt;
          labelRow.appendChild(col);
        });
        regionDetails.appendChild(labelRow);
        for (region of subGroupRegions) {
          regionDetails.appendChild(
            regionUtils.createRegionOperationsRow(region.id)
          );
        }
        tbody_subregions.appendChild(regionDetails);
        groupContainer.setAttribute("data-region-count", numberOfItemsPerPage);
        $(groupContainer).bind("scroll", function () {
          let tbody_subregions = document.getElementById(
            "tbody_subregions_operations_" + selectedRegionClassID
          );
          if (
            $(groupContainer).scrollTop() + $(this).innerHeight() >=
            $(this)[0].scrollHeight - 500
          ) {
            maxItem = parseInt(
              groupContainer.getAttribute("data-region-count")
            );
            groupContainer.setAttribute(
              "data-region-count",
              maxItem + numberOfItemsPerPage
            );
            if (maxItem >= groupRegions.length) return;
            let subGroupRegions = groupRegions.slice(
              maxItem,
              maxItem + numberOfItemsPerPage
            );
            regionDetails = document.createDocumentFragment();
            for (region of subGroupRegions) {
              regionDetails.appendChild(
                regionUtils.createRegionOperationsRow(region.id)
              );
            }
            tbody_subregions.appendChild(regionDetails);
          }
        });
        groupContainer.style.maxHeight = "400px";
        groupContainer.style.overflowY = "scroll";
      }
    });

    regionItemsRow.appendChild(regionItemsCol);
    regionItemsCol.appendChild(collapse_div);
    collapse_div.appendChild(table_subregions);
    table_subregions.appendChild(tbody_subregions);

    tbody.appendChild(regionClassRow);
    tbody.appendChild(regionItemsRow);
  }

  table.appendChild(thead);
  table.appendChild(tbody);
  tableContainer.appendChild(table);

  return tableContainer;
};

/**
 * @summary Creates a row for a region
 * @param {*} regionId Id of the region
 * @returns The row of the region
 */
regionUtils.createRegionOperationsRow = function (regionId) {
  const region = regionUtils._regions[regionId];
  const regionClass = region.regionClass;
  const regionClassID = HTMLElementUtils.stringToId("region_" + regionClass);

  const regionRow = HTMLElementUtils.createElement({
    kind: "tr",
    extraAttributes: {
      "data-escapedID": regionId,
      id: `operations_row_${regionId}`,
    },
  });

  const toggleVisibilityCheckCol = HTMLElementUtils.createElement({
    kind: "td",
  });

  const visibilityCheck = HTMLElementUtils.createElement({
    kind: "input",
    id: "singleRegionUI_" + regionId + "_check",
    extraAttributes: {
      class:
        "form-check-input regionUI-region-input regionUI-region-" +
        regionClassID +
        "-input",
      type: "checkbox",
    },
  });
  visibilityCheck.checked = true;
  toggleVisibilityCheckCol.appendChild(visibilityCheck);

  regionRow.appendChild(toggleVisibilityCheckCol);

  visibilityCheck.addEventListener("input", function (event) {
    var visible = event.target.checked;
    region.visibility = visible;
    glUtils.updateRegionLUTTextures();
    glUtils.draw();
  });

  const checkBoxCol = HTMLElementUtils.createElement({
    kind: "td",
    size: 2,
    extraAttributes: {
      class: "px-2 py-2",
    },
  });
  const checkBox = HTMLElementUtils.inputTypeCheckbox({
    id: region.id + "_selection_check",
    class: "form-check-input",
    value: false,
    eventListeners: {
      click: function () {
        this.checked
          ? regionUtils.selectRegion(region)
          : delete regionUtils._selectedRegions[region.id];
      },
    },
  });
  checkBoxCol.appendChild(checkBox);
  const regionNameInput = HTMLElementUtils.inputTypeText({
    id: region.id + "_scale_input",
    extraAttributes: {
      placeholder: "scale",
      value: regionUtils._regions[region.id].regionName,
      class: "col mx-1 input-sm form-control form-control-sm",
    },
  });
  regionNameInput.onkeyup = (event) => {
    regionUtils._regions[region.id].regionName = event.target.value;
    const otherNameInput = document.getElementById(region.id + "_name_ta");
    if (!otherNameInput) return;
    otherNameInput.value = event.target.value;
  };
  const regionNameCol = HTMLElementUtils.createElement({
    kind: "td",
  });
  regionNameCol.appendChild(regionNameInput);
  const regionScaleInput = HTMLElementUtils.inputTypeNumber({
    id: region.id + "_scale_input",
    min: 1,
    extraAttributes: {
      placeholder: "scale",
      value: regionUtils._regions[region.id].scale
        ? regionUtils._regions[region.id].scale
        : "100",
      class: "col mx-1 input-sm form-control form-control-sm",
    },
  });
  regionScaleInput.addEventListener("change", function () {
    regionUtils.resizeRegion(region.id, this.value);
  });
  const regionScaleCol = HTMLElementUtils.createElement({
    kind: "td",
    extraAttributes: {
      class: "px-2 py-2",
    },
  });
  regionScaleCol.appendChild(regionScaleInput);
  const regionOffsetDiv = document.createElement("div");
  regionOffsetDiv.classList.add("d-flex", "p-0");
  regionOffsetDiv.style.height = "39px";
  const regionOffsetInput = HTMLElementUtils.inputTypeNumber({
    id: region.id + "_offset_input",
    extraAttributes: {
      placeholder: "-+Pixels",
      class: "col mx-1 input-sm form-control form-control-sm",
    },
  });
  const generateOffsetButton = customElementUtils.dropDownButton(
    {
      id: region.id + "_offset_button",
      extraAttributes: {
        class: "btn btn-primary btn-sm",
        style: "width: 62px;",
      },
    },
    "Type",
    [
      {
        text: "Border",
        event: "onclick",
        handler: () => {
          if (!regionOffsetInput.value) return;
          const button = document.getElementById(region.id + "_offset_button");
          button.disabled = true;
          button.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>`;
          const worker = new Worker("static/js/utils/regionOffsetWorker.js");
          const point1 = turf.point([0, 0]);
          const point2 = turf.point([1, 0]);
          const distance = turf.distance(point1, point2, {
            units: "kilometers",
          });
          const offset =
            (regionOffsetInput.value / OSDViewerUtils.getImageWidth()) *
            distance;
          worker.postMessage([region, offset]);
          worker.onmessage = function (event) {
            if (!event.data) {
              interfaceUtils.alert(
                "An error ocurred applying the selected offset amount, for negative offsets, please make sure that the region is big enough to be offseted by that amount"
              );
              button.disabled = false;
              button.innerHTML = "Type";
              return;
            }
            regionUtils.drawOffsettedRegion(region, event.data, true);
            document.getElementById(region.id + "_show_preview").innerHTML =
              "Show preview";
            d3.select("#" + region.id + "preview" + "_poly").remove();
            button.disabled = false;
            button.innerHTML = "Type";
          };
          regionOffsetInput.value = "";
        },
      },
      {
        text: "Whole polygon",
        event: "onclick",
        handler: () => {
          if (!regionOffsetInput.value) return;
          const button = document.getElementById(region.id + "_offset_button");
          button.disabled = true;
          button.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>`;
          const worker = new Worker("static/js/utils/regionOffsetWorker.js");
          const point1 = turf.point([0, 0]);
          const point2 = turf.point([1, 0]);
          const distance = turf.distance(point1, point2, {
            units: "kilometers",
          });
          const offset =
            (regionOffsetInput.value / OSDViewerUtils.getImageWidth()) *
            distance;
          worker.postMessage([region, offset]);
          worker.onmessage = function (event) {
            if (!event.data) {
              interfaceUtils.alert(
                "An error ocurred applying the selected offset amount, for negative offsets, please make sure that the region is big enough to be offseted by that amount"
              );
              button.disabled = false;
              button.innerHTML = "Type";
              return;
            }
            regionUtils.drawOffsettedRegion(region, event.data, false);
            document.getElementById(region.id + "_show_preview").innerHTML =
              "Show preview";
            d3.select("#" + region.id + "preview" + "_poly").remove();
            button.disabled = false;
            button.innerHTML = "Type";
          };
          regionOffsetInput.value = "";
        },
      },
      {
        id: region.id + "_show_preview",
        text: "Show preview",
        event: "onclick",
        handler: function () {
          if (!regionOffsetInput.value) return;
          const option = this;
          const id = region.id + "preview";
          if (this.innerText.includes("Show")) {
            const button = document.getElementById(
              region.id + "_offset_button"
            );
            button.disabled = true;
            button.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>`;
            const worker = new Worker("static/js/utils/regionOffsetWorker.js");
            const point1 = turf.point([0, 0]);
            const point2 = turf.point([1, 0]);
            const distance = turf.distance(point1, point2, {
              units: "kilometers",
            });
            const offset =
              (regionOffsetInput.value / OSDViewerUtils.getImageWidth()) *
              distance;
            worker.postMessage([region, offset]);
            worker.onmessage = function (event) {
              regionUtils.drawRegionPath(
                regionUtils.arrayToObjectPoints(event.data),
                id
              );
              option.innerHTML = "Hide preview";
              button.disabled = false;
              button.innerHTML = "Type";
            };
            return;
          }
          d3.select("#" + id + "_poly").remove();
          option.innerHTML = "Show preview";
        },
      },
    ]
  );
  const regionOffsetCol = HTMLElementUtils.createElement({
    kind: "td",
    extraAttributes: {
      size: 3,
      class: "px-2 py-2",
    },
  });
  regionOffsetDiv.appendChild(regionOffsetInput);
  regionOffsetDiv.appendChild(generateOffsetButton);
  regionOffsetCol.appendChild(regionOffsetDiv);
  regionRow.appendChild(checkBoxCol);
  regionRow.appendChild(regionNameCol);
  regionRow.appendChild(regionScaleCol);
  regionRow.appendChild(regionOffsetCol);
  regionRow.onmouseover = function () {
    regionRow.style.background = "var(--bs-primary-light)";
    region.previousColor = region.polycolor;
    region.polycolor = "#39FF14";
    glUtils.updateRegionLUTTextures();
    glUtils.draw();
  };
  regionRow.onmouseout = function () {
    region.polycolor = region.previousColor;
    regionRow.style.background = "white";

    glUtils.updateRegionLUTTextures();
    glUtils.draw();
  };
  return regionRow;
};

/**
 * @summary Adds a region row to its corresponding region class accordion
 * @param {*} regionId Id of the region
 */
regionUtils.addRegionOperationsRow = function (regionId) {
  const operationLabels = [
    "Show",
    "Select",
    "Name",
    "Scale (%)",
    "Dilation/Erosion (Pixels)",
  ];
  const region = regionUtils._regions[regionId];
  const row = regionUtils.createRegionOperationsRow(regionId);
  const regionClassTable = document.getElementById(
    "tbody_subregions_operations_region_" + region.regionClass
  );
  if (!regionClassTable) {
    const allRegionClasses = Object.values(regionUtils._regions).map(function (
      e
    ) {
      return e.regionClass;
    });
    const table = document.getElementById("operations_accordion_table");
    const regionClassID = HTMLElementUtils.stringToId(
      "region_" + region.regionClass
    );
    const numberOfRegions = allRegionClasses.filter(
      (x) => x == region.regionClass
    ).length;
    const regionClassLabel = HTMLElementUtils.createElement({
      kind: "p",
      extraAttributes: {
        class: "mb-0",
        style: "font-weight: bold;",
        type: "checkbox",
      },
    });

    regionClassLabel.innerText = `${
      region.regionClass !== "" ? region.regionClass : "Unclassified"
    }`;

    const regionClassCountLabel = HTMLElementUtils.createElement({
      kind: "p",
      extraAttributes: {
        class: "mb-0",
        type: "checkbox",
        id: region.regionClass + "_operations_count",
      },
    });

    regionClassCountLabel.innerText = numberOfRegions;

    const regionClassRow = HTMLElementUtils.createElement({
      kind: "tr",
      extraAttributes: {
        "data-escapedID": regionClassID,
        id: `region_class_operations_accordion_${region.regionClass}`,
      },
    });

    const td0 = HTMLElementUtils.createElement({
      kind: "td",
      extraAttributes: {
        "data-bs-toggle": "collapse",
        "data-bs-target": "#operations_collapse_region_" + region.regionClass,
        "aria-expanded": "false",
        "aria-controls": "operations_collapse_region_" + region.regionClass,
        class: "collapse_button_transform collapsed",
      },
    });
    const td1 = HTMLElementUtils.createElement({ kind: "td" });
    const td2 = HTMLElementUtils.createElement({ kind: "td" });
    const countCol = HTMLElementUtils.createElement({ kind: "td" });

    td2.appendChild(regionClassLabel);
    countCol.appendChild(regionClassCountLabel);

    regionClassRow.appendChild(td0);
    regionClassRow.appendChild(td1);
    regionClassRow.appendChild(td2);
    regionClassRow.appendChild(countCol);

    var check0 = HTMLElementUtils.createElement({
      kind: "input",
      id: "regionUI_operations" + regionClassID + "_check",
      extraAttributes: {
        class: "form-check-input regionUI-region-input",
        type: "checkbox",
      },
    });
    check0.checked = true;
    td1.appendChild(check0);

    var check1 = HTMLElementUtils.createElement({
      kind: "input",
      id: "regionUI_operations" + regionClassID + "_hidden",
      extraAttributes: {
        class: "form-check-input region-hidden d-none regionUI-region-hidden",
        type: "checkbox",
      },
    });
    check1.checked = true;
    td1.appendChild(check1);

    check0.addEventListener("input", function (event) {
      var visible = event.target.checked;
      clist = interfaceUtils.getElementsByClassName(
        "regionUI-region-" + regionClassID + "-input"
      );
      for (var i = 0; i < clist.length; ++i) {
        clist[i].checked = visible;
      }
      groupRegions = Object.values(regionUtils._regions).filter(
        (x) => x.regionClass == region.regionClass
      );
      for (let item of groupRegions) {
        item.visibility = visible;
      }
      glUtils.updateRegionLUTTextures();
      glUtils.draw();
    });

    const regionItemsRow = document.createElement("tr");
    const regionItemsCol = document.createElement("td");
    regionItemsCol.classList.add("p-0");
    regionItemsCol.setAttribute("colspan", "100");
    let table_subregions = HTMLElementUtils.createElement({
      kind: "table",
      extraAttributes: { class: "table marker_table" },
    });
    var tbody_subregions = HTMLElementUtils.createElement({
      kind: "tbody",
      id: "tbody_subregions_operations_" + regionClassID,
    });
    let collapse_div = HTMLElementUtils.createElement({ kind: "div" });
    collapse_div.id = "operations_collapse_region_" + region.regionClass;
    collapse_div.setAttribute("data-region-class", region.regionClass);
    collapse_div.setAttribute("data-region-classID", regionClassID);
    collapse_div.classList.add("collapse");
    collapse_div.classList.add("container");
    collapse_div.classList.add("p-0");
    $(collapse_div).on("show.bs.collapse", function () {
      let selectedRegionClass = this.getAttribute("data-region-class");
      let selectedRegionClassID = this.getAttribute("data-region-classID");
      let tbody_subregions = document.getElementById(
        "tbody_subregions_operations_" + selectedRegionClassID
      );

      if (tbody_subregions.innerHTML == "") {
        var groupContainer = this;
        var numberOfItemsPerPage = 1000;
        let groupRegions = Object.values(regionUtils._regions).filter(
          (x) => x.regionClass == selectedRegionClass
        );
        let subGroupRegions = groupRegions.slice(0, numberOfItemsPerPage);
        regionDetails = document.createDocumentFragment();
        const labelRow = HTMLElementUtils.createElement({
          kind: "tr",
          extraAttributes: {
            style:
              "font-weight: bold; position: sticky; top: 0; background-color: var(--bs-primary); color: white;",
          },
        });
        operationLabels.forEach((opt) => {
          const col = HTMLElementUtils.createElement({
            kind: "td",
          });
          col.innerText = opt;
          labelRow.appendChild(col);
        });
        regionDetails.appendChild(labelRow);
        for (let item of subGroupRegions) {
          regionDetails.appendChild(
            regionUtils.createRegionOperationsRow(item.id)
          );
        }
        tbody_subregions.appendChild(regionDetails);
        groupContainer.setAttribute("data-region-count", numberOfItemsPerPage);
        $(groupContainer).bind("scroll", function () {
          let tbody_subregions = document.getElementById(
            "tbody_subregions_operations_" + selectedRegionClassID
          );
          if (
            $(groupContainer).scrollTop() + $(this).innerHeight() >=
            $(this)[0].scrollHeight - 500
          ) {
            maxItem = parseInt(
              groupContainer.getAttribute("data-region-count")
            );
            groupContainer.setAttribute(
              "data-region-count",
              maxItem + numberOfItemsPerPage
            );
            if (maxItem >= groupRegions.length) return;
            let subGroupRegions = groupRegions.slice(
              maxItem,
              maxItem + numberOfItemsPerPage
            );
            regionDetails = document.createDocumentFragment();
            for (region of subGroupRegions) {
              regionDetails.appendChild(
                regionUtils.createRegionOperationsRow(region.id)
              );
            }
            tbody_subregions.appendChild(regionDetails);
          }
        });
        groupContainer.style.maxHeight = "400px";
        groupContainer.style.overflowY = "scroll";
      }
    });

    regionItemsRow.appendChild(regionItemsCol);
    regionItemsCol.appendChild(collapse_div);
    collapse_div.appendChild(table_subregions);
    table_subregions.appendChild(tbody_subregions);

    table.appendChild(regionClassRow);
    table.appendChild(regionItemsRow);
    return;
  }
  const count = document.getElementById(
    region.regionClass + "_operations_count"
  );
  count.innerText = parseInt(count.innerText) + 1;
  try {
    regionClassTable.insertBefore(row, regionClassTable.firstChild.nextSibling);
  } catch {}
};

regionUtils.updateRegionOperationsListUI = function () {
  setTimeout(() => {
    const regionUI = regionUtils.createRegionOperationsTable();
    const container = interfaceUtils.getElementById("region-operations-list");
    container.innerHTML = "";
    container.appendChild(regionUI);
  }, 10);
  glUtils.updateRegionDataTextures();
  glUtils.updateRegionLUTTextures();
  glUtils.draw();
};

/**
 * @summary Adds the regions operations UI
 */
regionUtils.addRegionOperationsUI = function () {
  const regionOperationsUIContainer = interfaceUtils.getElementById(
    "region-operations-panel"
  );
  const buttonsContainer = document.createElement("div");
  buttonsContainer.classList.add(
    "d-flex",
    "flex-wrap",
    "mt-2",
    "py-1",
    "border-bottom"
  );
  const mergeButton = HTMLElementUtils.createButton({
    extraAttributes: { class: "btn btn-primary m-2" },
  });
  mergeButton.onclick = function () {
    const regions = Object.values(regionUtils._selectedRegions);
    if (regions.length < 2) return;
    regionUtils.mergeRegions(Object.values(regions));
    regionUtils.resetSelection();
  };
  mergeButton.style.width = "150px";
  mergeButton.innerHTML =
    '<img style="width: 30px; height: 20px;" src="static/misc/union.svg" /> Merge';
  const differenceButton = HTMLElementUtils.createButton({
    extraAttributes: { class: "btn btn-primary m-2" },
  });
  differenceButton.onclick = function () {
    const regions = Object.values(regionUtils._selectedRegions);
    if (regions.length < 2) return;
    regionUtils.regionsDifference(Object.values(regions));
    regionUtils.resetSelection();
  };
  differenceButton.style.width = "150px";
  differenceButton.innerHTML =
    '<img style="width: 30px; height: 20px;" src="static/misc/difference.svg" /> Difference';
  const intersectionButton = HTMLElementUtils.createButton({
    extraAttributes: { class: "btn btn-primary m-2" },
  });
  intersectionButton.onclick = function () {
    const regions = Object.values(regionUtils._selectedRegions);
    if (regions.length < 2) return;
    regionUtils.regionsIntersection(Object.values(regions));
    regionUtils.resetSelection();
  };
  intersectionButton.style.width = "150px";
  intersectionButton.innerHTML =
    '<img style="width: 30px; height: 20px;" src="static/misc/intersection.svg" /> Intersection';
  const duplicateButton = HTMLElementUtils.createButton({
    extraAttributes: { class: "btn btn-primary m-2" },
  });
  duplicateButton.onclick = function () {
    const regions = Object.values(regionUtils._selectedRegions);
    regionUtils.duplicateRegions(Object.values(regions));
    regionUtils.resetSelection();
  };
  duplicateButton.style.width = "150px";
  duplicateButton.innerHTML = '<i class="bi bi-back"></i> Duplicate';
  const deleteButton = HTMLElementUtils.createButton({
    extraAttributes: { class: "btn btn-primary m-2" },
  });
  deleteButton.onclick = function () {
    const regions = Object.values(regionUtils._selectedRegions);
    regionUtils.deleteRegions(
      Object.values(regions).map((region) => region.id)
    );
    regionUtils.resetSelection();
  };
  deleteButton.style.width = "150px";
  deleteButton.innerHTML = '<i class="bi bi-trash"></i> Delete';
  buttonsContainer.appendChild(deleteButton);
  buttonsContainer.appendChild(duplicateButton);
  buttonsContainer.appendChild(intersectionButton);
  buttonsContainer.appendChild(differenceButton);
  buttonsContainer.appendChild(mergeButton);
  regionOperationsUIContainer.appendChild(buttonsContainer);
  const regionOperationsUI = regionUtils.createRegionOperationsTable();
  regionOperationsUIContainer.appendChild(regionOperationsUI);
};

/**
 * @summary Deletes the rows of the regions corresponding to the ids
 * @param {*} regionIds Ids of the regions that will get their row removed
 */
regionUtils.deleteRegionOperationRows = function (regionId) {
  document.getElementById(`operations_row_${regionId}`).remove();
};

/**
 * @summary Deletes the region class accordion in the region operations UI
 * @param {*} regionClass Name of the class that will have its accordion removed
 */
regionUtils.deleteRegionOperationsAccordion = function (regionClass) {
  document
    .getElementById(`region_class_operations_accordion_${regionClass}`)
    .remove();
  document.getElementById(`operations_collapse_region_${regionClass}`).remove();
};
