regionUtils._selectedRegions = {};

regionUtils.regionOperationsOnOff = function () {
    overlayUtils._regionOperations = !overlayUtils._regionOperations;
    const op = tmapp["object_prefix"];
    const operationsRegionButtonIcon = document.getElementById(
        op + "_operations_regions_icon"
    );
    const regionAccordions = document.getElementById("markers-regions-panel")
    const regionOpertationsList = document.getElementById("region-operations-panel")
    if (overlayUtils._regionOperations) {
        console.log("adding ui")
        operationsRegionButtonIcon.classList.remove("bi-circle");
        operationsRegionButtonIcon.classList.add("bi-check-circle");
        regionUtils.showHint("Select regions");
        // Hide region accordions
        regionAccordions.style.display = "none"
        // Show region selection list 
        regionOpertationsList.classList.remove("d-none")
        // Add region selection UI
        regionUtils.addRegionOperationsUI()
    } else {
        regionUtils.resetSelection();
        operationsRegionButtonIcon.classList.remove("bi-check-circle");
        operationsRegionButtonIcon.classList.add("bi-circle");
        // Show region accordions
        regionAccordions.style.display = "block";
        // Remove region selection list contents
        regionOpertationsList.innerHTML = "";
        // Hide region selection UI
        regionOpertationsList.classList.add("d-none");
        regionUtils.hideHint();
    } 
};

/**
 * @summary Draws a path given a set of points and an Id
 * @param {*} points Region points used to construct the path
 * @param {*} regionId Id of the region
 */
regionUtils.drawRegionPath = function (points, regionId) {
  const canvasNode =
    overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
  const canvas = d3.select(canvasNode);
  const strokeWstr =
    regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom();
  canvas
    .append("path")
    .attr("d", regionUtils.pointsToPath(points))
    .attr("id", regionId + "_poly")
    .attr("class", "regionpoly")
    .attr("polycolor", "#FF0000")
    .attr("stroke-width", strokeWstr)
    .style("stroke", "#FF0000")
    .style("fill", "none")
    .append("title")
    .text(regionId)
    .attr("id", "path-title-" + regionId);
};

/**
 * @summary Deletes regions
 * @param {*} regions Array of regions to be deleted
 */
regionUtils.deleteRegions = function (regionIds) {
  regionIds.forEach((id) => {
    regionUtils.deleteRegion(id);
  });
  if (overlayUtils._regionOperations) {
    regionUtils.deleteRegionSelectionItems(regionIds);
  }
  regionUtils.updateAllRegionClassUI();
};

/**
 * @summary Duplicates regions
 * @param {*} regions Array of regions to be duplicated
 */
regionUtils.duplicateRegions = function (regions) {
  regions.forEach((region) => {
    const newRegionId = region.id + "duplicate";
    const hexColor = overlayUtils.randomColor("hex");
    regionUtils.addRegion(
      regionUtils.objectToArrayPoints(region.points),
      newRegionId,
      hexColor,
      region.regionClass,
      region.scale
    );
    regionUtils.drawRegionPath(region.points, newRegionId);
    regionUtils.addRegionSelectionItem(regionUtils._regions[newRegionId]);
  });
  regionUtils.updateAllRegionClassUI();
};

/**
 * @summary Generates a region that covers the instersection of the passed regions
 * @param {*} regions Array of regions to calculate the intersection
 */
regionUtils.regionsIntersection = function (regions) {
  try {
    const intersectionPoints = polygonClipping.intersection(
      ...regions.map((region) => regionUtils.objectToArrayPoints(region.points))
    );
    regionUtils._currentRegionId += 1;
    const newRegionId = "region" + regionUtils._currentRegionId;
    const hexColor = overlayUtils.randomColor("hex");
    regionUtils.addRegion(intersectionPoints, newRegionId, hexColor, "", "100");
    regionUtils.drawRegionPath(
      regionUtils.arrayToObjectPoints(intersectionPoints),
      newRegionId
    );
    regionUtils.updateAllRegionClassUI();
    regionUtils.addRegionSelectionItem(regionUtils._regions[newRegionId]);
  } catch {
    interfaceUtils.alert(
      "The selected regions have no interception between them"
    );
  }
};

/**
 * @summary Generates a region that is the difference between the selected regions
 * @param {*} regions Array of regions to calculate the difference
 */
regionUtils.regionsDifference = function (regions) {
  const differencePoints = polygonClipping.xor(
    ...regions.map((region) => regionUtils.objectToArrayPoints(region.points))
  );
  regionUtils._currentRegionId += 1;
  const newRegionId = "region" + regionUtils._currentRegionId;
  const hexColor = overlayUtils.randomColor("hex");
  regionUtils.addRegion(differencePoints, newRegionId, hexColor, "", "100");
  regionUtils.drawRegionPath(
    regionUtils.arrayToObjectPoints(differencePoints),
    newRegionId
  );
  regionUtils.updateAllRegionClassUI();
  regionUtils.addRegionSelectionItem(regionUtils._regions[newRegionId]);
};

/**
 * @summary Resizes a multipolygon maintaing the center of each of its polygons and
 * and updates the corresponding region's path
 * @param {*} regionId Id of the region being rescaled
 * @param {*} scale Scale factor to use in rescaling
 */
regionUtils.resizeRegion = function (regionId, scale) {
  const scaleFactor = scale / 100;
  const points = regionUtils.objectToArrayPoints(
    regionUtils._regions[regionId].points
  );

  //Iterate through each of the polygons in the multipolygon
  for (let i = 0; i < points.length; i++) {
    const polygon = points[i];
    const centroidBefore = calculatePolygonCentroid(polygon);

    // Scale up coordinates based on current region scale and new scale
    for (let j = 0; j < polygon.length; j++) {
      const ring = polygon[j];
      for (let k = 0; k < ring.length; k++) {
        const point = ring[k];
        point[0] =
          centroidBefore[0] +
          ((point[0] - centroidBefore[0]) * scaleFactor) /
            (regionUtils._regions[regionId].scale
              ? regionUtils._regions[regionId].scale / 100
              : 1);
        point[1] =
          centroidBefore[1] +
          ((point[1] - centroidBefore[1]) * scaleFactor) /
            (regionUtils._regions[regionId].scale
              ? regionUtils._regions[regionId].scale / 100
              : 1);
      }
    }

    // Calculate the difference between the previous center of the polygon and new one
    const centroidAfter = calculatePolygonCentroid(polygon);
    const centroidDiffX = centroidBefore[0] - centroidAfter[0];
    const centroidDiffY = centroidBefore[1] - centroidAfter[1];

    // Adjust coordinates to new polygon center
    for (let j = 0; j < polygon.length; j++) {
      const ring = polygon[j];
      for (let k = 0; k < ring.length; k++) {
        const point = ring[k];
        point[0] += centroidDiffX;
        point[1] += centroidDiffY;
      }
    }
  }

  // Save new region scale and points
  regionUtils._regions[regionId].scale = scale;
  regionUtils._regions[regionId].points =
    regionUtils.arrayToObjectPoints(points);

  // Adjust regions path to new coordinates
  d3.select(`#${regionId}_poly`).attr(
    "d",
    regionUtils.pointsToPath(regionUtils.arrayToObjectPoints(points))
  );

  // Returns the center of a given polygon
  function calculatePolygonCentroid(polygon) {
    let sumX = 0;
    let sumY = 0;
    let count = 0;

    for (const ring of polygon) {
      for (const point of ring) {
        sumX += point[0];
        sumY += point[1];
        count++;
      }
    }

    const centroidX = sumX / count;
    const centroidY = sumY / count;

    return [centroidX, centroidY];
  }
};

/**
 * @summary Generates an offsetted polygon
 * @param {*} region Region to base the new offset region
 * @param {*} offset Offset to be applied
 * @param {*} onlyBorder Determines if the resulting polygon will only have
 * an exterior ring or an exterior ring plus an interior ring determined by the
 * original region
 */
regionUtils.drawOffsettedRegion = function (region, points, onlyBorder) {
  if (onlyBorder) {
    points = polygonClipping.xor(
      regionUtils.objectToArrayPoints(region.points),
      points
    );
  }
  regionUtils._currentRegionId += 1;
  const newRegionId =
    region.id +
    "offsetR" +
    regionUtils._currentRegionId +
    (onlyBorder ? "border" : "");
  const hexColor = overlayUtils.randomColor("hex");
  regionUtils.addRegion(
    points,
    newRegionId,
    hexColor,
    region.regionClass,
    region.scale
  );
  regionUtils.drawRegionPath(
    regionUtils.arrayToObjectPoints(points),
    newRegionId
  );
  regionUtils.addRegionSelectionItem(regionUtils._regions[newRegionId]);
};

/**
 * @summary Converts Object based points into GeoJson format points
 * @param {*} points Object points with x and y properties to be converted into array pairs
 */
regionUtils.objectToArrayPoints = function (points) {
  return points.map((arr) =>
    arr.map((polygon) => polygon.map((point) => [point.x, point.y]))
  );
};

/**
 * @summary Converts GeoJson format points into object based points
 * @param {*} points GeoJson points to be converted into object points with x and y properties
 */
regionUtils.arrayToObjectPoints = function (points) {
  return points.map((arr) =>
    arr.map((secondArr) =>
      secondArr.map((coordinates) => {
        return {
          x: coordinates[0],
          y: coordinates[1],
        };
      })
    )
  );
};

/**
 * @summary Converts string coordinates into numbers
 * @param {*} points Array pairs of string coordinates
 */
regionUtils.stringToFloatPoints = function (points) {
  return points.map((arr) =>
    arr.map((secondArr) =>
      secondArr.map((coordinates) => [
        typeof coordinates[0] === "string"
          ? parseFloat(coordinates[0])
          : coordinates[0],
        typeof coordinates[1] === "string"
          ? parseFloat(coordinates[1])
          : coordinates[1],
      ])
    )
  );
};

/**
 * @summary Merges a collection of regions into one individual region
 * @param {*} regions Array of regions to be merged
 */
regionUtils.mergeRegions = function (regions) {
  const mergedPoints = polygonClipping.union(
    ...regions.map((region) => regionUtils.objectToArrayPoints(region.points))
  );
  regionUtils._currentRegionId += 1;
  const newRegionId = "region" + regionUtils._currentRegionId;
  const hexColor = overlayUtils.randomColor("hex");
  regionUtils.addRegion(mergedPoints, newRegionId, hexColor, "", "100");
  const canvasNode =
    overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
  const canvas = d3.select(canvasNode);
  const strokeWstr =
    regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom();
  canvas
    .append("path")
    .attr(
      "d",
      regionUtils.pointsToPath(regionUtils.arrayToObjectPoints(mergedPoints))
    )
    .attr("id", newRegionId + "_poly")
    .attr("class", "regionpoly")
    .attr("polycolor", "#FF0000")
    .attr("stroke-width", strokeWstr)
    .style("stroke", "#FF0000")
    .style("fill", "none")
    .append("title")
    .text(newRegionId)
    .attr("id", "path-title-" + newRegionId);
  regionUtils.updateAllRegionClassUI();
  regionUtils.addRegionSelectionItem(regionUtils._regions[newRegionId]);
};

// /**
//  * @summary Generates the region selection menu UI
//  */
// regionUtils.addRegionOperationsUI = function () {
//   const operationsListContainer = document.getElementById(
//     "regionOperationsList"
//   );
//   operationsListContainer.innerHTML = "";
//   const buttonsContainer = document.createElement("div");
//   buttonsContainer.classList.add(
//     "d-flex",
//     "flex-wrap",
//     "justify-content-end",
//     "mt-2",
//     "py-1",
//     "border-bottom"
//   );
//   const mergeButton = HTMLElementUtils.createButton({
//     extraAttributes: { class: "btn btn-primary m-2" },
//   });
//   mergeButton.onclick = function () {
//     const regions = Object.values(regionUtils._selectedRegions);
//     if (regions.length < 2) return;
//     regionUtils.mergeRegions(Object.values(regions));
//     regionUtils.resetSelection();
//   };
//   mergeButton.style.width = "150px";
//   mergeButton.innerHTML =
//     '<img style="width: 30px; height: 20px;" src="static/misc/union.svg" /> Merge';
//   const differenceButton = HTMLElementUtils.createButton({
//     extraAttributes: { class: "btn btn-primary m-2" },
//   });
//   differenceButton.onclick = function () {
//     const regions = Object.values(regionUtils._selectedRegions);
//     if (regions.length < 2) return;
//     regionUtils.regionsDifference(Object.values(regions));
//     regionUtils.resetSelection();
//   };
//   differenceButton.style.width = "150px";
//   differenceButton.innerHTML =
//     '<img style="width: 30px; height: 20px;" src="static/misc/difference.svg" /> Difference';
//   const intersectionButton = HTMLElementUtils.createButton({
//     extraAttributes: { class: "btn btn-primary m-2" },
//   });
//   intersectionButton.onclick = function () {
//     const regions = Object.values(regionUtils._selectedRegions);
//     if (regions.length < 2) return;
//     regionUtils.regionsIntersection(Object.values(regions));
//     regionUtils.resetSelection();
//   };
//   intersectionButton.style.width = "150px";
//   intersectionButton.innerHTML =
//     '<img style="width: 30px; height: 20px;" src="static/misc/intersection.svg" /> Intersection';
//   const duplicateButton = HTMLElementUtils.createButton({
//     extraAttributes: { class: "btn btn-primary m-2" },
//   });
//   duplicateButton.onclick = function () {
//     const regions = Object.values(regionUtils._selectedRegions);
//     regionUtils.duplicateRegions(Object.values(regions));
//     regionUtils.resetSelection();
//   };
//   duplicateButton.style.width = "150px";
//   duplicateButton.innerHTML = '<i class="bi bi-back"></i> Duplicate';
//   const deleteButton = HTMLElementUtils.createButton({
//     extraAttributes: { class: "btn btn-primary m-2" },
//   });
//   deleteButton.onclick = function () {
//     const regions = Object.values(regionUtils._selectedRegions);
//     regionUtils.deleteRegions(
//       Object.values(regions).map((region) => region.id)
//     );
//     regionUtils.resetSelection();
//   };
//   deleteButton.style.width = "150px";
//   deleteButton.innerHTML = '<i class="bi bi-trash"></i> Delete';
//   buttonsContainer.appendChild(deleteButton);
//   buttonsContainer.appendChild(duplicateButton);
//   buttonsContainer.appendChild(intersectionButton);
//   buttonsContainer.appendChild(differenceButton);
//   buttonsContainer.appendChild(mergeButton);
//   operationsListContainer.appendChild(buttonsContainer);
//   const regionsArray = Object.values(regionUtils._regions);
//   const regionClasses = new Set(
//     regionsArray.map((region) => region.regionClass)
//   );
//   regionClasses.forEach((regionClass) =>
//     regionUtils.addRegionClassSelectionAccordion(regionClass)
//   );
//   regionsArray.forEach((region) => regionUtils.addRegionSelectionItem(region));
// };

/**
 * @summary Generates the region selection menu UI
 */
regionUtils.addRegionOperationsUI = function () {
    const regionOperationsUIContainer = interfaceUtils.getElementById("region-operations-panel");
    const regionOperationsUI = interfaceUtils._rGenUIFuncs.createRegionOperationsTable();
    regionOperationsUIContainer.appendChild(regionOperationsUI)
};

/**
 * @summary Creates a new accordion in the selection menu for a given region class
 * @param {*} regionClass Name of the region class
 */
regionUtils.addRegionClassSelectionAccordion = function (regionClass) {
  const operationsListContainer = document.getElementById(
    "regionOperationsList"
  );
  const regionClassID = HTMLElementUtils.stringToId(regionClass);
  const classAccordion = HTMLElementUtils.createElement({
    kind: "div",
    extraAttributes: {
      class: "accordion-item region-accordion",
      id: "regionClassSelectionItem-" + regionClassID,
    },
  });
  const accordionHeader = HTMLElementUtils.createElement({
    kind: "h2",
    extraAttributes: {
      class: "accordion-header",
      id: "regionClassSelectionHeading-" + regionClassID,
    },
  });
  const accordionHeaderButton = HTMLElementUtils.createElement({
    kind: "button",
    innerHTML:
      "<i class='bi bi-pentagon'></i>&nbsp;" +
      (regionClass ? regionClass : "Unclassified") +
      " (<span id='numRegionsSelection-" +
      regionClassID +
      "'>0</span>&nbsp;region<span id='numRegionsSelectionS-" +
      regionClassID +
      "'></span>)&nbsp;<span class='text-warning' id='regionGroupWarningSelection-" +
      regionClassID +
      "'></span>",
    extraAttributes: {
      type: "button",
      class: "accordion-button collapsed",
      id: "regionClassSelectionHeading-" + regionClassID,
      "data-bs-toggle": "collapse",
      "data-bs-target": "#" + "regionClassSelection-" + regionClassID,
      "aria-expanded": "true",
      "aria-controls": "collapseOne",
    },
  });
  const accordionContent = HTMLElementUtils.createElement({
    kind: "div",
    extraAttributes: {
      class: "accordion-collapse collapse px-2",
      id: "regionClassSelection-" + regionClassID,
      "aria-labelledby": "headingOne",
      "data-bs-parent": "#regionAccordions",
    },
  });
  const table = HTMLElementUtils.createElement({
    kind: "table",
    extraAttributes: {
      class: "table regions_table",
      id: "operations-regions-selection-table-" + regionClassID,
    },
  });
  classAccordion.appendChild(accordionHeader);
  classAccordion.appendChild(accordionContent);
  accordionHeader.appendChild(accordionHeaderButton);
  const tableHead = document.createElement("thead");
  const tableHeadTr = document.createElement("tr");
  tableHead.appendChild(tableHeadTr);
  tableHeadTr.appendChild(
    HTMLElementUtils.createElement({ kind: "th", innerText: "Select" })
  );
  tableHeadTr.appendChild(
    HTMLElementUtils.createElement({ kind: "th", innerText: "Name" })
  );
  tableHeadTr.appendChild(
    HTMLElementUtils.createElement({ kind: "th", innerText: "Scale (%)" })
  );
  tableHeadTr.appendChild(
    HTMLElementUtils.createElement({
      kind: "th",
      innerText: "Offset Polygon Generation (Pixels)",
    })
  );
  table.appendChild(tableHead);
  accordionContent.appendChild(table);
  operationsListContainer.appendChild(classAccordion);
};

/**
 * @summary Deletes the selection items of the regions corresponding to the ids
 * @param {*} regionIds Ids of the regions that will get their selection item removed
 */
regionUtils.deleteRegionSelectionItems = function (regionIds) {
  const op = tmapp["object_prefix"];
  regionIds.forEach((id) => {
    document.getElementById(op + id + "_tr").remove();
  });
};

/**
 * @summary Adds a region list item in the region selection menu
 * @param {*} region Region to be added to the list
 */
regionUtils.addRegionSelectionItem = function (region) {
  const op = tmapp["object_prefix"];
  let table = document.getElementById(
    "operations-regions-selection-table-" +
      HTMLElementUtils.stringToId(region.regionClass)
  );
  if (!table) {
    regionUtils.addRegionClassSelectionAccordion(region.regionClass);
    table = document.getElementById(
      "operations-regions-selection-table-" +
        HTMLElementUtils.stringToId(region.regionClass)
    );
  }
  const regionRow = HTMLElementUtils.createElement({
    kind: "tr",
    extraAttributes: {
      class: "regiontr my-2 border-bottom",
      id: op + region.id + "_tr",
    },
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
    size: 4,
    extraAttributes: {
      style: "-webkit-line-clamp: 1;",
    },
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
      size: 3,
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
    regionRow.style.background = "#E6DFF4";
    const strokeWstr =
      regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom();
    const path = d3.select(`#${region.id}_poly`);
    path.attr("stroke-width", strokeWstr * 2);
    const highLightColor = "#39FF14";
    const d3Color = d3.rgb(highLightColor);
    d3Color.opacity = 0.8;
    path.style("fill", d3Color.rgb().toString());
  };
  regionRow.onmouseout = function () {
    regionRow.style.background = "white";
    const strokeWstr =
      regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom();
    d3.select(`#${region.id}_poly`).attr("stroke-width", strokeWstr);
    if (regionUtils._selectedRegions[region.id]) return;
    region.filled
      ? regionUtils.fillRegion(region.id, true)
      : regionUtils.fillRegion(region.id, false);
  };
  table.appendChild(regionRow);
  const regionClassCounter = document.getElementById(
    "numRegionsSelection-" + HTMLElementUtils.stringToId(region.regionClass)
  );
  regionClassCounter.innerHTML = parseInt(regionClassCounter.innerHTML) + 1;
  if (parseInt(regionClassCounter.innerHTML) > 1) {
    const regionsLabel = document.getElementById(
      "numRegionsSelectionS-" + HTMLElementUtils.stringToId(region.regionClass)
    );
    regionsLabel.innerText = "s";
  }
};

/**
 * @summary Adds a region to selected list
 * @param {*} region Region to be added to selection collection
 */
regionUtils.selectRegion = function (region) {
  regionUtils._selectedRegions[region.id] = region;
};

/**
 * @summary Unchecks all region checkboxes and empties the selected regions collection
 */
regionUtils.resetSelection = function () {
  const selectedRegions = Object.values(regionUtils._selectedRegions);
  selectedRegions.forEach((region) => {
    const checkBox = document.getElementById(`${region.id}_selection_check`);
    checkBox.checked = false;
    const path = d3.select(`#${region.id}_poly`);
    region.filled
      ? regionUtils.fillRegion(region.id, true)
      : regionUtils.fillRegion(region.id, false);
  });
  regionUtils._selectedRegions = {};
};
