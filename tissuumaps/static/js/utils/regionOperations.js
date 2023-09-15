regionUtils._selectedRegions = {};

/**
 * @summary Toggles the region toolbar menu on and off
 */
regionUtils.regionToolbarOnOff = function () {
  // Add region Toolbar
  regionUtils.addRegionToolbarUI();
  const op = tmapp["object_prefix"];
  
  if (!overlayUtils._regionToolbar) {
    regionUtils.resetSelection();
    if (overlayUtils._drawRegions) {
        regionUtils.regionsOnOff();
    }
    if (overlayUtils._freeHandDrawRegions) {
        regionUtils.freeHandRegionsOnOff();
    }
    if (overlayUtils._brushDrawRegions) {
        regionUtils.brushRegionsOnOff();
    }
    if (overlayUtils._regionSelection) {
        regionUtils.selectRegionsOnOff();
    }
  }
}

/**
 * @summary Draws a path given a set of points and an Id
 * @param {*} points Region points used to construct the path
 * @param {*} regionId Id of the region
 */
regionUtils.drawRegionPath = function (
  points,
  regionId,
  borderColor,
  fillColor,
  strokeDasharray
) {
  const escapedRegionId = HTMLElementUtils.stringToId(regionId);
  const canvasNode =
    overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
  const canvas = d3.select(canvasNode);
  const strokeWstr =
    2.5 * regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom();
  canvas
    .append("path")
    .attr("d", regionUtils.pointsToPath(points))
    .attr("id", escapedRegionId + "_poly")
    .attr("class", "regionpoly region_previewpoly")
    .attr("stroke-width", strokeWstr)
    .attr("stroke-dasharray", strokeDasharray ? strokeDasharray : "none")
    .style("stroke", borderColor ? borderColor : "#FF0000")
    .style("fill", fillColor ? fillColor : "none")
    .append("title")
    .text(regionId)
    .attr("id", "path-title-" + escapedRegionId);
};

/**
 * @summary Deletes regions
 * @param {*} regions Array of regions to be deleted
 */
regionUtils.deleteRegions = function (regionIds) {
  if (regionIds.length < 1) {
    interfaceUtils.alert("Please select at least one region");
    return;
  }
  regionIds.forEach((id) => {
    regionUtils.deleteRegion(id);
  });
  regionUtils.updateAllRegionClassUI();
  glUtils.updateRegionDataTextures();
  glUtils.updateRegionLUTTextures();
  glUtils.draw();
};

/**
 * @summary Split regions
 * @param {*} regions Array of regions to be deleted
 */
regionUtils.splitRegions = function (regionIds) {
  if (regionIds.length < 1) {
    interfaceUtils.alert("Please select at least one region");
    return;
  }
  regionIds.forEach((id) => {
    regionUtils.splitRegion(id);
  });
  regionUtils.updateAllRegionClassUI();
  glUtils.updateRegionDataTextures();
  glUtils.updateRegionLUTTextures();
  glUtils.draw();
};

/**
 * @summary Fill holes in regions
 * @param {*} regions Array of regions to be deleted
 */
regionUtils.fillHolesRegions = function (regionIds) {
  if (regionIds.length < 1) {
    interfaceUtils.alert("Please select at least one region");
    return;
  }
  regionIds.forEach((id) => {
    regionUtils.fillHolesRegion(id);
  });
  glUtils.updateRegionDataTextures();
  glUtils.updateRegionLUTTextures();
  glUtils.draw();
};

/**
 * @summary Duplicates regions
 * @param {*} regions Array of regions to be duplicated
 */
regionUtils.duplicateRegions = function (regions) {
  if (regions.length < 1) {
    interfaceUtils.alert("Please select at least one region");
    return;
  }
  regions.forEach((region) => {
    const newRegionId = "region" + (regionUtils._currentRegionId + 1);
    regionUtils._currentRegionId++;
    let viewportPoints = regionUtils.globalPointsToViewportPoints(region.globalPoints, region.collectionIndex);
    regionUtils.addRegion(
      regionUtils.objectToArrayPoints(viewportPoints),
      newRegionId,
      region.polycolor,
      region.regionClass,
      region.collectionIndex
    );
  });
  regionUtils.updateAllRegionClassUI();
  glUtils.updateRegionDataTextures();
  glUtils.updateRegionLUTTextures();
  glUtils.draw();
};

/**
 * @summary Generates a region that covers the instersection of the passed regions
 * @param {*} regions Array of regions to calculate the intersection
 */
regionUtils.regionsIntersection = function (regions) {
  if (regions.length < 2) {
    interfaceUtils.alert("Please select at least two regions");
    return;
  }
  try {
    const intersectionPoints = polygonClipping.intersection(
      ...regions.map((region) => {
        let viewportPoints = regionUtils.globalPointsToViewportPoints(region.globalPoints, region.collectionIndex);
        return regionUtils.objectToArrayPoints(viewportPoints)
      })
    );
    // TODO: Check that all regions have the same collection index. We can not intersect regions from different layers.
    const newRegionLayerIndex = regions[0].collectionIndex;
    regionUtils._currentRegionId += 1;
    const newRegionId = "region" + regionUtils._currentRegionId;
    regionUtils.addRegion(intersectionPoints, newRegionId, regions[0].polycolor, "", newRegionLayerIndex);
    regionUtils.updateAllRegionClassUI();
    regions.forEach((region) => {
      regionUtils.deleteRegion(region.id);
    });
  } catch {
    interfaceUtils.alert(
      "The selected regions have no interception between them"
    );
  }
  regionUtils.resetSelection();
  glUtils.updateRegionDataTextures();
  glUtils.updateRegionLUTTextures();
  glUtils.draw();
};

/**
 * @summary Generates a region that is the difference between the selected regions
 * @param {*} regions Array of regions to calculate the difference
 */
regionUtils.regionsDifference = function (regions) {
  if (regions.length < 2) {
    interfaceUtils.alert("Please select at least two regions");
    return;
  }
  const differencePoints = polygonClipping.xor(
    ...regions.map((region) => {
      let viewportPoints = regionUtils.globalPointsToViewportPoints(region.globalPoints, region.collectionIndex);
      return regionUtils.objectToArrayPoints(viewportPoints)
    })
  );
  // TODO: Check that all regions have the same collection index. We can not intersect regions from different layers.
  const newRegionLayerIndex = regions[0].collectionIndex;
  regionUtils._currentRegionId += 1;
  const newRegionId = "region" + regionUtils._currentRegionId;
  regionUtils.addRegion(differencePoints, newRegionId, regions[0].polycolor, "", newRegionLayerIndex);
  regionUtils.updateAllRegionClassUI();
  regions.forEach((region) => {
    regionUtils.deleteRegion(region.id);
  });
  regionUtils.resetSelection();
  glUtils.updateRegionDataTextures();
  glUtils.updateRegionLUTTextures();
  glUtils.draw();
};

/**
 * @summary Resizes a multipolygon maintaing the center of each of its polygons and
 * and updates the corresponding region's path
 * @param {*} regionId Id of the region being rescaled
 * @param {*} scale Scale factor to use in rescaling
 */
regionUtils.resizeRegion = function (regionId, scale, preview) {
  const scaleFactor = scale / 100;
  let globalPoints = regionUtils._regions[regionId].globalPoints;
  const points = regionUtils.objectToArrayPoints(
    globalPoints
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
  // TODO - replace .points with globalPoints?
  regionUtils._regions[regionId].globalPoints =
    regionUtils.arrayToObjectPoints(points);
  regionUtils.updateBbox(region);
  
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
  
  glUtils.updateRegionDataTextures();
  glUtils.updateRegionLUTTextures();
  glUtils.draw();
};

regionUtils.dilateRegion = function (regionId, offset, preview, onlyBorder) {
  if (!offset) return;
  const region = regionUtils._regions[regionId];
  const worker = new Worker("static/js/utils/regionOffsetWorker.js");
  const point1 = turf.point([0, 0]);
  const point2 = turf.point([1, 0]);
  const distance = turf.distance(point1, point2, {
    units: "kilometers",
  });
  const offsetScaled =
    (offset / OSDViewerUtils.getImageWidth()) *
    distance;
  let viewportPoints = regionUtils.globalPointsToViewportPoints(
    region.globalPoints,
    region.collectionIndex
  );
  worker.postMessage([viewportPoints, offsetScaled]);
  worker.onmessage = function (event) {
    let dilatedPoints = event.data;
    if (!event.data) {
      interfaceUtils.alert(
        "An error ocurred applying the selected offset amount, for negative offsets, please make sure that the region is big enough to be offseted by that amount"
      );
      button.disabled = false;
      button.innerHTML = "Type";
      return;
    }
    if (onlyBorder) {
      dilatedPoints = polygonClipping.xor(
        regionUtils.objectToArrayPoints(viewportPoints),
        dilatedPoints
      );
    }
    d3.select("#" + region.id + "preview" + "_poly").remove();
    if (preview) {
      regionUtils.drawRegionPath(
        regionUtils.arrayToObjectPoints(dilatedPoints),
        region.id + "preview"
      );
    }
    else {
      region.globalPoints = regionUtils.viewportPointsToGlobalPoints(
        regionUtils.arrayToObjectPoints(dilatedPoints),
        region.collectionIndex
      );
      regionUtils.updateBbox(region);
      
      regionUtils.deSelectRegion(region.id);
      regionUtils.selectRegion(region);
      glUtils.updateRegionDataTextures();
      glUtils.updateRegionLUTTextures();
      glUtils.draw();
    }
  };
}

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
  if (regions.length < 2) {
    interfaceUtils.alert("Please select at least two regions");
    return;
  }
  const mergedPoints = polygonClipping.union(
    ...regions.map((region) => {
      let viewportPoints = regionUtils.globalPointsToViewportPoints(region.globalPoints, region.collectionIndex);
      return regionUtils.objectToArrayPoints(viewportPoints)
    })
  );
  // TODO: Check that all regions have the same collection index. We can not intersect regions from different layers.
  const newRegionLayerIndex = regions[0].collectionIndex;
  regionUtils._currentRegionId += 1;
  const newRegionId = "region" + regionUtils._currentRegionId;
  regionUtils.addRegion(mergedPoints, newRegionId, regions[0].polycolor, "", newRegionLayerIndex);
  regionUtils.updateAllRegionClassUI();
  regions.forEach((region) => {
    regionUtils.deleteRegion(region.id);
  });
  regionUtils.resetSelection();
  glUtils.updateRegionDataTextures();
  glUtils.updateRegionLUTTextures();
  glUtils.draw();
};

/**
 * @summary Adds a region to selected list
 * @param {*} region Region to be added to selection collection
 */
regionUtils.selectRegion = function (region) {
  const escapedRegionId = HTMLElementUtils.stringToId(region.id);
  const regionClassID = HTMLElementUtils.stringToId("region_" + region.regionClass);

  regionUtils._selectedRegions[region.id] = region;
  let points = regionUtils.globalPointsToViewportPoints(region.globalPoints, region.collectionIndex);
  regionUtils.drawRegionPath(points, escapedRegionId + "_selected", "#0165fc")
  const checkBox = document.getElementById(`${escapedRegionId}_selection_check`);
  if (checkBox) checkBox.checked = true;
  const collapsibleRow = $(`#collapse_region_${regionClassID}`);
  if (collapsibleRow) collapsibleRow.collapse("show");
  setTimeout(() => {
    var tr = document.querySelectorAll('[data-escapedid="'+escapedRegionId+'"]')[0];
    if (tr != null) {
        tr.scrollIntoView({block: "nearest",inline: "nearest"});
        tr.classList.remove("transition_background")
        tr.classList.add("table-primary")
        setTimeout(function(){tr.classList.add("transition_background");tr.classList.remove("table-primary");},400);
    }
  },100)
  
  regionUtils.addRegionToolbarUI();
};

/**
 * @summary Removes a region to selected list
 * @param {*} region Region to be removed from selection collection
 */
regionUtils.deSelectRegion = function (regionId) {
    const escapedRegionId = HTMLElementUtils.stringToId(regionId);
    delete regionUtils._selectedRegions[regionId];
    d3.select("#" + escapedRegionId + "_selected" + "_poly").remove();
    const checkBox = document.getElementById(`${escapedRegionId}_selection_check`);
    if (checkBox) checkBox.checked = false;
    regionUtils.addRegionToolbarUI();
};

/**
 * @summary Unchecks all region checkboxes and empties the selected regions collection
 */
regionUtils.resetSelection = function () {
  const selectedRegions = Object.values(regionUtils._selectedRegions);
  selectedRegions.forEach((region) => {
    const escapedRegionId = HTMLElementUtils.stringToId(region.id);
    const checkBox = document.getElementById(`${escapedRegionId}_selection_check`);
    if (checkBox) checkBox.checked = false;
    d3.select("#" + escapedRegionId + "_selected" + "_poly").remove();
  });
  regionUtils._selectedRegions = {};
  regionUtils.addRegionToolbarUI();
};
