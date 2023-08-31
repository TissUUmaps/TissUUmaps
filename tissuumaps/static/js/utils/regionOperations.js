regionUtils._selectedRegions = {};

regionUtils.regionOperationsOnOff = function () {
  overlayUtils._regionOperations = !overlayUtils._regionOperations;
  const op = tmapp["object_prefix"];
  const operationsRegionButtonIcon = document.getElementById(
    op + "_operations_regions_icon"
  );
  const regionAccordions = document.getElementById("markers-regions-panel");
  const regionOpertationsList = document.getElementById(
    "region-operations-panel"
  );
  if (overlayUtils._regionOperations) {
    operationsRegionButtonIcon.classList.remove("bi-circle");
    operationsRegionButtonIcon.classList.add("bi-check-circle");
    regionUtils.showHint("Select regions");
    // Hide region accordions
    regionAccordions.style.display = "none";
    // Show region selection list
    regionOpertationsList.classList.remove("d-none");
    // Add region selection UI
    regionUtils.addRegionOperationsUI();
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
 * @param {String} regionid String id of region to fill
 * @summary Given a region id, fill this region in the interface */
regionUtils.fillRegion = function (regionid, value) {
  // if (value === undefined) {
  //     // we toggle
  //     if(regionUtils._regions[regionid].filled === 'undefined'){
  //         value = true;
  //     }
  //     else {
  //         value = !regionUtils._regions[regionid].filled;
  //     }
  // }
  // regionUtils._regions[regionid].filled=value;
  // if (!glUtils._showRegionsExperimental) {
  //     var newregioncolor = regionUtils._regions[regionid].polycolor;
  //     var d3color = d3.rgb(newregioncolor);
  //     var newStyle="";
  //     if(regionUtils._regions[regionid].filled){
  //         newStyle = "stroke: " + d3color.rgb().toString()+";";
  //         d3color.opacity=0.5;
  //         newStyle +="fill: "+d3color.rgb().toString()+";";
  //     }else{
  //         newStyle = "stroke: " + d3color.rgb().toString() + "; fill: none;";
  //     }
  //     document.getElementById(regionid + "_poly").setAttribute("style", newStyle);
  // }
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
  regionUtils.updateRegionOperationsListUI();
};

/**
 * @summary Duplicates regions
 * @param {*} regions Array of regions to be duplicated
 */
regionUtils.duplicateRegions = function (regions) {
  Object.values(regionUtils._regions).forEach((region) => {
    const newRegionId = region.id + "duplicate";
    const hexColor = overlayUtils.randomColor("hex");
    regionUtils.addRegion(
      regionUtils.objectToArrayPoints(region.points),
      newRegionId,
      hexColor,
      region.regionClass,
      region.scale
    );
    //regionUtils.drawRegionPath(region.points, newRegionId);
    //regionUtils.addRegionSelectionItem(regionUtils._regions[newRegionId]);
  });
  regionUtils.updateAllRegionClassUI();
  regionUtils.updateRegionOperationsListUI();
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
    // regionUtils.drawRegionPath(
    //   regionUtils.arrayToObjectPoints(intersectionPoints),
    //   newRegionId
    // );
    regionUtils.updateAllRegionClassUI();
    regionUtils.updateRegionOperationsListUI();
    // regionUtils.addRegionSelectionItem(regionUtils._regions[newRegionId]);
  } catch {
    interfaceUtils.alert(
      "The selected regions have no interception between them"
    );
  }
  glUtils.updateRegionDataTextures();
  glUtils.updateRegionLUTTextures();
  glUtils.draw();
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
  //   regionUtils.drawRegionPath(
  //     regionUtils.arrayToObjectPoints(differencePoints),
  //     newRegionId
  //   );
  regionUtils.updateAllRegionClassUI();
  regionUtils.updateRegionOperationsListUI();
  //regionUtils.addRegionSelectionItem(regionUtils._regions[newRegionId]);
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

  regionUtils._regions[regionId] = regionUtils.updateRegionCoordinates(
    regionUtils._regions[regionId]
  );

  glUtils.updateRegionDataTextures();
  glUtils.updateRegionLUTTextures();
  glUtils.draw();
};

regionUtils.updateRegionCoordinates = function (region) {
  const points = regionUtils.objectToArrayPoints(region.points);
  const newPoints = [];
  const newGlobalPoints = [];
  const viewer = tmapp[tmapp["object_prefix"] + "_viewer"];
  let _xmin = parseFloat(points[0][0][0][0]),
    _xmax = parseFloat(points[0][0][0][0]),
    _ymin = parseFloat(points[0][0][0][1]),
    _ymax = parseFloat(points[0][0][0][1]);
  for (let i = 0; i < points.length; i++) {
    subregion = [];
    globalSubregion = [];
    for (let j = 0; j < points[i].length; j++) {
      polygon = [];
      globalPolygon = [];
      for (let k = 0; k < points[i][j].length; k++) {
        let x = parseFloat(points[i][j][k][0]);
        let y = parseFloat(points[i][j][k][1]);

        if (x > _xmax) _xmax = x;
        if (x < _xmin) _xmin = x;
        if (y > _ymax) _ymax = y;
        if (y < _ymin) _ymin = y;
        polygon.push({ x: x, y: y });
        let tiledImage = viewer.world.getItemAt(0);
        let imageCoord = tiledImage.viewportToImageCoordinates(x, y, true);
        globalPolygon.push({ x: imageCoord.x, y: imageCoord.y });
      }
      subregion.push(polygon);
      globalSubregion.push(globalPolygon);
    }
    newPoints.push(subregion);
    newGlobalPoints.push(globalSubregion);
  }
  (region._xmin = _xmin),
    (region._xmax = _xmax),
    (region._ymin = _ymin),
    (region._ymax = _ymax);
  const tiledImage = viewer.world.getItemAt(0);
  const _min_imageCoord = tiledImage.viewportToImageCoordinates(_xmin, _ymin);
  const _max_imageCoord = tiledImage.viewportToImageCoordinates(_xmax, _ymax);
  (region._gxmin = _min_imageCoord.x),
    (region._gxmax = _max_imageCoord.x),
    (region._gymin = _min_imageCoord.y),
    (region._gymax = _max_imageCoord.y);
  region.points = newPoints;
  region.globalPoints = newGlobalPoints;
  return region;
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
  //   regionUtils.drawRegionPath(
  //     regionUtils.arrayToObjectPoints(points),
  //     newRegionId
  //   );
  //regionUtils.addRegionSelectionItem(regionUtils._regions[newRegionId]);
  regionUtils.updateRegionOperationsListUI();
  glUtils.updateRegionDataTextures();
  glUtils.updateRegionLUTTextures();
  glUtils.draw();
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
  //   const canvasNode =
  //     overlayUtils._d3nodes[tmapp["object_prefix"] + "_regions_svgnode"].node();
  //   const canvas = d3.select(canvasNode);
  //   const strokeWstr =
  //     regionUtils._polygonStrokeWidth / tmapp["ISS_viewer"].viewport.getZoom();
  //   canvas
  //     .append("path")
  //     .attr(
  //       "d",
  //       regionUtils.pointsToPath(regionUtils.arrayToObjectPoints(mergedPoints))
  //     )
  //     .attr("id", newRegionId + "_poly")
  //     .attr("class", "regionpoly")
  //     .attr("polycolor", "#FF0000")
  //     .attr("stroke-width", strokeWstr)
  //     .style("stroke", "#FF0000")
  //     .style("fill", "none")
  //     .append("title")
  //     .text(newRegionId)
  //     .attr("id", "path-title-" + newRegionId);
  regionUtils.updateAllRegionClassUI();
  regionUtils.updateRegionOperationsListUI();
  // regionUtils.addRegionSelectionItem(regionUtils._regions[newRegionId]);
  glUtils.updateRegionDataTextures();
  glUtils.updateRegionLUTTextures();
  glUtils.draw();
};

/**
 * @summary Deletes the selection items of the regions corresponding to the ids
 * @param {*} regionIds Ids of the regions that will get their selection item removed
 */
regionUtils.deleteRegionSelectionItems = function (regionIds) {
  regionUtils.updateAllRegionClassUI();
  regionUtils.updateRegionOperationsListUI();
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
