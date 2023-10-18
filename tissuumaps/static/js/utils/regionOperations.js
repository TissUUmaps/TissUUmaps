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
    regionUtils.deleteRegion(id, true);
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
 * Run clipper binary operation on a set of regions
 */
regionUtils.clipperPolygons = function (polygons, operation) {
  let subject = null;
  let capitalConversion = "x" in polygons[0][0][0][0];
  for (let i = 0; i < polygons.length; i++) {
      if (subject == null) {
        subject = new clipperShape (polygons[i].flat(), closed = true, capitalConversion = capitalConversion, integerConversion = false, removeDuplicates = false)
      }
      else {
        let path2 = new clipperShape (polygons[i].flat(), closed = true, capitalConversion = capitalConversion, integerConversion = false, removeDuplicates = false)
        subject =  subject[operation](path2);
      }
  }
  let separatePaths = subject.separateShapes().map((path) => path.paths);
  return separatePaths;
}

/**
 * Run clipper binary operation on a set of regions
 */
regionUtils.clipperRegions = function (regions, operation) {
  const polygons = regions.map((region) => region.globalPoints);
  const solution_paths = regionUtils.clipperPolygons(polygons, operation);
  return regionUtils.regionToLowerCase(solution_paths);
}


/**
 * @summary Generates a region that covers the instersection of the passed regions
 * @param {*} regions Array of regions to calculate the intersection
 */
regionUtils.regionsClipper = function (regions, operation) {
  if (regions.length < 2) {
    interfaceUtils.alert("Please select at least two regions");
    return;
  }
  try {
    const intersectionPoints = regionUtils.clipperRegions(regions, operation);
    // TODO: Check that all regions have the same collection index. We can not intersect regions from different layers.
    let mainRegion = regions.shift();
    mainRegion.globalPoints = intersectionPoints;
    regionUtils.updateBbox(mainRegion);

    regions.forEach((region) => {
      regionUtils.deleteRegion(region.id, true);
    });
    regionUtils.updateAllRegionClassUI();
  } catch (error){
    console.log(error);
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
 * @summary Resizes a multipolygon maintaing the center of each of its polygons and
 * and updates the corresponding region's path
 * @param {*} regionId Id of the region being rescaled
 * @param {*} scale Scale factor to use in rescaling
 */
regionUtils.resizeRegion = function (regionId, scale, preview) {
  const escapedRegionId = HTMLElementUtils.stringToId(regionId);
  const region = regionUtils._regions[regionId];
  const scaleFactor = scale / 100;
  let globalPoints = region.globalPoints;
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
            (region.scale
              ? region.scale / 100
              : 1);
        point[1] =
          centroidBefore[1] +
          ((point[1] - centroidBefore[1]) * scaleFactor) /
            (region.scale
              ? region.scale / 100
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
  const newGlobalPoints = regionUtils.arrayToObjectPoints(points);
  d3.select("#" + escapedRegionId + "preview" + "_poly").remove();
  if (preview) {
    regionUtils.drawRegionPath(
      regionUtils.globalPointsToViewportPoints(newGlobalPoints, region.collectionIndex),
      escapedRegionId + "preview",
      null, "#ffffff99"
    );
  }
  else {
    // Save new region scale and points
    region.scale = scale;
    // TODO - replace .points with globalPoints?
    region.globalPoints = newGlobalPoints;

    regionUtils.deSelectRegion(region.id);
    regionUtils.selectRegion(region);
    regionUtils.updateBbox(region);
  }
  
  
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

regionUtils.dilateRegion = function (regionId, offset, preview, onlyBorder) {
  if (!offset) return;
  const options = {
    jointType : 'jtRound',
    endType : 'etClosedPolygon',
    miterLimit : 2.0,
    roundPrecision : 0.25
  }
  const region = regionUtils._regions[regionId];
  const path = new clipperShape (region.globalPoints.flat(), closed = true, capitalConversion = true, integerConversion = false, removeDuplicates = false)
  let pathOut = path.offset( offset, options);
  let dilatedPoints = null;
  if (onlyBorder) {
    pathOut = pathOut.xor(path);
  }
  pathOut = pathOut.separateShapes().map((path) => path.paths);
  dilatedPoints = regionUtils.regionToLowerCase(pathOut);

  const escapedRegionId = HTMLElementUtils.stringToId(regionId);
  d3.select("#" + escapedRegionId + "preview" + "_poly").remove();
  if (preview) {
    regionUtils.drawRegionPath(
      regionUtils.globalPointsToViewportPoints(dilatedPoints, region.collectionIndex),
      escapedRegionId + "preview"
    );
  }
  else {
    region.globalPoints = dilatedPoints;
    regionUtils.updateBbox(region);
    
    regionUtils.deSelectRegion(region.id);
    regionUtils.selectRegion(region);
  }
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
 * @summary Converts clipper-js with upper case X and Y to lower cas x and y for GeoJSON
 * @param {*} points  
 * @returns 
 */
regionUtils.regionToLowerCase = function (points) {
  return points.map((arr) =>
    arr.map((secondArr) =>
      secondArr.map((coordinates) => {
        return {
          x: coordinates.X,
          y: coordinates.Y,
        };
      })
    )
  );
};

/**
 * @summary Converts GeoJSON with lower cas x and y to upper case X and Y for clipper-js
 * @param {*} points  
 * @returns 
 */
regionUtils.regionToUpperCase = function (points) {
  return points.map((arr) =>
    arr.map((secondArr) =>
      secondArr.map((coordinates) => {
        return {
          X: coordinates.x,
          Y: coordinates.y,
        };
      })
    )
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
 * @summary Adds a region to selected list
 * @param {*} region Region to be added to selection collection
 */
regionUtils.selectRegion = function (region, skipHighlight) {
  const escapedRegionId = HTMLElementUtils.stringToId(region.id);
  const regionClassID = HTMLElementUtils.stringToId("region_" + region.regionClass);

  regionUtils._selectedRegions[region.id] = region;
  let points = regionUtils.globalPointsToViewportPoints(region.globalPoints, region.collectionIndex);
  regionUtils.drawRegionPath(points, escapedRegionId + "_selected", "#0165fc")
  const checkBox = document.getElementById(`${escapedRegionId}_selection_check`);
  if (checkBox) checkBox.checked = true;
  if (!skipHighlight) {regionUtils.highlightRegion(region.id)};
  regionUtils.addRegionToolbarUI();
};

/**
 * @summary Highlight and scroll to a region in the right panel
 * @param {*} regionid ID of region to be highlighted
 */
regionUtils.highlightRegion = async function (regionid) {
    // wait 50 ms
    await new Promise(r => setTimeout(r, 50));
    // Highlight region in right panel
    const region = regionUtils._regions[regionid];
    const escapedRegionId = HTMLElementUtils.stringToId(region.id);
    const regionClassID = HTMLElementUtils.stringToId("region_" + region.regionClass);
    const collapsibleRow = $(`#collapse_region_${regionClassID}`);
    collapsibleRow[0].setAttribute("data-region-selected", escapedRegionId);
    if (collapsibleRow) {
      collapsibleRow.collapse("show");
      // trigger show.bs.collapse on the collapsible row to highlight the region
      collapsibleRow.trigger("show.bs.collapse");
    }
    setTimeout(() => {
        var tr = document.querySelectorAll('[data-escapedid="'+escapedRegionId+'"]')[0];
        if (tr != null) {
            tr.scrollIntoView({block: "nearest",inline: "nearest"});
            tr.classList.remove("transition_background")
            tr.classList.add("table-primary")
            setTimeout(function(){tr.classList.add("transition_background");tr.classList.remove("table-primary");},400);
        }
    },200)
}

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
