import { describe, expect, it } from "vitest";

import { createLabels } from "../model/labels";
import { createPoints } from "../model/points";
import { createShapes } from "../model/shapes";
import { ProjectUtils } from "./ProjectUtils";

describe("ProjectUtils", () => {
  const labels = createLabels({
    id: "labels",
    name: "Labels",
    layer: "layer",
    dataSource: { type: "tiff" },
  });
  const points = createPoints({
    id: "points",
    name: "Points",
    layer: "layer",
    dataSource: { type: "csv" },
  });
  const shapes = createShapes({
    id: "shapes",
    name: "Shapes",
    layer: "layer",
    dataSource: { type: "geojson" },
  });
  const project = { labels: [labels], points: [points], shapes: [shapes] };

  describe("getMarkerConfigs", () => {
    it("returns the marker configurations of the points", () => {
      expect(ProjectUtils.getMarkerConfigs(project)).toStrictEqual([
        points.pointMarker,
      ]);
    });
  });

  describe("getSizeConfigs", () => {
    it("returns the size configurations of the points", () => {
      expect(ProjectUtils.getSizeConfigs(project)).toStrictEqual([
        points.pointSize,
      ]);
    });
  });

  describe("getColorConfigs", () => {
    it("returns the color configurations of the labels, points and shapes", () => {
      expect(ProjectUtils.getColorConfigs(project)).toStrictEqual([
        labels.labelColor,
        points.pointColor,
        shapes.shapeFillColor,
        shapes.shapeStrokeColor,
      ]);
    });
  });

  describe("getVisibilityConfigs", () => {
    it("returns the visibility configurations of the labels, points and shapes", () => {
      expect(ProjectUtils.getVisibilityConfigs(project)).toStrictEqual([
        labels.labelVisibility,
        points.pointVisibility,
        shapes.shapeVisibility,
        shapes.shapeFillVisibility,
        shapes.shapeStrokeVisibility,
      ]);
    });
  });

  describe("getOpacityConfigs", () => {
    it("returns the opacity configurations of the labels, points and shapes", () => {
      expect(ProjectUtils.getOpacityConfigs(project)).toStrictEqual([
        labels.labelOpacity,
        points.pointOpacity,
        shapes.shapeOpacity,
        shapes.shapeFillOpacity,
        shapes.shapeStrokeOpacity,
      ]);
    });
  });
});
