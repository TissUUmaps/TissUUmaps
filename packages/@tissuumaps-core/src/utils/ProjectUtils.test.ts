import { describe, expect, it } from "vitest";

import { createImage } from "../model/image";
import { createLabels } from "../model/labels";
import { createLayer } from "../model/layer";
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
  const layers = [
    createLayer({ id: "shown", name: "Shown" }),
    createLayer({ id: "hidden", name: "Hidden", visibility: false }),
  ];

  describe("hasData", () => {
    const empty = {
      images: [],
      labels: [],
      points: [],
      shapes: [],
      tables: [],
    };

    it("returns false without data objects", () => {
      expect(ProjectUtils.hasData(empty)).toBe(false);
    });

    it("returns true with a data object", () => {
      expect(ProjectUtils.hasData({ ...empty, points: [points] })).toBe(true);
    });
  });

  describe("isObjectVisible", () => {
    it("returns true for a visible object on a visible layer", () => {
      expect(
        ProjectUtils.isObjectVisible(
          { visibility: true, layer: "shown" },
          layers,
        ),
      ).toBe(true);
    });

    it("returns false for a hidden object", () => {
      expect(
        ProjectUtils.isObjectVisible(
          { visibility: false, layer: "shown" },
          layers,
        ),
      ).toBe(false);
    });

    it("returns false for an object on a hidden layer", () => {
      expect(
        ProjectUtils.isObjectVisible(
          { visibility: true, layer: "hidden" },
          layers,
        ),
      ).toBe(false);
    });

    it("checks only the object for layers from a table column", () => {
      const layer = { column: "layer" };
      expect(
        ProjectUtils.isObjectVisible({ visibility: true, layer }, layers),
      ).toBe(true);
      expect(
        ProjectUtils.isObjectVisible({ visibility: false, layer }, layers),
      ).toBe(false);
    });
  });

  describe("getLoadedObjectIds", () => {
    const images = [
      createImage({
        id: "visible",
        name: "Visible",
        layer: "shown",
        dataSource: { type: "tiff" },
      }),
      createImage({
        id: "hidden",
        name: "Hidden",
        layer: "shown",
        visibility: false,
        dataSource: { type: "tiff" },
      }),
      createImage({
        id: "on-hidden-layer",
        name: "On hidden layer",
        layer: "hidden",
        dataSource: { type: "tiff" },
      }),
    ];

    it("adds the visible objects", () => {
      expect(ProjectUtils.getLoadedObjectIds(images, layers, [])).toEqual([
        "visible",
      ]);
    });

    it("keeps the loaded hidden objects, in project order", () => {
      expect(
        ProjectUtils.getLoadedObjectIds(images, layers, [
          "on-hidden-layer",
          "hidden",
        ]),
      ).toEqual(["visible", "hidden", "on-hidden-layer"]);
    });

    it("drops the IDs that do not identify an object", () => {
      expect(
        ProjectUtils.getLoadedObjectIds(images, layers, ["visible", "deleted"]),
      ).toEqual(["visible"]);
    });

    it("returns the given IDs if they are the same objects", () => {
      const loadedObjectIds = ["hidden", "visible"];
      expect(
        ProjectUtils.getLoadedObjectIds(images, layers, loadedObjectIds),
      ).toBe(loadedObjectIds);
    });
  });

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
