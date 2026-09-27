import { describe, expect, it } from "vitest";

import {
  type HighlightedItemGroup,
  createLabels,
  createPoints,
  createShapes,
} from "@tissuumaps/core";

import {
  type HighlightableState,
  highlightItemGroup,
} from "./highlightItemGroup";

describe("highlightItemGroup", () => {
  // the labels and the shapes share their ID and their table
  const state: HighlightableState = {
    labels: [
      createLabels({
        id: "cells",
        name: "Labels",
        layer: "layer",
        dataSource: { type: "tiff", table: "cells" },
      }),
    ],
    points: [
      createPoints({
        id: "points",
        name: "Points",
        layer: "layer",
        dataSource: { type: "csv", table: "cells" },
      }),
      createPoints({
        id: "other",
        name: "Other points",
        layer: "layer",
        dataSource: { type: "csv", table: "cells" },
      }),
    ],
    shapes: [
      createShapes({
        id: "cells",
        name: "Shapes",
        layer: "layer",
        dataSource: { type: "geojson", table: "cells" },
        shapeStrokeVisibility: { constant: { value: false } },
      }),
    ],
    opacityMaps: [
      { id: "highlightedItemGroup", name: "Project map", values: { A: 0 } },
    ],
  };
  const highlightedShapesGroup: HighlightedItemGroup = {
    annotatedObject: { shapesId: "cells" },
    column: "cluster",
    group: "A",
  };
  const opacityConfig = {
    groupBy: { column: "cluster", map: "highlightedItemGroup" },
  };
  const visibilityConfig = { constant: { value: true } };

  it("returns the state itself without a highlighted group", () => {
    expect(highlightItemGroup(state, null)).toBe(state);
  });

  it("shows only the highlighted group of its object", () => {
    const highlightedState = highlightItemGroup(state, highlightedShapesGroup);

    expect(highlightedState.shapes[0]!.shapeOpacity).toEqual(opacityConfig);
    expect(highlightedState.opacityMaps[0]).toEqual({
      id: "highlightedItemGroup",
      name: "Highlighted group",
      values: { A: 1 },
      default: 0,
    });
  });

  it("leaves the other objects on the same table untouched", () => {
    const highlightedState = highlightItemGroup(state, highlightedShapesGroup);

    expect(highlightedState.labels).toBe(state.labels);
    expect(highlightedState.points).toBe(state.points);
  });

  it("leaves the other objects of its collection untouched", () => {
    const highlightedState = highlightItemGroup(state, {
      ...highlightedShapesGroup,
      annotatedObject: { pointsId: "points" },
    });

    expect(highlightedState.points[0]!.pointOpacity).toEqual(opacityConfig);
    expect(highlightedState.points[1]).toBe(state.points[1]);
  });

  it("overrides labels", () => {
    const highlightedState = highlightItemGroup(state, {
      ...highlightedShapesGroup,
      annotatedObject: { labelsId: "cells" },
    });

    expect(highlightedState.labels[0]!.labelOpacity).toEqual(opacityConfig);
    expect(highlightedState.labels[0]!.labelVisibility).toEqual(
      visibilityConfig,
    );
    expect(highlightedState.shapes).toBe(state.shapes);
  });

  it("puts its map before the project maps, which it keeps", () => {
    const highlightedState = highlightItemGroup(state, highlightedShapesGroup);

    expect(highlightedState.opacityMaps.slice(1)).toEqual(state.opacityMaps);
  });

  it("shows the highlighted group even if it is hidden", () => {
    const highlightedState = highlightItemGroup(state, highlightedShapesGroup);

    expect(highlightedState.shapes[0]!.shapeVisibility).toEqual(
      visibilityConfig,
    );
  });

  it("keeps the fill and stroke settings of shapes", () => {
    const highlightedState = highlightItemGroup(state, highlightedShapesGroup);

    expect(highlightedState.shapes[0]!.shapeStrokeVisibility).toEqual({
      constant: { value: false },
    });
    expect(highlightedState.shapes[0]!.shapeFillOpacity).toEqual(
      state.shapes[0]!.shapeFillOpacity,
    );
  });
});
