import { describe, expect, it } from "vitest";

import {
  type ColorConfig,
  type RawPoints,
  createLabels,
  createLayer,
  createPoints,
  createShapes,
} from "@tissuumaps/core";

import { getActiveFrom, getColorbars } from "./getColorbars";

const fromColor: ColorConfig = {
  from: { column: "score", palette: "viridis" },
};

const constantColor: ColorConfig = {
  constant: { value: { r: 255, g: 0, b: 0 } },
};

const layers = [
  createLayer({ id: "shown", name: "Shown" }),
  createLayer({ id: "hidden", name: "Hidden", visibility: false }),
];

function createPointsWith(id: string, raw: Partial<RawPoints> = {}) {
  return createPoints({
    id,
    name: id,
    layer: "shown",
    dataSource: { type: "csv", table: "cells" },
    pointColor: fromColor,
    ...raw,
  });
}

function getColorbarKeys(state: Partial<Parameters<typeof getColorbars>[0]>) {
  return getColorbars({
    layers,
    labels: [],
    points: [],
    shapes: [],
    ...state,
  }).map(({ key }) => key);
}

describe("getActiveFrom", () => {
  it("returns the active from source", () => {
    expect(getActiveFrom(fromColor)).toEqual(fromColor.from);
  });

  it("returns null while another source is active", () => {
    expect(getActiveFrom(constantColor)).toBeNull();
    expect(
      getActiveFrom({ ...fromColor, ...constantColor, source: "constant" }),
    ).toBeNull();
  });
});

describe("getColorbars", () => {
  it("lists the objects whose colors come from a table column", () => {
    expect(
      getColorbarKeys({
        points: [
          createPointsWith("from"),
          createPointsWith("constant", { pointColor: constantColor }),
        ],
      }),
    ).toEqual(["points-from"]);
  });

  it("leaves out hidden objects and objects on hidden layers", () => {
    expect(
      getColorbarKeys({
        points: [
          createPointsWith("hidden", { visibility: false }),
          createPointsWith("on-hidden-layer", { layer: "hidden" }),
          createPointsWith("layer-from-column", {
            layer: { column: "layer" },
          }),
        ],
      }),
    ).toEqual(["points-layer-from-column"]);
  });

  it("leaves out only a constantly hidden visibility", () => {
    expect(
      getColorbarKeys({
        points: [
          createPointsWith("constant", {
            pointVisibility: { constant: { value: false } },
          }),
          createPointsWith("from-column", {
            pointVisibility: { from: { column: "shown" } },
          }),
        ],
      }),
    ).toEqual(["points-from-column"]);
  });

  it("leaves out labels without a table", () => {
    expect(
      getColorbarKeys({
        labels: [
          createLabels({
            id: "with-table",
            name: "With table",
            layer: "shown",
            dataSource: { type: "tiff", table: "cells" },
            labelColor: fromColor,
          }),
          createLabels({
            id: "without-table",
            name: "Without table",
            layer: "shown",
            dataSource: { type: "tiff" },
            labelColor: fromColor,
          }),
        ],
      }),
    ).toEqual(["labels-with-table"]);
  });

  it("lists the fill and the stroke of shapes", () => {
    const shapes = {
      name: "Shapes",
      layer: "shown",
      dataSource: { type: "geojson", table: "cells" },
      shapeFillColor: fromColor,
      shapeStrokeColor: fromColor,
    };
    expect(
      getColorbarKeys({
        shapes: [
          createShapes({ id: "shown", ...shapes }),
          createShapes({
            id: "hidden",
            ...shapes,
            shapeVisibility: { constant: { value: false } },
          }),
        ],
      }),
    ).toEqual(["shapes-shown-fill", "shapes-shown-stroke"]);
  });
});
