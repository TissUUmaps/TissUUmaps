import { describe, expect, it } from "vitest";

import {
  type ColorConfig,
  type RawPoints,
  createLabels,
  createLayer,
  createPoints,
  createShapes,
} from "@tissuumaps/core";

import { getColorbars } from "./getColorbars";

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

function getColorbarNames(state: Partial<Parameters<typeof getColorbars>[0]>) {
  return getColorbars({
    layers,
    labels: [],
    points: [],
    shapes: [],
    ...state,
  }).map(({ name }) => name);
}

const shapes = {
  name: "Shapes",
  layer: "shown",
  dataSource: { type: "geojson", table: "cells" },
  shapeFillColor: fromColor,
  shapeStrokeColor: fromColor,
};

describe("getColorbars", () => {
  it("lists the objects whose colors come from a table column", () => {
    expect(
      getColorbarNames({
        points: [
          createPointsWith("from"),
          createPointsWith("constant", { pointColor: constantColor }),
        ],
      }),
    ).toEqual(["from"]);
  });

  it("leaves out hidden objects and objects on hidden layers", () => {
    expect(
      getColorbarNames({
        points: [
          createPointsWith("hidden", { visibility: false }),
          createPointsWith("on-hidden-layer", { layer: "hidden" }),
          createPointsWith("layer-from-column", {
            layer: { column: "layer" },
          }),
        ],
      }),
    ).toEqual(["layer-from-column"]);
  });

  it("leaves out only a constantly hidden visibility", () => {
    expect(
      getColorbarNames({
        points: [
          createPointsWith("constant", {
            pointVisibility: { constant: { value: false } },
          }),
          createPointsWith("from-column", {
            pointVisibility: { from: { column: "shown" } },
          }),
        ],
      }),
    ).toEqual(["from-column"]);
  });

  it("leaves out labels without a table", () => {
    expect(
      getColorbarNames({
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
    ).toEqual(["With table"]);
  });

  it("shares one colorbar between the fill and the stroke of shapes", () => {
    expect(
      getColorbarNames({
        shapes: [
          createShapes({ id: "shown", ...shapes }),
          createShapes({
            id: "hidden",
            ...shapes,
            shapeVisibility: { constant: { value: false } },
          }),
        ],
      }),
    ).toEqual(["Shapes"]);
  });

  it("names the fill and the stroke of shapes with different colorbars", () => {
    expect(
      getColorbarNames({
        shapes: [
          createShapes({
            id: "shapes",
            ...shapes,
            shapeStrokeColor: { from: { column: "score", palette: "magma" } },
          }),
        ],
      }),
    ).toEqual(["Shapes (fill)", "Shapes (stroke)"]);
  });

  it("names the fill of shapes whose stroke is not from a column", () => {
    expect(
      getColorbarNames({
        shapes: [
          createShapes({
            id: "shapes",
            ...shapes,
            shapeStrokeColor: constantColor,
          }),
        ],
      }),
    ).toEqual(["Shapes (fill)"]);
  });

  it("shares one colorbar between objects with the same table, column, palette and range", () => {
    expect(
      getColorbarNames({
        points: [
          createPointsWith("own table"),
          createPointsWith("explicit table", {
            dataSource: { type: "csv" },
            pointColor: {
              from: { table: "cells", column: "score", palette: "viridis" },
            },
          }),
          createPointsWith("other table", {
            dataSource: { type: "csv", table: "nuclei" },
          }),
          createPointsWith("other column", {
            pointColor: { from: { column: "area", palette: "viridis" } },
          }),
          createPointsWith("other range", {
            pointColor: {
              from: { column: "score", palette: "viridis", range: [0, 1] },
            },
          }),
        ],
      }),
    ).toEqual([
      "own table, explicit table",
      "other table",
      "other column",
      "other range",
    ]);
  });

  it("shares one colorbar between tables for the same column, palette and range", () => {
    const range: [number, number] = [0, 1];
    expect(
      getColorbars({
        layers,
        labels: [],
        points: [
          createPointsWith("cells", {
            pointColor: {
              from: {
                table: "cells",
                column: "score",
                palette: "viridis",
                range,
              },
            },
          }),
          createPointsWith("nuclei", {
            dataSource: { type: "csv", table: "nuclei" },
            pointColor: {
              from: { column: "score", palette: "viridis", range },
            },
          }),
        ],
        shapes: [],
      }).map(({ name, from }) => ({ name, table: from.table })),
    ).toEqual([{ name: "cells, nuclei", table: undefined }]);
  });

  it("keeps one colorbar per table for a range with only one end", () => {
    const pointColor: ColorConfig = {
      from: { column: "score", palette: "viridis", range: [0, null] },
    };
    expect(
      getColorbarNames({
        points: [
          createPointsWith("cells", { pointColor }),
          createPointsWith("nuclei", {
            dataSource: { type: "csv", table: "nuclei" },
            pointColor,
          }),
        ],
      }),
    ).toEqual(["cells", "nuclei"]);
  });

  it("leaves out objects without a table", () => {
    expect(
      getColorbarNames({
        points: [
          createPointsWith("without table", { dataSource: { type: "csv" } }),
        ],
      }),
    ).toEqual([]);
  });
});
