import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { defaultLabelColorPalette } from "@tissuumaps/core";

import { ParquetShapesDataProvider } from "../parquet/ParquetShapesDataProvider";
import { SpatialDataProjectProvider } from "./SpatialDataProjectProvider";
import type { CoordinateTransformation } from "./SpatialDataTransformUtils";

const storeUrl = "https://data.example/data.zarr";

const identity = {
  flip: false,
  scale: 1,
  rotation: 0,
  translation: { x: 0, y: 0 },
};

/** Matches the ID of an object named after an element */
function elementID(name: string): string {
  return expect.stringMatching(new RegExp(`^${name}-`)) as string;
}

/** A coordinate transformation from an element's axes into a coordinate system */
function transformation(
  coordinateSystem: string,
  axes: string[],
  type: "identity" | { scale: number[] } = "identity",
): CoordinateTransformation {
  const systemAxes = axes.map((name) => ({ name }));
  const input = { name: "element", axes: systemAxes };
  const output = { name: coordinateSystem, axes: systemAxes };
  return type === "identity"
    ? { type: "identity", input, output }
    : { type: "scale", scale: type.scale, input, output };
}

/** The attributes of an image or labels element, nested as OME-Zarr 0.5 */
function rasterAttrs(
  coordinateTransformations: CoordinateTransformation[],
  labels = false,
) {
  return {
    ome: {
      multiscales: [
        {
          axes: (labels ? ["y", "x"] : ["c", "y", "x"]).map((name) => ({
            name,
          })),
          coordinateTransformations,
        },
      ],
      ...(labels ? { "image-label": {} } : {}),
    },
    spatialdata_attrs: { version: "0.1" },
  };
}

/** The attributes of a shapes or points element */
function vectorAttrs(
  kind: "shapes" | "points",
  coordinateTransformations: CoordinateTransformation[],
  spatialDataAttrs: Record<string, unknown> = {},
) {
  return {
    "encoding-type": `ngff:${kind}`,
    axes: ["x", "y"],
    coordinateTransformations,
    spatialdata_attrs: { version: "0.1", ...spatialDataAttrs },
  };
}

/** The attributes of a table element */
function tableAttrs(
  region: string | string[] | null,
  instanceKey: string | null,
) {
  return {
    "encoding-type": "anndata",
    "spatialdata-encoding-type": "ngff:regions_table",
    region,
    region_key: region !== null ? "region" : null,
    instance_key: instanceKey,
  };
}

/**
 * Stubs `fetch` to serve a Zarr v3 store, by default at {@link storeUrl}, with
 * consolidated metadata, whose groups have the given attributes by path
 *
 * The groups listing the elements of a kind are added as needed.
 */
function stubStore(
  groups: Record<string, Record<string, unknown>>,
  url = storeUrl,
) {
  const nodes = new Map<string, Record<string, unknown>>();
  for (const [path, attributes] of Object.entries(groups)) {
    const segments = path.split("/");
    for (let i = 1; i < segments.length; i++) {
      const parent = segments.slice(0, i).join("/");
      if (!nodes.has(parent)) {
        nodes.set(parent, {
          zarr_format: 3,
          node_type: "group",
          attributes: {},
        });
      }
    }
    nodes.set(path, { zarr_format: 3, node_type: "group", attributes });
  }
  const root = {
    zarr_format: 3,
    node_type: "group",
    attributes: { spatialdata_attrs: { version: "0.1" } },
    consolidated_metadata: {
      kind: "inline",
      must_understand: false,
      metadata: Object.fromEntries(nodes),
    },
  };
  const files = new Map<string, string>([
    [`${url}/zarr.json`, JSON.stringify(root)],
    ...[...nodes].map(
      ([path, node]) =>
        [`${url}/${path}/zarr.json`, JSON.stringify(node)] as const,
    ),
  ]);
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = input instanceof Request ? input.url : String(input);
    const body = files.get(url.split("?")[0]!);
    return Promise.resolve(
      body !== undefined
        ? new Response(body)
        : new Response(null, { status: 404, statusText: "Not Found" }),
    );
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("SpatialDataProjectProvider", () => {
  const provider = new SpatialDataProjectProvider();

  beforeEach(() => {
    // polygons are told from circles by the GeoParquet file
    vi.spyOn(
      ParquetShapesDataProvider.prototype,
      "supports",
    ).mockImplementation((source) =>
      Promise.resolve(source.includes("/polygons/")),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  describe("supports", () => {
    it("supports the root group of a SpatialData store", async () => {
      stubStore({});
      expect(await provider.supports(storeUrl, null)).toBe(true);
    });

    it("does not support the group of an element", async () => {
      stubStore({ "images/image": rasterAttrs([]) });
      expect(await provider.supports(`${storeUrl}/images/image`, null)).toBe(
        false,
      );
    });

    it("does not support a source that is no Zarr group", async () => {
      stubStore({});
      expect(
        await provider.supports("https://data.example/other.zarr", null),
      ).toBe(false);
    });

    it("throws for the workspace root without workspace", async () => {
      await expect(provider.supports("/", null)).rejects.toThrow();
    });
  });

  describe("load", () => {
    it("maps the elements of a store to objects", async () => {
      stubStore({
        "images/image": rasterAttrs([
          transformation("global", ["c", "y", "x"], { scale: [1, 2, 2] }),
        ]),
        "labels/cells": rasterAttrs(
          [transformation("global", ["y", "x"])],
          true,
        ),
        "shapes/polygons": vectorAttrs("shapes", [
          transformation("global", ["x", "y"]),
        ]),
        "shapes/circles": vectorAttrs("shapes", [
          transformation("global", ["x", "y"]),
        ]),
        "points/transcripts": vectorAttrs(
          "points",
          [transformation("global", ["x", "y"])],
          { feature_key: "gene" },
        ),
        "tables/cell_table": tableAttrs("cells", "cell_id"),
        "tables/unannotated_table": tableAttrs(null, null),
      });
      const { project, source } = await provider.load(storeUrl);
      expect(source).toBe(storeUrl);
      expect(project.name).toBe("data");
      expect(project.layers).toEqual([
        { id: expect.any(String) as string, name: "global", visibility: true },
      ]);
      const layer = project.layers![0]!.id;

      const [cellTable, unannotatedTable, circleTable, pointTable] =
        project.tables!;
      expect(project.tables).toHaveLength(4);
      expect(cellTable).toEqual({
        id: elementID("cell_table"),
        name: "cell_table",
        dataSource: {
          type: "zarr",
          source: "data.zarr/tables/cell_table",
          idColumn: "obs/cell_id",
        },
      });
      expect(unannotatedTable!.dataSource).toMatchObject({
        idColumn: undefined,
      });
      expect(circleTable).toEqual({
        id: elementID("circles"),
        name: "circles",
        dataSource: {
          type: "parquet",
          source: "data.zarr/shapes/circles/shapes.parquet",
        },
      });
      expect(pointTable).toEqual({
        id: elementID("transcripts"),
        name: "transcripts",
        dataSource: {
          type: "parquet",
          source: "data.zarr/points/transcripts/points.parquet",
        },
      });

      expect(project.images).toEqual([
        {
          id: elementID("image"),
          name: "image",
          layer,
          visibility: true,
          transform: { ...identity, scale: 2 },
          dataSource: { type: "ome-zarr", source: "data.zarr/images/image" },
        },
      ]);
      expect(project.labels).toEqual([
        {
          id: elementID("cells"),
          name: "cells",
          layer,
          visibility: false,
          transform: identity,
          dataSource: {
            type: "ome-zarr",
            source: "data.zarr/labels/cells",
            table: cellTable!.id,
          },
        },
      ]);
      expect(project.shapes).toEqual([
        {
          id: elementID("polygons"),
          name: "polygons",
          layer,
          visibility: false,
          transform: identity,
          dataSource: {
            type: "parquet",
            source: "data.zarr/shapes/polygons/shapes.parquet",
            table: undefined,
          },
        },
      ]);
      expect(project.points).toEqual([
        {
          id: elementID("circles"),
          name: "circles",
          layer,
          visibility: false,
          transform: identity,
          dataSource: {
            type: "table",
            table: circleTable!.id,
            x: "geometry[x]",
            y: "geometry[y]",
          },
          pointSize: { from: { column: "radius", unit: "data" } },
          pointSizeFactor: 2,
        },
        {
          id: elementID("transcripts"),
          name: "transcripts",
          layer,
          visibility: false,
          transform: identity,
          dataSource: {
            type: "table",
            table: pointTable!.id,
            x: "x",
            y: "y",
          },
          pointColor: {
            groupBy: {
              column: "gene",
              map: undefined,
              palette: defaultLabelColorPalette,
            },
          },
        },
      ]);
    });

    it("links a table only to the polygons or labels it annotates alone", async () => {
      stubStore({
        "labels/cells": rasterAttrs(
          [transformation("global", ["y", "x"])],
          true,
        ),
        "shapes/polygons": vectorAttrs("shapes", [
          transformation("global", ["x", "y"]),
        ]),
        "shapes/circles": vectorAttrs("shapes", [
          transformation("global", ["x", "y"]),
        ]),
        "points/transcripts": vectorAttrs("points", [
          transformation("global", ["x", "y"]),
        ]),
        "tables/shared_table": tableAttrs(["cells", "polygons"], "id"),
        "tables/circle_table": tableAttrs("circles", "id"),
        "tables/point_table": tableAttrs("transcripts", "id"),
      });
      const { project } = await provider.load(storeUrl);
      expect(project.labels![0]!.dataSource).toMatchObject({
        table: undefined,
      });
      expect(project.shapes![0]!.dataSource).toMatchObject({
        table: undefined,
      });
      const tableIDs = project.tables!.slice(0, 3).map(({ id }) => id);
      for (const { dataSource } of project.points!) {
        expect(tableIDs).not.toContain((dataSource as { table: string }).table);
      }
    });

    it("creates an object per coordinate system an element maps into", async () => {
      stubStore({
        "images/image": rasterAttrs([
          transformation("global", ["c", "y", "x"]),
          transformation("aligned", ["c", "y", "x"], { scale: [1, 2, 2] }),
        ]),
      });
      const { project } = await provider.load(storeUrl);
      const layerIDs = new Map(
        project.layers!.map(({ id, name }) => [name, id]),
      );
      expect(project.images).toEqual([
        expect.objectContaining({
          layer: layerIDs.get("global"),
          transform: identity,
        }),
        expect.objectContaining({
          layer: layerIDs.get("aligned"),
          transform: { ...identity, scale: 2 },
        }),
      ]);
      expect(project.images![0]!.id).not.toBe(project.images![1]!.id);
    });

    it("shows the global layer only", async () => {
      stubStore({
        "images/a": rasterAttrs([transformation("other", ["c", "y", "x"])]),
        "images/b": rasterAttrs([transformation("other", ["c", "y", "x"])]),
        "images/c": rasterAttrs([transformation("global", ["c", "y", "x"])]),
      });
      const { project } = await provider.load(storeUrl);
      expect(
        project.layers!.map(({ name, visibility }) => [name, visibility]),
      ).toEqual([
        ["other", false],
        ["global", true],
      ]);
    });

    it("shows the first layer holding the most objects without a global one", async () => {
      stubStore({
        "images/a": rasterAttrs([transformation("one", ["c", "y", "x"])]),
        "images/b": rasterAttrs([transformation("two", ["c", "y", "x"])]),
        "images/c": rasterAttrs([transformation("two", ["c", "y", "x"])]),
        "images/d": rasterAttrs([transformation("three", ["c", "y", "x"])]),
        "images/e": rasterAttrs([transformation("three", ["c", "y", "x"])]),
      });
      const { project } = await provider.load(storeUrl);
      expect(
        project.layers!.map(({ name, visibility }) => [name, visibility]),
      ).toEqual([
        ["one", false],
        ["two", true],
        ["three", false],
      ]);
    });

    it("shows only the first object of each layer, images first", async () => {
      stubStore({
        "images/a": rasterAttrs([transformation("global", ["c", "y", "x"])]),
        "images/b": rasterAttrs([
          transformation("global", ["c", "y", "x"]),
          transformation("other", ["c", "y", "x"]),
        ]),
        "labels/cells": rasterAttrs(
          [transformation("global", ["y", "x"])],
          true,
        ),
        "points/transcripts": vectorAttrs("points", [
          transformation("third", ["x", "y"]),
        ]),
      });
      const { project } = await provider.load(storeUrl);
      const layerIDs = new Map(
        project.layers!.map(({ id, name }) => [name, id]),
      );
      expect(
        [...project.images!, ...project.labels!, ...project.points!].map(
          ({ name, layer, visibility }) => [name, layer, visibility],
        ),
      ).toEqual([
        ["a", layerIDs.get("global"), true],
        ["b", layerIDs.get("global"), false],
        ["b", layerIDs.get("other"), true],
        ["cells", layerIDs.get("global"), false],
        ["transcripts", layerIDs.get("third"), true],
      ]);
    });

    it("shows polygons before circles, and circles before points", async () => {
      stubStore({
        "shapes/polygons": vectorAttrs("shapes", [
          transformation("first", ["x", "y"]),
        ]),
        "shapes/circles": vectorAttrs("shapes", [
          transformation("first", ["x", "y"]),
          transformation("second", ["x", "y"]),
        ]),
        "points/transcripts": vectorAttrs("points", [
          transformation("first", ["x", "y"]),
          transformation("second", ["x", "y"]),
          transformation("third", ["x", "y"]),
        ]),
      });
      const { project } = await provider.load(storeUrl);
      const layerIDs = new Map(
        project.layers!.map(({ id, name }) => [name, id]),
      );
      expect(
        [...project.shapes!, ...project.points!].map(
          ({ name, layer, visibility }) => [name, layer, visibility],
        ),
      ).toEqual([
        ["polygons", layerIDs.get("first"), true],
        ["circles", layerIDs.get("first"), false],
        ["circles", layerIDs.get("second"), true],
        ["transcripts", layerIDs.get("first"), false],
        ["transcripts", layerIDs.get("second"), false],
        ["transcripts", layerIDs.get("third"), true],
      ]);
    });

    it("skips an element whose metadata marks another kind, with a warning", async () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      stubStore({
        "images/image": rasterAttrs([
          transformation("global", ["c", "y", "x"]),
        ]),
        "images/transcripts": vectorAttrs("points", [
          transformation("global", ["x", "y"]),
        ]),
      });
      const { project } = await provider.load(storeUrl);
      expect(project.images!.map(({ name }) => name)).toEqual(["image"]);
      expect(project.points).toEqual([]);
      expect(warn).toHaveBeenCalledWith(
        "Skipping 'images/transcripts', whose metadata marks it as 'points'",
      );
    });

    it("warns about an approximated transformation", async () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      stubStore({
        "images/image": rasterAttrs([
          transformation("global", ["c", "y", "x"], { scale: [1, 2, 3] }),
        ]),
      });
      await provider.load(storeUrl);
      expect(warn).toHaveBeenCalledWith(
        "The transformation of 'images/image' into 'global' is approximated, as it scales the axes differently or shears them",
      );
    });

    it("names the element whose transformations cannot be read", async () => {
      stubStore({
        "images/image": rasterAttrs([
          {
            ...transformation("global", ["c", "y", "x"]),
            type: "bogus",
          } as unknown as CoordinateTransformation,
        ]),
      });
      const promise = provider.load(storeUrl);
      await expect(promise).rejects.toThrow(
        "Failed to read the transformations of 'images/image'",
      );
      await expect(promise).rejects.toHaveProperty(
        "cause.message",
        "Coordinate transformation type 'bogus' is not supported",
      );
    });

    it("strips a trailing slash from the source", async () => {
      stubStore({
        "images/image": rasterAttrs([
          transformation("global", ["c", "y", "x"]),
        ]),
      });
      const { project, source } = await provider.load(`${storeUrl}/`);
      expect(source).toBe(storeUrl);
      expect(project.images![0]!.dataSource.source).toBe(
        "data.zarr/images/image",
      );
    });

    it("loads a store served at an origin root, named after the host", async () => {
      const originUrl = "http://localhost:8000";
      stubStore(
        {
          "images/image": rasterAttrs([
            transformation("global", ["c", "y", "x"]),
          ]),
        },
        originUrl,
      );
      const { project, source } = await provider.load(`${originUrl}/`);
      expect(source).toBe(originUrl);
      expect(project.name).toBe("localhost:8000");
      expect(project.images![0]!.dataSource.source).toBe("images/image");
    });

    it("throws if aborted", async () => {
      const fetchMock = stubStore({});
      await expect(
        provider.load(storeUrl, { signal: AbortSignal.abort() }),
      ).rejects.toThrow();
      expect(fetchMock).not.toHaveBeenCalled();
    });
  });
});
