import { afterEach, describe, expect, it, vi } from "vitest";

import { SpatialDataProjectProvider } from "./SpatialDataProjectProvider";

const storeURL = "https://data.example/datasets/data.zarr";

function system(name: string, axes: string[]) {
  return { name, axes: axes.map((axis) => ({ name: axis })) };
}

function identity(coordinateSystem: string, axes: string[]) {
  return {
    type: "identity",
    input: system(axes.join(""), axes),
    output: system(coordinateSystem, axes),
  };
}

/**
 * The attributes of an image or labels element, as spatialdata writes them,
 * mapping into the given coordinate systems by identity unless transformations
 * are given
 */
function rasterAttributes(
  coordinateSystems: string[],
  labels = false,
  transformations?: object[],
) {
  const axes = labels ? ["y", "x"] : ["c", "y", "x"];
  return {
    ome: {
      multiscales: [
        {
          axes: axes.map((name) => ({ name })),
          datasets: [{ path: "0" }],
          coordinateTransformations:
            transformations ??
            coordinateSystems.map((name) => identity(name, axes)),
        },
      ],
      ...(labels ? { "image-label": {} } : {}),
    },
    spatialdata_attrs: { version: "0.3" },
  };
}

function tableAttributes(region: string | string[]) {
  return {
    "encoding-type": "anndata",
    "spatialdata-encoding-type": "ngff:regions_table",
    region,
    region_key: "region",
    instance_key: "cell_id",
    version: "0.2",
  };
}

/** The root `zarr.json` of a store with the given groups and attributes */
function rootMetadata(
  attributes: Record<string, unknown>,
  groups: { [path: string]: Record<string, unknown> },
) {
  const group = (attrs: Record<string, unknown>) => ({
    zarr_format: 3,
    node_type: "group",
    attributes: attrs,
  });
  return JSON.stringify({
    ...group(attributes),
    consolidated_metadata: {
      kind: "inline",
      must_understand: false,
      metadata: Object.fromEntries(
        Object.entries(groups).map(([path, attrs]) => [path, group(attrs)]),
      ),
    },
  });
}

const spatialDataAttributes = { spatialdata_attrs: { version: "0.2" } };

/** Stubs `fetch` to serve the given files, and a 404 for any other URL */
function stubFetch(files: { [url: string]: string }) {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const url = input instanceof Request ? input.url : String(input);
      const body = files[url];
      return Promise.resolve(
        body !== undefined
          ? new Response(body)
          : new Response(null, { status: 404 }),
      );
    }),
  );
}

/** A directory handle holding the given files, by path */
function fakeDirectory(
  name: string,
  files: { [path: string]: string },
): FileSystemDirectoryHandle {
  const notFound = (entryName: string) =>
    Promise.reject(new DOMException(entryName, "NotFoundError"));
  return {
    kind: "directory",
    name,
    getDirectoryHandle(entryName: string) {
      const prefix = `${entryName}/`;
      const children = Object.fromEntries(
        Object.entries(files)
          .filter(([path]) => path.startsWith(prefix))
          .map(([path, content]) => [path.slice(prefix.length), content]),
      );
      return Object.keys(children).length > 0
        ? Promise.resolve(fakeDirectory(entryName, children))
        : notFound(entryName);
    },
    getFileHandle(entryName: string) {
      const content = files[entryName];
      if (content === undefined) {
        return Object.keys(files).some((path) =>
          path.startsWith(`${entryName}/`),
        )
          ? Promise.reject(new DOMException(entryName, "TypeMismatchError"))
          : notFound(entryName);
      }
      return Promise.resolve({
        kind: "file",
        name: entryName,
        getFile: () => Promise.resolve(new File([content], entryName)),
      });
    },
  } as unknown as FileSystemDirectoryHandle;
}

describe("SpatialDataProjectProvider", () => {
  const provider = new SpatialDataProjectProvider();

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  describe("supports", () => {
    it("recognizes a store whose root lists SpatialData attributes", async () => {
      stubFetch({
        [`${storeURL}/zarr.json`]: rootMetadata(spatialDataAttributes, {}),
      });
      await expect(provider.supports(storeURL, null)).resolves.toBe(true);
    });

    it("does not recognize another Zarr store", async () => {
      stubFetch({ [`${storeURL}/zarr.json`]: rootMetadata({}, {}) });
      await expect(provider.supports(storeURL, null)).resolves.toBe(false);
    });

    it.each([
      ["an image", rasterAttributes(["global"])],
      ["labels", rasterAttributes(["global"], true)],
      [
        "shapes",
        {
          "encoding-type": "ngff:shapes",
          axes: ["x", "y"],
          coordinateTransformations: [identity("global", ["x", "y"])],
          spatialdata_attrs: { version: "0.3" },
        },
      ],
      [
        "points",
        {
          "encoding-type": "ngff:points",
          axes: ["x", "y"],
          coordinateTransformations: [identity("global", ["x", "y"])],
          spatialdata_attrs: { feature_key: "gene", version: "0.2" },
        },
      ],
    ])(
      "does not recognize the group of %s, despite its SpatialData attributes",
      async (_, attributes) => {
        stubFetch({ [`${storeURL}/zarr.json`]: rootMetadata(attributes, {}) });
        await expect(provider.supports(storeURL, null)).resolves.toBe(false);
      },
    );

    it("does not recognize a source that is no Zarr group", async () => {
      stubFetch({});
      await expect(provider.supports(storeURL, null)).resolves.toBe(false);
    });

    it("checks the workspace root", async () => {
      const workspace = fakeDirectory("dataset", {
        "zarr.json": rootMetadata(spatialDataAttributes, {}),
      });
      await expect(provider.supports("/", workspace)).resolves.toBe(true);
    });
  });

  describe("load", () => {
    const groups = {
      images: {},
      "images/he": rasterAttributes(["global"]),
      "images/aligned": rasterAttributes(["global", "aligned"]),
      labels: {},
      "labels/cells": rasterAttributes(["global"], true),
      "labels/nuclei": rasterAttributes(["global"], true),
      tables: {},
      "tables/cell_table": tableAttributes("cells"),
      "tables/shared_table": tableAttributes(["cells", "nuclei"]),
    };
    const root = rootMetadata(spatialDataAttributes, groups);

    async function load(source = storeURL) {
      stubFetch({ [`${storeURL}/zarr.json`]: root });
      return provider.load(source);
    }

    it("maps images and labels to OME-Zarr objects, relative to the store", async () => {
      const { project, source } = await load();
      expect(source).toBe(storeURL);
      expect(project.name).toBe("data");
      expect(
        project.images!.map(({ name, dataSource }) => [name, dataSource]),
      ).toEqual([
        ["he", { type: "ome-zarr", source: "data.zarr/images/he" }],
        ["aligned", { type: "ome-zarr", source: "data.zarr/images/aligned" }],
        ["aligned", { type: "ome-zarr", source: "data.zarr/images/aligned" }],
      ]);
      expect(project.labels!.map(({ name }) => name)).toEqual([
        "cells",
        "nuclei",
      ]);
    });

    it("drops a trailing slash from the source", async () => {
      const { source, project } = await load(`${storeURL}/`);
      expect(source).toBe(storeURL);
      expect(project.images![0]!.dataSource.source).toBe("data.zarr/images/he");
    });

    it("creates a layer per coordinate system, showing global", async () => {
      const { project } = await load();
      const layers = project.layers!;
      expect(layers.map(({ name, visibility }) => [name, visibility])).toEqual([
        ["global", true],
        ["aligned", false],
      ]);
      const [global, aligned] = layers.map(({ id }) => id);
      expect(project.images!.map(({ layer }) => layer)).toEqual([
        global,
        global,
        aligned,
      ]);
    });

    it("shows the coordinate system with the most elements without global", async () => {
      stubFetch({
        [`${storeURL}/zarr.json`]: rootMetadata(spatialDataAttributes, {
          images: {},
          "images/a": rasterAttributes(["fov1"]),
          "images/b": rasterAttributes(["fov2"]),
          "images/c": rasterAttributes(["fov2"]),
        }),
      });
      const { project } = await provider.load(storeURL);
      expect(
        project.layers!.map(({ name, visibility }) => [name, visibility]),
      ).toEqual([
        ["fov1", false],
        ["fov2", true],
      ]);
    });

    it("gives every object an ID of its own, without colons", async () => {
      const { project } = await load();
      const ids = [
        ...project.layers!,
        ...project.images!,
        ...project.labels!,
        ...project.tables!,
      ].map(({ id }) => id);
      expect(new Set(ids).size).toBe(ids.length);
      expect(ids.every((id) => !id.includes(":"))).toBe(true);
      expect(project.images![0]!.id).toMatch(/^he-/);
    });

    it("maps tables to Zarr tables keyed by instance, linked to the element they alone annotate", async () => {
      const { project } = await load();
      expect(
        project.tables!.map(({ name, dataSource }) => [name, dataSource]),
      ).toEqual([
        [
          "cell_table",
          {
            type: "zarr",
            source: "data.zarr/tables/cell_table",
            idColumn: "obs/cell_id",
          },
        ],
        [
          "shared_table",
          {
            type: "zarr",
            source: "data.zarr/tables/shared_table",
            idColumn: "obs/cell_id",
          },
        ],
      ]);
      const [cellTable] = project.tables!;
      expect(
        project.labels!.map(({ name, dataSource }) => [
          name,
          (dataSource as { table?: string }).table,
        ]),
      ).toEqual([
        ["cells", cellTable!.id],
        ["nuclei", undefined],
      ]);
    });

    it("warns about an approximated transformation", async () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const attributes = rasterAttributes([], false, [
        {
          type: "scale",
          scale: [1, 2, 4],
          input: system("cyx", ["c", "y", "x"]),
          output: system("global", ["c", "y", "x"]),
        },
      ]);
      stubFetch({
        [`${storeURL}/zarr.json`]: rootMetadata(spatialDataAttributes, {
          images: {},
          "images/stretched": attributes,
        }),
      });
      const { project } = await provider.load(storeURL);
      expect(project.images![0]!.transform!.scale).toBeCloseTo(Math.SQRT2 * 2);
      expect(warn).toHaveBeenCalledWith(
        'The transformation of images/stretched into "global" is approximated, as it scales the axes differently or shears them.',
      );
    });

    it("loads a store that is the workspace root, with sources relative to it", async () => {
      const workspace = fakeDirectory("dataset", { "zarr.json": root });
      const { project, source } = await provider.load("/", { workspace });
      expect(source).toBe("/");
      expect(project.name).toBe("dataset");
      expect(project.images![0]!.dataSource.source).toBe("images/he");
      expect(project.tables![0]!.dataSource.source).toBe("tables/cell_table");
    });

    it("loads a store in the workspace", async () => {
      const workspace = fakeDirectory("", { "data.zarr/zarr.json": root });
      const { project, source } = await provider.load("/data.zarr", {
        workspace,
      });
      expect(source).toBe("/data.zarr");
      expect(project.images![0]!.dataSource.source).toBe("data.zarr/images/he");
    });

    it("rejects a store without consolidated metadata", async () => {
      stubFetch({
        [`${storeURL}/zarr.json`]: JSON.stringify({
          zarr_format: 3,
          node_type: "group",
          attributes: spatialDataAttributes,
        }),
      });
      await expect(provider.load(storeURL)).rejects.toThrow();
    });
  });
});
