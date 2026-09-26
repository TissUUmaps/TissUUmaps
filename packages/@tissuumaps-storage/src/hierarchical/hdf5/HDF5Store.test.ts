import h5wasm from "h5wasm";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import type {
  HierarchicalStoreArray,
  HierarchicalStoreGroup,
} from "../HierarchicalStore";
import { HierarchicalTableReader } from "../HierarchicalTableReader";
import { HDF5Store } from "./HDF5Store";

const storeFixturePath = "/store.h5";
const annDataFixturePath = "/fixture.h5ad";

function writeStoreFixture(): void {
  const file = new h5wasm.File(storeFixturePath, "w");
  const group = file.create_group("group");
  group.create_attribute("encoding-type", "csc_matrix");
  group.create_attribute("shape", new Int32Array([3, 2]));
  group.create_dataset({ name: "int32", data: new Int32Array([1, 2, 3]) });
  group.create_dataset({ name: "int64", data: new BigInt64Array([1n, 2n]) });
  group.create_dataset({ name: "float", data: new Float32Array([1.5, 2.5]) });
  group.create_dataset({ name: "strings", data: ["a", "bc"] });
  file.create_dataset({ name: "scalar", data: "sample" });
  file.create_dataset({
    name: "matrix",
    data: new Float64Array([0, 1, 2, 3, 4, 5]),
    shape: [3, 2],
  });
  file.create_dataset({
    name: "compound",
    data: new Map<string, Float32Array | Int32Array>([
      ["x", new Float32Array([1, 2])],
      ["y", new Int32Array([3, 4])],
    ]),
  });
  file.close();
}

function writeAnnDataFixture(): void {
  const file = new h5wasm.File(annDataFixturePath, "w");
  file.create_attribute("encoding-type", "anndata");

  const obs = file.create_group("obs");
  obs.create_attribute("encoding-type", "dataframe");
  obs.create_attribute("_index", "_index");
  obs.create_dataset({ name: "_index", data: ["c0", "c1", "c2"] });
  const cellType = obs.create_group("cell_type");
  cellType.create_attribute("encoding-type", "categorical");
  cellType.create_dataset({ name: "codes", data: new Int8Array([0, -1, 1]) });
  cellType.create_dataset({ name: "categories", data: ["T", "B"] });

  const x = file.create_group("X");
  x.create_attribute("encoding-type", "csc_matrix");
  x.create_attribute("shape", new Int32Array([3, 2]));
  x.create_dataset({ name: "data", data: new Float32Array([5, 7]) });
  x.create_dataset({ name: "indices", data: new Int32Array([0, 2]) });
  x.create_dataset({ name: "indptr", data: new Int32Array([0, 2, 2]) });

  const variables = file.create_group("var");
  variables.create_attribute("encoding-type", "dataframe");
  variables.create_attribute("_index", "_index");
  variables.create_dataset({ name: "_index", data: ["CD3", "CD4"] });

  file.close();
}

beforeAll(async () => {
  await h5wasm.ready;
  writeStoreFixture();
  writeAnnDataFixture();
});

describe("HDF5Store", () => {
  let store: HDF5Store;

  beforeAll(() => {
    store = new HDF5Store(new h5wasm.File(storeFixturePath, "r"));
  });

  afterAll(() => {
    store.close();
  });

  async function getArray(path: string): Promise<HierarchicalStoreArray> {
    const node = await store.get(path);
    if (node?.kind !== "array") {
      throw new Error(`"${path}" is not an array`);
    }
    return node;
  }

  async function getGroup(path: string): Promise<HierarchicalStoreGroup> {
    const node = await store.get(path);
    if (node?.kind !== "group") {
      throw new Error(`"${path}" is not a group`);
    }
    return node;
  }

  describe("get", () => {
    it("returns the root and nested groups with their attributes and keys", async () => {
      const root = await getGroup("");
      expect(root.keys).toEqual(["compound", "group", "matrix", "scalar"]);
      const group = await getGroup("group");
      expect(group.attrs).toEqual({
        "encoding-type": "csc_matrix",
        shape: new Int32Array([3, 2]),
      });
      expect(group.keys).toEqual(["float", "int32", "int64", "strings"]);
    });

    it("returns arrays with their shape and data type", async () => {
      const arrays = await Promise.all(
        [
          "group/int32",
          "group/int64",
          "group/float",
          "group/strings",
          "scalar",
          "matrix",
          "compound",
        ].map(getArray),
      );
      expect(arrays.map(({ shape, dataType }) => [shape, dataType])).toEqual([
        [[3], "integer"],
        [[2], "integer"],
        [[2], "float"],
        [[2], "string"],
        [[], "string"],
        [[3, 2], "float"],
        [[2], "other"],
      ]);
    });

    it("returns null for missing paths", async () => {
      expect(await store.get("missing")).toBeNull();
      expect(await store.get("group/missing")).toBeNull();
    });

    it("rejects an aborted signal", async () => {
      await expect(
        store.get("group", { signal: AbortSignal.abort() }),
      ).rejects.toThrow();
    });
  });

  describe("read", () => {
    it("reads typed arrays, strings and 64-bit integers", async () => {
      expect(await (await getArray("group/int32")).read()).toEqual(
        new Int32Array([1, 2, 3]),
      );
      expect(await (await getArray("group/float")).read()).toEqual(
        new Float32Array([1.5, 2.5]),
      );
      expect(await (await getArray("group/strings")).read()).toEqual([
        "a",
        "bc",
      ]);
      expect(await (await getArray("group/int64")).read()).toEqual(
        new BigInt64Array([1n, 2n]),
      );
    });

    it("rejects an aborted signal", async () => {
      const array = await getArray("group/int32");
      await expect(
        array.read({ signal: AbortSignal.abort() }),
      ).rejects.toThrow();
    });
  });

  describe("slice", () => {
    it("reads a range of a 1-D array", async () => {
      const array = await getArray("group/int32");
      expect(await array.slice([[1, 3]])).toEqual(new Int32Array([2, 3]));
    });

    it("reads a column of a 2-D array", async () => {
      const array = await getArray("matrix");
      expect(await array.slice([null, [1, 2]])).toEqual(
        new Float64Array([1, 3, 5]),
      );
    });

    it("returns an empty array for an empty range", async () => {
      const array = await getArray("group/int32");
      expect(await array.slice([[1, 1]])).toHaveLength(0);
    });

    it("rejects an aborted signal", async () => {
      const array = await getArray("group/int32");
      await expect(
        array.slice([[0, 1]], { signal: AbortSignal.abort() }),
      ).rejects.toThrow();
    });
  });
});

describe("HierarchicalTableReader over HDF5Store", () => {
  it("reads an AnnData file", async () => {
    const reader = await HierarchicalTableReader.open(
      new HDF5Store(new h5wasm.File(annDataFixturePath, "r")),
    );
    expect(reader.numRows).toBe(3);
    expect(reader.columns).toContainEqual({
      kind: "matrix",
      path: "X",
      numColumns: 2,
      selectors: ["CD3", "CD4"],
    });
    expect(await reader.readColumn("obs/cell_type")).toEqual(["T", "", "B"]);
    expect(await reader.readColumn("X[CD3]", { numRows: 3 })).toEqual(
      new Float64Array([5, 0, 7]),
    );
    reader.close();
  });
});
