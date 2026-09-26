import { describe, expect, it } from "vitest";
import type * as zarr from "zarrita";

import type {
  HierarchicalStoreArray,
  HierarchicalStoreGroup,
} from "../HierarchicalStore";
import { HierarchicalTableReader } from "../HierarchicalTableReader";
import { ZarrStore } from "./ZarrStore";

/** An in-memory Zarr store and the node metadata to consolidate */
type Fixture = {
  map: Map<string, Uint8Array>;
  metadata: Record<string, unknown>;
};

const encoder = new TextEncoder();

function toJson(value: unknown): Uint8Array {
  return encoder.encode(JSON.stringify(value));
}

function toBytes(array: ArrayBufferView): Uint8Array {
  return new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
}

/** Encodes strings in the numcodecs vlen-utf8 format */
function encodeVlenUtf8(values: string[]): Uint8Array {
  const items = values.map((value) => encoder.encode(value));
  const size = items.reduce((sum, item) => sum + 4 + item.length, 4);
  const bytes = new Uint8Array(size);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, items.length, true);
  let offset = 4;
  for (const item of items) {
    view.setUint32(offset, item.length, true);
    bytes.set(item, offset + 4);
    offset += 4 + item.length;
  }
  return bytes;
}

/** Encodes strings as NumPy `<U<numChars>` values */
function encodeFixedUnicode(values: string[], numChars: number): Uint8Array {
  const codePoints = new Uint32Array(values.length * numChars);
  values.forEach((value, i) => {
    Array.from(value).forEach((char, j) => {
      codePoints[i * numChars + j] = char.codePointAt(0)!;
    });
  });
  return toBytes(codePoints);
}

const names = ["a", "bc", "déf"];
const boolBytes = new Uint8Array([1, 0, 1]);
const spatialBytes = toBytes(new Float64Array([0, 1, 2, 3, 4, 5]));

function writeV2Metadata(
  fixture: Fixture,
  path: string,
  name: string,
  value: unknown,
): void {
  const key = path !== "" ? `${path}/${name}` : name;
  fixture.metadata[key] = value;
  fixture.map.set(`/${key}`, toJson(value));
}

function writeV2Group(
  fixture: Fixture,
  path: string,
  attrs: Record<string, unknown> = {},
): void {
  writeV2Metadata(fixture, path, ".zgroup", { zarr_format: 2 });
  writeV2Metadata(fixture, path, ".zattrs", attrs);
}

function writeV2Array(
  fixture: Fixture,
  path: string,
  dtype: unknown,
  shape: number[],
  chunk: Uint8Array,
  filters: unknown[] | null = null,
): void {
  writeV2Metadata(fixture, path, ".zarray", {
    zarr_format: 2,
    shape,
    chunks: shape,
    dtype,
    compressor: null,
    fill_value: null,
    filters,
    order: "C",
  });
  fixture.map.set(`/${path}/${shape.map(() => 0).join(".")}`, chunk);
}

function writeV2Strings(fixture: Fixture, path: string, values: string[]) {
  writeV2Array(fixture, path, "|O", [values.length], encodeVlenUtf8(values), [
    { id: "vlen-utf8" },
  ]);
}

function writeV3Group(
  fixture: Fixture,
  path: string,
  attributes: Record<string, unknown> = {},
): void {
  const node = { zarr_format: 3, node_type: "group", attributes };
  fixture.metadata[path] = node;
  fixture.map.set(`/${path}/zarr.json`, toJson(node));
}

function writeV3Array(
  fixture: Fixture,
  path: string,
  dataType: string,
  shape: number[],
  chunk: Uint8Array,
): void {
  const isString = dataType === "string";
  const node = {
    zarr_format: 3,
    node_type: "array",
    shape,
    data_type: dataType,
    chunk_grid: { name: "regular", configuration: { chunk_shape: shape } },
    chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
    fill_value: isString ? "" : dataType === "bool" ? false : 0,
    codecs: [
      isString
        ? { name: "vlen-utf8" }
        : { name: "bytes", configuration: { endian: "little" } },
    ],
    attributes: {},
  };
  fixture.metadata[path] = node;
  fixture.map.set(`/${path}/zarr.json`, toJson(node));
  fixture.map.set(`/${path}/c/${shape.map(() => 0).join("/")}`, chunk);
}

/**
 * Writes a Zarr v2 store with consolidated metadata
 *
 * @param prefix - The path of the group holding the nodes, "" for the root
 * @param metadataKey - The key of the consolidated metadata
 */
function writeV2Fixture(
  prefix = "",
  metadataKey = ".zmetadata",
): Map<string, Uint8Array> {
  const fixture: Fixture = { map: new Map(), metadata: {} };
  const parts = prefix.split("/").filter((part) => part !== "");
  const base = parts.length > 0 ? `${parts.join("/")}/` : "";
  parts.forEach((_, i) => writeV2Group(fixture, parts.slice(0, i).join("/")));
  writeV2Group(fixture, parts.join("/"), { title: "fixture" });
  writeV2Group(fixture, `${base}obs`, { _index: "names" });
  writeV2Array(
    fixture,
    `${base}obs/int32`,
    "<i4",
    [3],
    toBytes(new Int32Array([1, 2, 3])),
  );
  writeV2Array(
    fixture,
    `${base}obs/int64`,
    "<i8",
    [3],
    toBytes(new BigInt64Array([1n, 2n, 3n])),
  );
  writeV2Array(
    fixture,
    `${base}obs/float32`,
    "<f4",
    [3],
    toBytes(new Float32Array([1.5, 2.5, 3.5])),
  );
  writeV2Array(fixture, `${base}obs/bool`, "|b1", [3], boolBytes);
  writeV2Strings(fixture, `${base}obs/names`, names);
  writeV2Group(fixture, `${base}obsm`);
  writeV2Array(fixture, `${base}obsm/spatial`, "<f8", [3, 2], spatialBytes);
  writeV2Group(fixture, `${base}uns`);
  writeV2Array(
    fixture,
    `${base}uns/fixed`,
    "<U5",
    [3],
    encodeFixedUnicode(names, 5),
  );
  writeV2Array(
    fixture,
    `${base}uns/structured`,
    [
      ["x", "<i4"],
      ["y", "<f8"],
    ],
    [1],
    new Uint8Array(12),
  );
  fixture.map.set(
    `/${metadataKey}`,
    toJson({ zarr_consolidated_format: 1, metadata: fixture.metadata }),
  );
  return fixture.map;
}

/**
 * Writes a Zarr v3 store with its metadata consolidated in the root
 * `zarr.json`, as zarr-python does
 *
 * @param prefix - The path of the group holding the nodes, "" for the root
 */
function writeV3Fixture(prefix = ""): Map<string, Uint8Array> {
  const fixture: Fixture = { map: new Map(), metadata: {} };
  const parts = prefix.split("/").filter((part) => part !== "");
  const base = parts.length > 0 ? `${parts.join("/")}/` : "";
  // the root group is the consolidated zarr.json written last
  for (let i = 1; i < parts.length; i++) {
    writeV3Group(fixture, parts.slice(0, i).join("/"));
  }
  if (parts.length > 0) {
    writeV3Group(fixture, parts.join("/"), { title: "fixture" });
  }
  writeV3Group(fixture, `${base}obs`, { _index: "names" });
  writeV3Array(
    fixture,
    `${base}obs/int32`,
    "int32",
    [3],
    toBytes(new Int32Array([1, 2, 3])),
  );
  writeV3Array(
    fixture,
    `${base}obs/int64`,
    "int64",
    [3],
    toBytes(new BigInt64Array([1n, 2n, 3n])),
  );
  writeV3Array(
    fixture,
    `${base}obs/float32`,
    "float32",
    [3],
    toBytes(new Float32Array([1.5, 2.5, 3.5])),
  );
  writeV3Array(fixture, `${base}obs/bool`, "bool", [3], boolBytes);
  writeV3Array(
    fixture,
    `${base}obs/names`,
    "string",
    [3],
    encodeVlenUtf8(names),
  );
  writeV3Group(fixture, `${base}obsm`);
  writeV3Array(fixture, `${base}obsm/spatial`, "float64", [3, 2], spatialBytes);
  writeV3Group(fixture, `${base}uns`);
  fixture.map.set(
    "/zarr.json",
    toJson({
      zarr_format: 3,
      node_type: "group",
      attributes: parts.length > 0 ? {} : { title: "fixture" },
      consolidated_metadata: {
        kind: "inline",
        must_understand: false,
        metadata: fixture.metadata,
      },
    }),
  );
  return fixture.map;
}

const formats = [
  ["v2", writeV2Fixture],
  ["v3", writeV3Fixture],
] as const;

function toAsyncStore(map: Map<string, Uint8Array>): zarr.AsyncReadable {
  return { get: (key) => Promise.resolve(map.get(key)) };
}

function openFixture(
  map: Map<string, Uint8Array>,
  group = "",
): Promise<ZarrStore> {
  return ZarrStore.open(toAsyncStore(map), group);
}

async function getArray(
  store: ZarrStore,
  path: string,
): Promise<HierarchicalStoreArray> {
  const node = await store.get(path);
  if (node?.kind !== "array") {
    throw new Error(`"${path}" is not an array`);
  }
  return node;
}

async function getGroup(
  store: ZarrStore,
  path: string,
): Promise<HierarchicalStoreGroup> {
  const node = await store.get(path);
  if (node?.kind !== "group") {
    throw new Error(`"${path}" is not a group`);
  }
  return node;
}

describe("ZarrStore", () => {
  describe("open", () => {
    it.each(formats)(
      "lists the nodes of consolidated %s metadata",
      async (_, writeFixture) => {
        const store = await openFixture(writeFixture());
        expect((await getGroup(store, "")).keys).toEqual([
          "obs",
          "obsm",
          "uns",
        ]);
      },
    );

    it("falls back to the undotted zmetadata key", async () => {
      const store = await openFixture(writeV2Fixture("", "zmetadata"));
      expect((await getGroup(store, "")).keys).toEqual(["obs", "obsm", "uns"]);
    });

    it.each(formats)(
      "lists the nodes of a sub-group relative to it (%s)",
      async (_, writeFixture) => {
        const store = await openFixture(
          writeFixture("tables/adata"),
          "tables/adata",
        );
        const root = await getGroup(store, "");
        expect(root.attrs).toEqual({ title: "fixture" });
        expect(root.keys).toEqual(["obs", "obsm", "uns"]);
        expect((await getArray(store, "obs/int32")).shape).toEqual([3]);
        expect(await store.get("tables")).toBeNull();
      },
    );

    it("rejects a store without consolidated metadata", async () => {
      const map = writeV2Fixture();
      map.delete("/.zmetadata");
      await expect(openFixture(map)).rejects.toThrow(
        "no consolidated metadata",
      );
    });

    it("rejects a path that is not a group", async () => {
      const map = writeV2Fixture();
      await expect(openFixture(map, "obs/int32")).rejects.toThrow(
        'no group at "obs/int32"',
      );
      await expect(openFixture(map, "missing")).rejects.toThrow(
        'no group at "missing"',
      );
    });

    it("rejects an aborted signal", async () => {
      const reason = new Error("aborted");
      await expect(
        ZarrStore.open(toAsyncStore(writeV2Fixture()), "", {
          signal: AbortSignal.abort(reason),
        }),
      ).rejects.toBe(reason);
    });
  });

  describe("get", () => {
    it.each(formats)(
      "returns groups with their attributes and keys (%s)",
      async (_, writeFixture) => {
        const store = await openFixture(writeFixture());
        const root = await getGroup(store, "");
        expect(root.attrs).toEqual({ title: "fixture" });
        const obs = await getGroup(store, "obs");
        expect(obs.attrs).toEqual({ _index: "names" });
        expect(obs.keys).toEqual([
          "int32",
          "int64",
          "float32",
          "bool",
          "names",
        ]);
      },
    );

    it.each(formats)(
      "returns arrays with their shape and data type (%s)",
      async (_, writeFixture) => {
        const store = await openFixture(writeFixture());
        const arrays = await Promise.all(
          [
            "obs/int32",
            "obs/int64",
            "obs/float32",
            "obs/bool",
            "obs/names",
            "obsm/spatial",
          ].map((path) => getArray(store, path)),
        );
        expect(arrays.map(({ shape, dataType }) => [shape, dataType])).toEqual([
          [[3], "integer"],
          [[3], "integer"],
          [[3], "float"],
          [[3], "boolean"],
          [[3], "string"],
          [[3, 2], "float"],
        ]);
      },
    );

    it("returns v2 fixed-width unicode arrays as strings", async () => {
      const store = await openFixture(writeV2Fixture());
      expect((await getArray(store, "uns/fixed")).dataType).toBe("string");
    });

    it("returns null for missing paths", async () => {
      const store = await openFixture(writeV2Fixture());
      expect(await store.get("missing")).toBeNull();
      expect(await store.get("obs/missing")).toBeNull();
    });

    it("fetches no attributes for v2 arrays", async () => {
      const map = writeV2Fixture();
      const keys: string[] = [];
      const store = await ZarrStore.open(
        {
          get: (key) => {
            keys.push(key);
            return Promise.resolve(map.get(key));
          },
        },
        "",
      );
      await getArray(store, "obs/int32");
      expect(keys).not.toContain("/obs/int32/.zattrs");
    });

    it("returns null for a node zarrita cannot parse", async () => {
      const store = await openFixture(writeV2Fixture());
      expect(await store.get("uns/structured")).toBeNull();
      expect((await getGroup(store, "uns")).keys).toEqual([
        "fixed",
        "structured",
      ]);
    });

    it("rejects an aborted signal", async () => {
      const reason = new Error("aborted");
      const store = await openFixture(writeV2Fixture());
      await expect(
        store.get("obs", { signal: AbortSignal.abort(reason) }),
      ).rejects.toBe(reason);
    });
  });

  describe("read", () => {
    it.each(formats)(
      "reads typed arrays, booleans as bytes, strings and 64-bit integers (%s)",
      async (_, writeFixture) => {
        const store = await openFixture(writeFixture());
        const read = async (path: string) =>
          (await getArray(store, path)).read();
        expect(await read("obs/int32")).toEqual(new Int32Array([1, 2, 3]));
        expect(await read("obs/int64")).toEqual(
          new BigInt64Array([1n, 2n, 3n]),
        );
        expect(await read("obs/float32")).toEqual(
          new Float32Array([1.5, 2.5, 3.5]),
        );
        expect(await read("obs/bool")).toEqual(new Uint8Array([1, 0, 1]));
        expect(await read("obs/names")).toEqual(names);
      },
    );

    it("reads v2 fixed-width unicode strings", async () => {
      const store = await openFixture(writeV2Fixture());
      expect(await (await getArray(store, "uns/fixed")).read()).toEqual(names);
    });

    it("rejects an aborted signal", async () => {
      const reason = new Error("aborted");
      const store = await openFixture(writeV2Fixture());
      const array = await getArray(store, "obs/int32");
      await expect(
        array.read({ signal: AbortSignal.abort(reason) }),
      ).rejects.toBe(reason);
    });
  });

  describe("slice", () => {
    it.each(formats)(
      "reads ranges of 1-D and 2-D arrays (%s)",
      async (_, writeFixture) => {
        const store = await openFixture(writeFixture());
        expect(
          await (await getArray(store, "obs/names")).slice([[1, 3]]),
        ).toEqual(["bc", "déf"]);
        expect(
          await (await getArray(store, "obsm/spatial")).slice([null, [1, 2]]),
        ).toEqual(new Float64Array([1, 3, 5]));
      },
    );

    it("returns an empty array for an empty range", async () => {
      const store = await openFixture(writeV2Fixture());
      expect(
        await (await getArray(store, "obs/int32")).slice([[1, 1]]),
      ).toHaveLength(0);
      expect(
        await (await getArray(store, "obsm/spatial")).slice([null, [1, 1]]),
      ).toHaveLength(0);
    });

    it("rejects an aborted signal", async () => {
      const reason = new Error("aborted");
      const store = await openFixture(writeV2Fixture());
      const array = await getArray(store, "obs/int32");
      await expect(
        array.slice([[0, 1]], { signal: AbortSignal.abort(reason) }),
      ).rejects.toBe(reason);
    });
  });
});

describe("HierarchicalTableReader over ZarrStore", () => {
  it("reads an AnnData store", async () => {
    const fixture: Fixture = { map: new Map(), metadata: {} };
    writeV2Group(fixture, "", { "encoding-type": "anndata" });
    writeV2Group(fixture, "obs", {
      "encoding-type": "dataframe",
      _index: "_index",
    });
    writeV2Strings(fixture, "obs/_index", ["c0", "c1", "c2"]);
    writeV2Group(fixture, "obs/cell_type", { "encoding-type": "categorical" });
    writeV2Array(
      fixture,
      "obs/cell_type/codes",
      "|i1",
      [3],
      toBytes(new Int8Array([0, -1, 1])),
    );
    writeV2Strings(fixture, "obs/cell_type/categories", ["T", "B"]);
    writeV2Group(fixture, "X", {
      "encoding-type": "csc_matrix",
      shape: [3, 2],
    });
    writeV2Array(
      fixture,
      "X/data",
      "<f4",
      [2],
      toBytes(new Float32Array([5, 7])),
    );
    writeV2Array(
      fixture,
      "X/indices",
      "<i4",
      [2],
      toBytes(new Int32Array([0, 2])),
    );
    writeV2Array(
      fixture,
      "X/indptr",
      "<i8",
      [3],
      toBytes(new BigInt64Array([0n, 2n, 2n])),
    );
    writeV2Group(fixture, "var", {
      "encoding-type": "dataframe",
      _index: "_index",
    });
    writeV2Strings(fixture, "var/_index", ["CD3", "CD4"]);
    fixture.map.set(
      "/.zmetadata",
      toJson({ zarr_consolidated_format: 1, metadata: fixture.metadata }),
    );

    const reader = await HierarchicalTableReader.open(
      await openFixture(fixture.map),
    );
    expect(reader.numRows).toBe(3);
    expect(await reader.readColumn("obs/cell_type")).toEqual(["T", "", "B"]);
    expect(await reader.readColumn("X[CD3]", { numRows: 3 })).toEqual(
      new Float64Array([5, 0, 7]),
    );
    expect(await reader.readColumn("X[CD4]", { numRows: 3 })).toEqual(
      new Float64Array([0, 0, 0]),
    );
    reader.close();
  });
});
