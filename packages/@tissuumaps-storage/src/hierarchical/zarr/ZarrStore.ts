import * as zarr from "zarrita";

import type {
  HierarchicalStore,
  HierarchicalStoreArray,
  HierarchicalStoreDataType,
  HierarchicalStoreGroup,
  HierarchicalStoreNode,
  HierarchicalStoreValues,
} from "../HierarchicalStore";

/** A Zarr store whose nodes can be listed */
type ListableStore = zarr.Listable<zarr.AsyncReadable>;

/** Zarr data types; object arrays hold strings */
const dataTypes: Record<string, HierarchicalStoreDataType> = {
  int8: "integer",
  int16: "integer",
  int32: "integer",
  int64: "integer",
  uint8: "integer",
  uint16: "integer",
  uint32: "integer",
  uint64: "integer",
  float16: "float",
  float32: "float",
  float64: "float",
  bool: "boolean",
  string: "string",
  "v2:object": "string",
};

/**
 * A {@link HierarchicalStore} over a Zarr store, such as the `tables` of a SpatialData
 * store
 *
 * Reads are asynchronous fetches, so unlike HDF5 this store does not need a
 * Web Worker. Listing the nodes requires consolidated metadata, as a Zarr
 * store is a key-value store.
 */
export class ZarrStore implements HierarchicalStore {
  private readonly _store: ListableStore;
  /** The path of the group the table is read from, "" for the store itself */
  private readonly _group: string;
  /** The Zarr version of the store */
  private readonly _version: "v2" | "v3";
  /** The kind of every node, by path below the group */
  private readonly _kinds: Map<string, "array" | "group">;
  /** The names of the children of every group, by path below the group */
  private readonly _children: Map<string, string[]>;

  private constructor(
    store: ListableStore,
    group: string,
    version: "v2" | "v3",
    kinds: Map<string, "array" | "group">,
    children: Map<string, string[]>,
  ) {
    this._store = store;
    this._group = group;
    this._version = version;
    this._kinds = kinds;
    this._children = children;
  }

  /**
   * Opens a group of a Zarr store from its consolidated metadata
   *
   * @param store - Any asynchronous store zarrita can read
   * @param group - The path of the group to read, empty for the store itself
   * @param options - Optional abort signal
   * @returns A store over the group
   * @throws Error if the store has no consolidated metadata, or no group at
   * the given path
   */
  static async open(
    store: zarr.AsyncReadable,
    group: string,
    options?: { signal?: AbortSignal },
  ): Promise<ZarrStore> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const listableStore = await openConsolidated(store, { signal });
    if (listableStore === null) {
      throw new Error(
        "The Zarr store has no consolidated metadata, so its columns cannot be listed.",
      );
    }
    const groupPath = group !== "" ? `/${group}` : "";
    const kinds = new Map<string, "array" | "group">();
    const children = new Map<string, string[]>();
    for (const { path, kind } of listableStore.contents()) {
      if (path !== (groupPath || "/") && !path.startsWith(`${groupPath}/`)) {
        continue;
      }
      const relativePath = path.slice(groupPath.length + 1);
      kinds.set(relativePath, kind);
      if (relativePath === "") {
        continue;
      }
      const lastSlash = relativePath.lastIndexOf("/");
      const parent = lastSlash >= 0 ? relativePath.slice(0, lastSlash) : "";
      const name = relativePath.slice(lastSlash + 1);
      const siblings = children.get(parent);
      if (siblings !== undefined) {
        siblings.push(name);
      } else {
        children.set(parent, [name]);
      }
    }
    if (kinds.get("") !== "group") {
      throw new Error(`The Zarr store has no group at "${group}".`);
    }
    // probed once: zarr.open() would otherwise request a missing zarr.json
    // for every node of a v2 store
    const version = await zarr.open
      .v3(zarr.root(listableStore).resolve(`/${group}`), {
        kind: "group",
        signal,
      })
      .then(
        () => "v3" as const,
        () => "v2" as const,
      );
    signal?.throwIfAborted();
    return new ZarrStore(listableStore, group, version, kinds, children);
  }

  async get(
    path: string,
    options?: { signal?: AbortSignal },
  ): Promise<HierarchicalStoreNode | null> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const kind = this._kinds.get(path);
    if (kind === undefined) {
      return null;
    }
    const nodePath = [this._group, path].filter((part) => part !== "");
    const location = zarr.root(this._store).resolve(`/${nodePath.join("/")}`);
    try {
      if (kind === "group") {
        const group =
          this._version === "v3"
            ? await zarr.open.v3(location, { kind: "group", signal })
            : await zarr.open.v2(location, { kind: "group", signal });
        return new ZarrGroup(group.attrs, this._children.get(path) ?? []);
      }
      // no attrs: a HierarchicalStoreArray has none, and zarr-python leaves empty ones out
      // of the consolidated metadata, so each open would fetch a missing file
      const array =
        this._version === "v3"
          ? await zarr.open.v3(location, { kind: "array", signal })
          : await zarr.open.v2(location, {
              kind: "array",
              attrs: false,
              signal,
            });
      return new ZarrArray(array);
    } catch {
      signal?.throwIfAborted();
      // a node whose metadata zarrita cannot read, such as a structured
      // dtype under "uns", is reported as absent rather than failing the store
      return null;
    }
  }

  // nothing to release: reads are fetches
  close(): void {}
}

/**
 * Reads the consolidated metadata of a Zarr store
 *
 * @param store - Any asynchronous store zarrita can read
 * @param options - Optional abort signal
 * @returns The store with its nodes listed, or `null` if it has no
 * consolidated metadata
 */
async function openConsolidated(
  store: zarr.AsyncReadable,
  options?: { signal?: AbortSignal },
): Promise<ListableStore | null> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const consolidatedStore = await zarr.withMaybeConsolidatedMetadata(store);
  signal?.throwIfAborted(); // withMaybeConsolidatedMetadata() does not throw on abort
  if ("contents" in consolidatedStore) {
    return consolidatedStore;
  }
  // spatialdata writes the Zarr v2 consolidated metadata without the leading
  // dot the spec prescribes, and so do the Vitessce fixtures
  const undottedStore = await zarr.withMaybeConsolidatedMetadata(store, {
    format: "v2",
    metadataKey: "zmetadata",
  });
  signal?.throwIfAborted(); // withMaybeConsolidatedMetadata() does not throw on abort
  return "contents" in undottedStore ? undottedStore : null;
}

class ZarrGroup implements HierarchicalStoreGroup {
  readonly kind = "group";
  readonly attrs: Record<string, unknown>;
  readonly keys: string[];

  constructor(attrs: Record<string, unknown>, keys: string[]) {
    this.attrs = attrs;
    this.keys = keys;
  }
}

class ZarrArray implements HierarchicalStoreArray {
  readonly kind = "array";
  readonly shape: number[];
  readonly dataType: HierarchicalStoreDataType;
  private readonly _array: zarr.Array<zarr.DataType, zarr.Readable>;

  constructor(array: zarr.Array<zarr.DataType, zarr.Readable>) {
    this._array = array;
    this.shape = array.shape;
    this.dataType =
      dataTypes[array.dtype] ??
      (isFixedWidthStringType(array.dtype) ? "string" : "other");
  }

  async read(options?: {
    signal?: AbortSignal;
  }): Promise<HierarchicalStoreValues> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const chunk = await zarr.get(this._array, null, { signal });
    return toStoreValues(chunk.data);
  }

  async slice(
    ranges: ([number, number] | null)[],
    options?: { signal?: AbortSignal },
  ): Promise<HierarchicalStoreValues> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    if (ranges.some((range) => range !== null && range[0] >= range[1])) {
      // zarrita rejects empty selections, which an all-zero column of a
      // sparse matrix produces
      return [];
    }
    const chunk = await zarr.get(
      this._array,
      ranges.map((range) =>
        range !== null ? zarr.slice(range[0], range[1]) : null,
      ),
      { signal },
    );
    return toStoreValues(chunk.data);
  }
}

/** Fixed-width string types are spelled `v2:U<n>` (unicode) or `v2:S<n>` */
function isFixedWidthStringType(dtype: string): boolean {
  return dtype.startsWith("v2:U") || dtype.startsWith("v2:S");
}

/**
 * Converts the array-like views of zarrita into indexable arrays
 *
 * Booleans become the 0/1 bytes they are stored as, which is also how h5py
 * writes them to HDF5.
 */
function toStoreValues(data: unknown): HierarchicalStoreValues {
  if (data instanceof zarr.BoolArray) {
    return new Uint8Array(data.buffer, data.byteOffset, data.length);
  }
  if (Array.isArray(data) || ArrayBuffer.isView(data)) {
    return data as HierarchicalStoreValues;
  }
  return Array.from(data as Iterable<unknown>);
}
