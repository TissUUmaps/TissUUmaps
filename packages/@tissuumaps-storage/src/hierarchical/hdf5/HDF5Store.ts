import h5wasm, { type Dataset, type Group, type File as H5File } from "h5wasm";

import { ArrayUtils, type TypedArrayOrArray } from "@tissuumaps/core";

import type {
  HierarchicalStore,
  HierarchicalStoreArray,
  HierarchicalStoreDataType,
  HierarchicalStoreGroup,
  HierarchicalStoreNode,
} from "../HierarchicalStore";

/** HDF5 datatype classes (H5T_class_t); enums read as their integer codes */
const dataTypes: Record<number, HierarchicalStoreDataType> = {
  0: "integer", // H5T_INTEGER
  1: "float", // H5T_FLOAT
  3: "string", // H5T_STRING
  8: "integer", // H5T_ENUM
};

/**
 * A {@link HierarchicalStore} over an HDF5 file opened with h5wasm
 *
 * h5wasm reads synchronously through the Emscripten file system.
 * {@link HDF5Store.open} mounts the source with file systems that only work
 * in a Web Worker.
 */
export class HDF5Store implements HierarchicalStore {
  private readonly _file: H5File;

  /**
   * @param file - An HDF5 file opened for reading
   */
  constructor(file: H5File) {
    this._file = file;
  }

  /**
   * Mounts the source into the Emscripten file system and opens it
   *
   * URLs are read on demand through HTTP range requests (falling back to a
   * full download when the server does not advertise byte serving); files are
   * mounted without being copied. Both require a Web Worker context.
   *
   * @param source - The file or URL to open
   * @param options - Optional abort signal
   * @returns A store for the opened file
   * @throws Error if the source is not an HDF5 file
   */
  static async open(
    source: File | string,
    options?: { signal?: AbortSignal },
  ): Promise<HDF5Store> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const { FS } = await h5wasm.ready;
    signal?.throwIfAborted(); // h5wasm.ready does not throw on abort
    let path;
    if (typeof source === "string") {
      FS.createLazyFile("/", "data.h5", source, true, false);
      path = "/data.h5";
    } else {
      FS.mkdir("/work");
      const { WORKERFS } = FS.filesystems as {
        WORKERFS: FS.FileSystemType;
      };
      FS.mount(WORKERFS, { files: [source] }, "/work");
      path = `/work/${source.name}`;
    }
    const file = new h5wasm.File(path, "r");
    // h5wasm reports a file it cannot open with an invalid id, not an error
    if (file.file_id < 0n) {
      throw new Error("The source is not an HDF5 file.");
    }
    return new HDF5Store(file);
  }

  get(
    path: string,
    options?: { signal?: AbortSignal },
  ): Promise<HierarchicalStoreNode | null> {
    const { signal } = options ?? {};
    if (signal?.aborted) {
      return Promise.reject(signal.reason as Error);
    }
    const entity = path === "" ? this._file : this._file.get(path);
    if (entity instanceof h5wasm.Group) {
      return Promise.resolve(new HDF5Group(entity));
    }
    if (entity instanceof h5wasm.Dataset) {
      return Promise.resolve(new HDF5Array(entity));
    }
    return Promise.resolve(null);
  }

  close(): void {
    this._file.close();
  }
}

class HDF5Group implements HierarchicalStoreGroup {
  readonly kind = "group";
  private readonly _group: Group;
  private _attrs: Record<string, unknown> | undefined;

  constructor(group: Group) {
    this._group = group;
  }

  get attrs(): Record<string, unknown> {
    return (this._attrs ??= Object.fromEntries(
      Object.entries(this._group.attrs).map(([name, attr]) => [
        name,
        attr.value,
      ]),
    ));
  }

  get keys(): string[] {
    return this._group.keys();
  }
}

class HDF5Array implements HierarchicalStoreArray {
  readonly kind = "array";
  readonly shape: number[];
  readonly dataType: HierarchicalStoreDataType;
  private readonly _dataset: Dataset;

  constructor(dataset: Dataset) {
    const { metadata } = dataset;
    this._dataset = dataset;
    this.shape = metadata.shape ?? [];
    this.dataType = dataTypes[metadata.type] ?? "other";
  }

  read(options?: {
    signal?: AbortSignal;
  }): Promise<TypedArrayOrArray<unknown>> {
    const { signal } = options ?? {};
    if (signal?.aborted) {
      return Promise.reject(signal.reason as Error);
    }
    return new Promise((resolve) => {
      resolve(toStoreValues(this._dataset.value));
    });
  }

  slice(
    ranges: ([number, number] | null)[],
    options?: { signal?: AbortSignal },
  ): Promise<TypedArrayOrArray<unknown>> {
    const { signal } = options ?? {};
    if (signal?.aborted) {
      return Promise.reject(signal.reason as Error);
    }
    return new Promise((resolve) => {
      resolve(
        toStoreValues(this._dataset.slice(ranges.map((range) => range ?? []))),
      );
    });
  }
}

/**
 * Converts the values h5wasm reads into the arrays the store contract holds
 *
 * 64-bit integers, which h5wasm reads as `BigInt64Array`/`BigUint64Array`,
 * become 64-bit floats.
 */
function toStoreValues(values: unknown): TypedArrayOrArray<unknown> {
  if (values instanceof BigInt64Array || values instanceof BigUint64Array) {
    return ArrayUtils.parseSafeInts(values);
  }
  return values as TypedArrayOrArray<unknown>;
}
