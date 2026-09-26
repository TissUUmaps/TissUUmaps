import type { TypedArrayOrArray } from "@tissuumaps/core";

/** Data types of a {@link HierarchicalStoreArray}, as far as the reader tells them apart */
export type HierarchicalStoreDataType =
  "integer" | "float" | "string" | "boolean" | "other";

/**
 * The values of a {@link HierarchicalStoreArray}; 64-bit integers read as
 * `BigInt64Array`/`BigUint64Array`
 */
export type HierarchicalStoreValues =
  TypedArrayOrArray<unknown> | BigInt64Array | BigUint64Array;

/** A named container of other nodes */
export interface HierarchicalStoreGroup {
  readonly kind: "group";
  /** Attribute values by name */
  readonly attrs: Readonly<Record<string, unknown>>;
  /** The names of the child nodes, in store order */
  readonly keys: string[];
}

/** An n-dimensional array of values */
export interface HierarchicalStoreArray {
  readonly kind: "array";
  /** The dimensions; empty for a scalar */
  readonly shape: number[];
  /** The type of the values */
  readonly dataType: HierarchicalStoreDataType;

  /**
   * @param options - Optional abort signal
   * @returns All values
   */
  read(options?: { signal?: AbortSignal }): Promise<HierarchicalStoreValues>;

  /**
   * @param ranges - One `[start, end)` range per dimension, or `null` for the
   * whole dimension
   * @param options - Optional abort signal
   * @returns The values within the ranges; an empty range yields an empty
   * array of any type
   */
  slice(
    ranges: ([number, number] | null)[],
    options?: { signal?: AbortSignal },
  ): Promise<HierarchicalStoreValues>;
}

/** A node of a {@link HierarchicalStore} */
export type HierarchicalStoreNode =
  HierarchicalStoreGroup | HierarchicalStoreArray;

/**
 * A hierarchical container of groups and arrays, such as an HDF5 file or a
 * Zarr store
 *
 * Paths are slash-separated without a leading slash; the empty path is the
 * root group.
 */
export interface HierarchicalStore {
  /**
   * @param path - The path of the node
   * @param options - Optional abort signal
   * @returns The node, or `null` if there is none at the path
   */
  get(
    path: string,
    options?: { signal?: AbortSignal },
  ): Promise<HierarchicalStoreNode | null>;

  /** Closes the store */
  close(): void;
}
