import type { TypedArrayOrArray } from "@tissuumaps/core";

import { ColumnQueryUtils } from "../ColumnQueryUtils";
import type {
  HierarchicalStore,
  HierarchicalStoreArray,
  HierarchicalStoreGroup,
} from "../HierarchicalStore";
import type { HierarchicalTableColumn } from "../HierarchicalTable";

/**
 * Reads the AnnData on-disk encoding of a hierarchical store
 *
 * Groups carrying an AnnData `encoding-type` attribute are decoded wherever
 * they are: `categorical`, `nullable-integer`, `nullable-boolean` and
 * `nullable-string-array` groups as columns, `csc_matrix` and `csr_matrix`
 * groups as matrices. The AnnData object of a store, if any, also names the
 * columns of its expression matrices by its `var` index.
 */
export class AnnDataUtils {
  /** Category labels of an AnnData file (anndata < 0.8), never a column */
  private static readonly _legacyCategoriesGroupName = "__categories";

  /** The encodings of a group that is read as one column or as a matrix */
  private static readonly _columnEncodingTypes = new Set([
    "categorical",
    "nullable-integer",
    "nullable-boolean",
    "nullable-string-array",
    "csc_matrix",
    "csr_matrix",
  ]);

  /**
   * Paths of the matrices named by the `var` index of their AnnData object,
   * relative to that object
   */
  private static readonly _variableMatrixPattern =
    /^(raw\/)?(X|layers\/[^/]+)$/;

  /**
   * @param group - A group of the store
   * @returns The AnnData encoding of the group, if it carries one
   */
  static getEncodingType(group: HierarchicalStoreGroup): string | undefined {
    const value = group.attrs["encoding-type"];
    return typeof value === "string" ? value : undefined;
  }

  /**
   * @param name - The name of a node
   * @returns Whether the node holds the category labels of a file written by
   * anndata < 0.8, which are never a column
   */
  static isLegacyCategoriesGroupName(name: string): boolean {
    return name === AnnDataUtils._legacyCategoriesGroupName;
  }

  /**
   * @param group - A group of the store
   * @returns Whether the group is an AnnData object
   */
  static isAnnDataObject(group: HierarchicalStoreGroup): boolean {
    return AnnDataUtils.getEncodingType(group) === "anndata";
  }

  /**
   * @param group - A group of the store
   * @returns Whether the group is encoded as one column or as a matrix, and
   * so is not walked for columns
   */
  static isColumnGroup(group: HierarchicalStoreGroup): boolean {
    const encodingType = AnnDataUtils.getEncodingType(group);
    return (
      encodingType !== undefined &&
      AnnDataUtils._columnEncodingTypes.has(encodingType)
    );
  }

  /**
   * Describes a group encoded as a column or a matrix
   *
   * @param group - The group, see {@link AnnDataUtils.isColumnGroup}
   * @param path - The path of the group
   * @returns The column, or `undefined` for a matrix without a
   * two-dimensional `shape` attribute
   */
  static getColumn(
    group: HierarchicalStoreGroup,
    path: string,
  ): HierarchicalTableColumn | undefined {
    switch (AnnDataUtils.getEncodingType(group)) {
      case "categorical":
      case "nullable-integer":
      case "nullable-boolean":
      case "nullable-string-array":
        return { kind: "dataset", path };
      case "csc_matrix":
      case "csr_matrix": {
        const shape = AnnDataUtils._getShapeAttribute(group);
        return shape !== undefined
          ? { kind: "matrix", path, numColumns: shape[1]! }
          : undefined;
      }
      default:
        return undefined;
    }
  }

  /**
   * Reads the number of rows of a group encoded as a column or a matrix
   *
   * @param store - The store to read from
   * @param group - The group, see {@link AnnDataUtils.isColumnGroup}
   * @param path - The path of the group
   * @param options - Optional abort signal
   * @returns The number of rows
   * @throws Error if a dataset of the column's encoding is missing, or if a
   * matrix has no two-dimensional `shape` attribute
   */
  static async getNumRows(
    store: HierarchicalStore,
    group: HierarchicalStoreGroup,
    path: string,
    options?: { signal?: AbortSignal },
  ): Promise<number> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    switch (AnnDataUtils.getEncodingType(group)) {
      case "categorical":
        return (
          await AnnDataUtils._getChildArray(store, path, "codes", { signal })
        ).shape[0]!;
      case "nullable-integer":
      case "nullable-boolean":
      case "nullable-string-array":
        return (
          await AnnDataUtils._getChildArray(store, path, "values", { signal })
        ).shape[0]!;
      default: {
        const shape = AnnDataUtils._getShapeAttribute(group);
        if (shape === undefined) {
          throw new Error(`"${path}" is a group, not a column`);
        }
        return shape[0]!;
      }
    }
  }

  /**
   * Reads the values of a group encoded as a column or a matrix
   *
   * @param store - The store to read from
   * @param group - The group, see {@link AnnDataUtils.isColumnGroup}
   * @param path - The path of the group
   * @param index - The index of the matrix column, `undefined` for a column
   * @param options - Optional abort signal
   * @returns The column values, with 64-bit integers converted to numbers
   * @throws Error if the matrix is stored as CSR, if a dataset of the
   * column's encoding is missing, or if a 64-bit integer is outside the safe
   * integer range
   */
  static async readColumn(
    store: HierarchicalStore,
    group: HierarchicalStoreGroup,
    path: string,
    index: number | undefined,
    options?: { signal?: AbortSignal },
  ): Promise<TypedArrayOrArray<unknown>> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    switch (AnnDataUtils.getEncodingType(group)) {
      case "categorical":
        return await AnnDataUtils._readCategorical(store, path, { signal });
      case "nullable-integer":
      case "nullable-boolean":
        return await AnnDataUtils._readNullableNumbers(store, path, { signal });
      case "nullable-string-array":
        return await AnnDataUtils._readNullableStrings(store, path, {
          signal,
        });
      case "csc_matrix":
        return await AnnDataUtils._readSparseColumn(
          store,
          group,
          path,
          index!,
          {
            signal,
          },
        );
      case "csr_matrix":
        throw new Error(
          `Matrix "${path}" is stored as CSR; only CSC matrices support column reads`,
        );
      default:
        throw new Error(`"${path}" is a group, not a column`);
    }
  }

  /**
   * @param store - The store to read from
   * @param path - The path of the dataframe group
   * @param options - Optional abort signal
   * @returns The path of the index of the dataframe, or `undefined` if the
   * group is no dataframe or names no index
   */
  static async getDataFrameIndexPath(
    store: HierarchicalStore,
    path: string,
    options?: { signal?: AbortSignal },
  ): Promise<string | undefined> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const group = await store.get(path, { signal });
    if (
      group === null ||
      group.kind !== "group" ||
      AnnDataUtils.getEncodingType(group) !== "dataframe"
    ) {
      return undefined;
    }
    const indexName = group.attrs["_index"];
    return typeof indexName === "string"
      ? ColumnQueryUtils.joinPath(path, indexName)
      : undefined;
  }

  /**
   * Gives the AnnData expression matrices the selectors of their `var` index
   *
   * `X` and the `layers` of an AnnData object have one column per variable, so
   * they can be addressed by variable name, such as `X[CD3]`. A `var` index
   * that cannot be read, or whose length does not match the matrix, leaves the
   * matrix with its column indices.
   *
   * @param store - The store to read from
   * @param columns - The columns to name, modified in place
   * @param annDataPath - The path of the AnnData object of the store, if any
   * @param options - Optional abort signal
   */
  static async assignMatrixSelectors(
    store: HierarchicalStore,
    columns: HierarchicalTableColumn[],
    annDataPath: string | undefined,
    options?: { signal?: AbortSignal },
  ): Promise<void> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    if (annDataPath === undefined) {
      return;
    }
    const selectorsByPath = new Map<string, string[] | undefined>();
    for (const column of columns) {
      if (column.kind !== "matrix") {
        continue;
      }
      const varPath = AnnDataUtils._getVariableDataFramePath(
        column.path,
        annDataPath,
      );
      if (varPath === undefined) {
        continue;
      }
      if (!selectorsByPath.has(varPath)) {
        const names = await AnnDataUtils._readDataFrameIndex(store, varPath, {
          signal,
        });
        selectorsByPath.set(
          varPath,
          names !== undefined
            ? ColumnQueryUtils.getMatrixSelectors(names)
            : undefined,
        );
      }
      const selectors = selectorsByPath.get(varPath);
      if (selectors !== undefined && selectors.length === column.numColumns) {
        column.selectors = selectors;
      }
    }
  }

  /**
   * @param path - The path of a matrix column
   * @param annDataPath - The path of the AnnData object of the store
   * @returns The path of the `var` dataframe naming the columns of the matrix,
   * or `undefined` if the matrix is not an expression matrix of the object
   */
  private static _getVariableDataFramePath(
    path: string,
    annDataPath: string,
  ): string | undefined {
    if (annDataPath !== "" && !path.startsWith(`${annDataPath}/`)) {
      return undefined;
    }
    const match = AnnDataUtils._variableMatrixPattern.exec(
      annDataPath === "" ? path : path.slice(annDataPath.length + 1),
    );
    return match !== null
      ? ColumnQueryUtils.joinPath(annDataPath, `${match[1] ?? ""}var`)
      : undefined;
  }

  /**
   * Reads the index values of an AnnData dataframe
   *
   * @param store - The store to read from
   * @param path - The path of the dataframe group
   * @param options - Optional abort signal
   * @returns The index values as strings, or `undefined` if the dataframe or
   * its index cannot be read
   */
  private static async _readDataFrameIndex(
    store: HierarchicalStore,
    path: string,
    options?: { signal?: AbortSignal },
  ): Promise<string[] | undefined> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const indexPath = await AnnDataUtils.getDataFrameIndexPath(store, path, {
      signal,
    });
    if (indexPath === undefined) {
      return undefined;
    }
    try {
      const node = await store.get(indexPath, { signal });
      if (node === null) {
        return undefined;
      }
      const values =
        node.kind === "array"
          ? await node.read({ signal })
          : await AnnDataUtils.readColumn(store, node, indexPath, undefined, {
              signal,
            });
      return Array.from(values, String);
    } catch {
      signal?.throwIfAborted();
      // an index this reader cannot decode only costs the matrix its names
      return undefined;
    }
  }

  private static _getShapeAttribute(
    group: HierarchicalStoreGroup,
  ): number[] | undefined {
    const value = group.attrs["shape"];
    if (value === undefined || value === null || typeof value !== "object") {
      return undefined;
    }
    const shape = Array.from(value as ArrayLike<number | bigint>, Number);
    return shape.length === 2 ? shape : undefined;
  }

  private static async _getChildArray(
    store: HierarchicalStore,
    path: string,
    name: string,
    options?: { signal?: AbortSignal },
  ): Promise<HierarchicalStoreArray> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const child = await store.get(`${path}/${name}`, { signal });
    if (child === null || child.kind !== "array") {
      throw new Error(`Column "${path}" is missing its "${name}" dataset`);
    }
    return child;
  }

  private static async _readChildArray(
    store: HierarchicalStore,
    path: string,
    name: string,
    options?: { signal?: AbortSignal },
  ): Promise<TypedArrayOrArray<unknown>> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const child = await AnnDataUtils._getChildArray(store, path, name, {
      signal,
    });
    return await child.read({ signal });
  }

  private static async _readCategorical(
    store: HierarchicalStore,
    path: string,
    options?: { signal?: AbortSignal },
  ): Promise<string[] | Float64Array> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const codes = (await AnnDataUtils._readChildArray(store, path, "codes", {
      signal,
    })) as ArrayLike<number>;
    const categories = await AnnDataUtils._readChildArray(
      store,
      path,
      "categories",
      { signal },
    );
    if (Array.isArray(categories)) {
      const labels = categories as string[];
      return Array.from(codes, (code) => (code < 0 ? "" : labels[code]!));
    }
    const numericCategories = categories as ArrayLike<number>;
    return Float64Array.from(codes, (code) =>
      code < 0 ? NaN : numericCategories[code]!,
    );
  }

  /**
   * Reads a nullable string column
   *
   * @param store - The store to read from
   * @param path - The path of the column group
   * @param options - Optional abort signal
   * @returns The values, with the masked ones as empty strings
   */
  private static async _readNullableStrings(
    store: HierarchicalStore,
    path: string,
    options?: { signal?: AbortSignal },
  ): Promise<string[]> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const values = (await AnnDataUtils._readChildArray(store, path, "values", {
      signal,
    })) as ArrayLike<unknown>;
    const mask = (await AnnDataUtils._readChildArray(store, path, "mask", {
      signal,
    })) as ArrayLike<number | boolean>;
    return Array.from(values, (value, i) => (mask[i] ? "" : String(value)));
  }

  private static async _readNullableNumbers(
    store: HierarchicalStore,
    path: string,
    options?: { signal?: AbortSignal },
  ): Promise<Float64Array> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const values = (await AnnDataUtils._readChildArray(store, path, "values", {
      signal,
    })) as ArrayLike<number | boolean>;
    const mask = (await AnnDataUtils._readChildArray(store, path, "mask", {
      signal,
    })) as ArrayLike<number | boolean>;
    return Float64Array.from(values, (value, i) =>
      mask[i] ? NaN : Number(value),
    );
  }

  private static async _readSparseColumn(
    store: HierarchicalStore,
    group: HierarchicalStoreGroup,
    path: string,
    index: number,
    options?: { signal?: AbortSignal },
  ): Promise<Float64Array> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const shape = AnnDataUtils._getShapeAttribute(group);
    if (shape === undefined) {
      throw new Error(
        `Matrix "${path}" has no two-dimensional shape attribute`,
      );
    }
    const indptr = await AnnDataUtils._getChildArray(store, path, "indptr", {
      signal,
    });
    const bounds = (await indptr.slice([[index, index + 2]], {
      signal,
    })) as ArrayLike<number>;
    const range: [number, number] = [bounds[0]!, bounds[1]!];
    const data = await AnnDataUtils._getChildArray(store, path, "data", {
      signal,
    });
    const values = (await data.slice([range], { signal })) as ArrayLike<
      number | boolean
    >;
    const indices = await AnnDataUtils._getChildArray(store, path, "indices", {
      signal,
    });
    const rows = (await indices.slice([range], {
      signal,
    })) as ArrayLike<number>;
    const column = new Float64Array(shape[0]!);
    for (let i = 0; i < rows.length; i++) {
      column[rows[i]!] = Number(values[i]);
    }
    return column;
  }
}
