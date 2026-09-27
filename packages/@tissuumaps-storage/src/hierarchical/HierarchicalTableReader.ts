import {
  MathUtils,
  type TypedArray,
  type TypedArrayOrArray,
} from "@tissuumaps/core";

import { ColumnQueryUtils } from "./ColumnQueryUtils";
import type {
  HierarchicalStore,
  HierarchicalStoreGroup,
} from "./HierarchicalStore";
import type {
  HierarchicalTable,
  HierarchicalTableColumn,
} from "./HierarchicalTable";
import { AnnDataUtils } from "./profiles/AnnDataUtils";

/** Data types whose values can be used as a column */
const readableDataTypes = new Set(["integer", "float", "string", "boolean"]);

/**
 * Reads columns from a {@link HierarchicalStore}, understanding the AnnData
 * profile (see {@link AnnDataUtils}) where present
 *
 * Any 1-D array is a column and any 2-D array is a matrix. Groups carrying an
 * AnnData `encoding-type` attribute are decoded wherever they are. The
 * AnnData object of the store, if any, also gives the row count of its `obs`
 * index and the column selectors of its expression matrices.
 */
export class HierarchicalTableReader implements HierarchicalTable {
  readonly columns: HierarchicalTableColumn[];
  readonly numRows: number;
  private readonly _store: HierarchicalStore;

  private constructor(
    store: HierarchicalStore,
    columns: HierarchicalTableColumn[],
    numRows: number,
  ) {
    this._store = store;
    this.columns = columns;
    this.numRows = numRows;
  }

  /**
   * Lists the columns of a store and infers its row count
   *
   * Groups are walked recursively, except groups encoded as a column or a
   * matrix. Scalars, arrays of more than two dimensions, arrays of non-scalar
   * types and nodes whose name contains a bracket are skipped.
   *
   * The row count is the length of the `obs` index of the AnnData object,
   * otherwise the first column's.
   *
   * @param store - The store to read from; closed by
   * {@link HierarchicalTable.close}, or here if this method rejects
   * @param options - Optional abort signal
   * @returns The reader
   * @throws Error if the store has no root group, no columns or more than one
   * AnnData object, or if the node that gives the row count is not a column
   */
  static async open(
    store: HierarchicalStore,
    options?: { signal?: AbortSignal },
  ): Promise<HierarchicalTableReader> {
    const { signal } = options ?? {};
    try {
      signal?.throwIfAborted();
      const root = await store.get("", { signal });
      if (root === null || root.kind !== "group") {
        throw new Error("The store has no root group.");
      }
      const columns: HierarchicalTableColumn[] = [];
      const annDataPaths = AnnDataUtils.isAnnDataObject(root) ? [""] : [];
      await collectColumns(store, root, "", columns, annDataPaths, { signal });
      if (annDataPaths.length > 1) {
        throw new Error(
          `The store holds ${annDataPaths.length} AnnData objects (${annDataPaths.map((path) => `"/${path}"`).join(", ")}); point the source at one of them.`,
        );
      }
      const annDataPath = annDataPaths[0];
      const numRows = await inferNumRows(store, columns, annDataPath, {
        signal,
      });
      await AnnDataUtils.assignMatrixSelectors(store, columns, annDataPath, {
        signal,
      });
      return new HierarchicalTableReader(store, columns, numRows);
    } catch (error) {
      store.close();
      throw error;
    }
  }

  /**
   * Reads the values of a column
   *
   * 64-bit integers are converted to numbers.
   *
   * @param query - The column query (see {@link ColumnQueryUtils})
   * @param options - The number of table rows, checked against the column
   * length, and an abort signal
   * @returns The column values
   * @throws Error if the query addresses no column, if the column length
   * differs from the number of table rows, if the matrix is stored as CSR, if
   * a dataset of the column's encoding is missing, or if a 64-bit integer is
   * outside the safe integer range
   */
  async readColumn(
    query: string,
    options?: { numRows?: number; signal?: AbortSignal },
  ): Promise<TypedArrayOrArray<unknown>> {
    const { numRows, signal } = options ?? {};
    signal?.throwIfAborted();
    const resolved = ColumnQueryUtils.resolveColumn(this.columns, query);
    if (resolved === null) {
      throw new Error(`Column query "${query}" addresses no column`);
    }
    const { column, index } = resolved;
    const values = await readColumnValues(this._store, column.path, index, {
      signal,
    });
    if (numRows !== undefined && values.length !== numRows) {
      throw new Error(
        `Column "${query}" has ${values.length} rows, but the table has ${numRows}`,
      );
    }
    return values;
  }

  /**
   * Reads the minimum and maximum value of a numeric column
   *
   * @param query - The column query (see {@link ColumnQueryUtils})
   * @param options - See {@link HierarchicalTableReader.readColumn}
   * @returns The [min, max] range, or `undefined` if the column is not numeric
   * or holds no two distinct finite values
   * @throws Error see {@link HierarchicalTableReader.readColumn}
   */
  async readRange(
    query: string,
    options?: { numRows?: number; signal?: AbortSignal },
  ): Promise<[number, number] | undefined> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const values = await this.readColumn(query, options);
    if (typeof values[0] !== "number") {
      return undefined;
    }
    const [vmin, vmax] = await MathUtils.computeRange(values as TypedArray, {
      signal,
    });
    return vmin < vmax ? [vmin, vmax] : undefined;
  }

  /** Closes the store */
  close(): void {
    this._store.close();
  }
}

async function collectColumns(
  store: HierarchicalStore,
  group: HierarchicalStoreGroup,
  prefix: string,
  columns: HierarchicalTableColumn[],
  annDataPaths: string[],
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  // sorted, as a store may list its nodes in write order
  for (const name of group.keys.toSorted()) {
    if (
      AnnDataUtils.isLegacyCategoriesGroupName(name) ||
      !ColumnQueryUtils.isQueryableName(name)
    ) {
      continue;
    }
    const path = `${prefix}${name}`;
    const node = await store.get(path, { signal });
    if (node === null) {
      continue;
    }
    if (node.kind === "group") {
      if (AnnDataUtils.isColumnGroup(node)) {
        const column = AnnDataUtils.getColumn(node, path);
        if (column !== undefined) {
          columns.push(column);
        }
      } else {
        if (AnnDataUtils.isAnnDataObject(node)) {
          annDataPaths.push(path);
        }
        await collectColumns(store, node, `${path}/`, columns, annDataPaths, {
          signal,
        });
      }
    } else {
      const { shape, dataType } = node;
      if (!readableDataTypes.has(dataType)) {
        continue;
      }
      if (shape.length === 1) {
        columns.push({ kind: "dataset", path });
      } else if (shape.length === 2) {
        columns.push({ kind: "matrix", path, numColumns: shape[1]! });
      }
    }
  }
}

async function inferNumRows(
  store: HierarchicalStore,
  columns: HierarchicalTableColumn[],
  annDataPath: string | undefined,
  options?: { signal?: AbortSignal },
): Promise<number> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  if (annDataPath !== undefined) {
    const indexPath = await AnnDataUtils.getDataFrameIndexPath(
      store,
      ColumnQueryUtils.joinPath(annDataPath, "obs"),
      { signal },
    );
    if (indexPath !== undefined) {
      return await getNumRows(store, indexPath, { signal });
    }
  }
  const firstColumn = columns[0];
  if (firstColumn === undefined) {
    throw new Error("No columns found in the store.");
  }
  return await getNumRows(store, firstColumn.path, { signal });
}

async function getNumRows(
  store: HierarchicalStore,
  path: string,
  options?: { signal?: AbortSignal },
): Promise<number> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const node = await store.get(path, { signal });
  if (node === null) {
    throw new Error(`Column "${path}" does not exist`);
  }
  if (node.kind === "array") {
    return node.shape[0]!;
  }
  if (!AnnDataUtils.isColumnGroup(node)) {
    throw new Error(`"${path}" is a group, not a column`);
  }
  return await AnnDataUtils.getNumRows(store, node, path, { signal });
}

/**
 * Reads the values of a column
 *
 * @param store - The store to read from
 * @param path - The path of the column
 * @param index - The index of the matrix column, `undefined` for a dataset
 * column
 * @param options - Optional abort signal
 * @returns The column values, with 64-bit integers converted to numbers
 * @throws Error if the path addresses no column, if the matrix is stored as
 * CSR, if a dataset of the column's encoding is missing, or if a 64-bit
 * integer is outside the safe integer range
 */
async function readColumnValues(
  store: HierarchicalStore,
  path: string,
  index: number | undefined,
  options?: { signal?: AbortSignal },
): Promise<TypedArrayOrArray<unknown>> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const node = await store.get(path, { signal });
  if (node === null) {
    throw new Error(`Column "${path}" does not exist`);
  }
  if (node.kind === "array") {
    return index === undefined
      ? await node.read({ signal })
      : await node.slice([null, [index, index + 1]], { signal });
  }
  if (!AnnDataUtils.isColumnGroup(node)) {
    throw new Error(`"${path}" is a group, not a column`);
  }
  return await AnnDataUtils.readColumn(store, node, path, index, { signal });
}
