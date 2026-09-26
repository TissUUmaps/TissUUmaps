import type { TypedArrayOrArray } from "@tissuumaps/core";

/**
 * One column that can be addressed by a column query
 *
 * A `dataset` column is a 1-D array, or an AnnData `categorical`,
 * `nullable-integer`, `nullable-boolean` or `nullable-string-array` group,
 * addressed by its path. A `matrix` column is a 2-D array or an AnnData
 * `csc_matrix`/`csr_matrix` group, whose columns are addressed as
 * `path[selector]` (see {@link ColumnQueryUtils}).
 */
export type HierarchicalTableColumn =
  | { kind: "dataset"; path: string }
  | {
      kind: "matrix";
      path: string;
      numColumns: number;
      /**
       * The selector of each matrix column, for expression matrices named by
       * an AnnData `var` index: the variable name, or the column index where
       * {@link ColumnQueryUtils.getMatrixSelectors} cannot use the name
       */
      selectors?: string[];
    };

/** An open hierarchical table: the columns of a store and a way to read them */
export interface HierarchicalTable {
  /** The columns, in path order */
  readonly columns: HierarchicalTableColumn[];
  /** The row count inferred from the store */
  readonly numRows: number;

  /**
   * Reads the values of a column
   *
   * @param query - The column query (see {@link ColumnQueryUtils})
   * @param options - The number of table rows, checked against the column
   * length, and an abort signal
   * @returns The column values
   */
  readColumn(
    query: string,
    options?: { numRows?: number; signal?: AbortSignal },
  ): Promise<TypedArrayOrArray<unknown>>;

  /**
   * Reads the minimum and maximum value of a numeric column
   *
   * @param query - The column query (see {@link ColumnQueryUtils})
   * @param options - See {@link HierarchicalTable.readColumn}
   * @returns The [min, max] range, or `undefined` if the column is not numeric
   * or holds no two distinct finite values
   */
  readRange(
    query: string,
    options?: { numRows?: number; signal?: AbortSignal },
  ): Promise<[number, number] | undefined>;

  /** Closes the table and releases its store */
  close(): void;
}
