import type { TypedArrayOrArray } from "@tissuumaps/core";

/**
 * One column that can be addressed by a column query
 *
 * A `dataset` column is a 1-D array, or an AnnData `categorical`,
 * `nullable-integer`, `nullable-boolean` or `nullable-string-array` group,
 * addressed by its path. A `matrix` column is a 2-D array or an AnnData
 * `csc_matrix`/`csr_matrix` group, whose columns are addressed as
 * `path[selector]` (see {@link ColumnQueryUtils}).
 *
 * `categorical` is what the encoding declares, if anything (see
 * `TableData.isColumnCategorical`).
 */
export type HierarchicalTableColumn =
  | {
      kind: "dataset";
      path: string;
      categorical?: boolean;
    }
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

/**
 * A query group of the store (see `TableData.getQueryGroups`)
 *
 * A leaf at the path of a matrix column holds the matrix's columns, named by
 * their selectors. Any other leaf holds the dataset columns directly below its
 * path, but for the dataframe index at `indexPath`.
 */
export type HierarchicalTableQueryGroup = {
  id: string;
  name: string;
  indexPath?: string;
  subgroups?: HierarchicalTableQueryGroup[];
};

/** An open hierarchical table: the columns of a store and a way to read them */
export interface HierarchicalTable {
  /** The columns, in path order */
  readonly columns: HierarchicalTableColumn[];
  /** The row count inferred from the store */
  readonly numRows: number;
  /** The query groups, in the canonical order of the format */
  readonly queryGroups: HierarchicalTableQueryGroup[];

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

  /** Closes the table and releases its store */
  close(): void;
}
