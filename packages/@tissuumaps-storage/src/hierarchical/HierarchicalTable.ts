import type { TypedArrayOrArray } from "@tissuumaps/core";

/**
 * Why a column of a store cannot be read, although it is listed (see the
 * `unavailable` field of `TableColumnQuerySuggestion`)
 *
 * - `"row-major-sparse"`: a column of a sparse matrix stored by rows (an
 *   AnnData CSR matrix), which cannot be read column by column
 * - `"length-mismatch"`: a column whose length is not the table's number of
 *   rows (e.g. a column of the AnnData `var` dataframe)
 * - `"unsupported-type"`: a column of a data type that cannot be read (e.g.
 *   compound or opaque HDF5 types)
 */
export const HierarchicalColumnUnavailability = {
  RowMajorSparse: "row-major-sparse",
  LengthMismatch: "length-mismatch",
  UnsupportedType: "unsupported-type",
} as const;
export type HierarchicalColumnUnavailability =
  (typeof HierarchicalColumnUnavailability)[keyof typeof HierarchicalColumnUnavailability];

/**
 * One column that can be addressed by a column query
 *
 * A `dataset` column is a 1-D array, or an AnnData `categorical`,
 * `nullable-integer`, `nullable-boolean` or `nullable-string-array` group,
 * addressed by its path. A `matrix` column is a 2-D array or an AnnData
 * `csc_matrix`/`csr_matrix` group, whose columns are addressed as
 * `path[selector]` (see {@link ColumnQueryUtils}).
 *
 * `numRows` is the column length where it is known without reading values
 * (from the array shape, the matrix `shape` attribute, or the index of the
 * enclosing AnnData dataframe); `categorical` is what the encoding declares,
 * if anything (see `TableData.isColumnCategorical`); `unavailable` tells why
 * the column cannot be read, as far as the metadata tells.
 */
export type HierarchicalTableColumn =
  | {
      kind: "dataset";
      path: string;
      numRows?: number;
      categorical?: boolean;
      unavailable?: HierarchicalColumnUnavailability;
    }
  | {
      kind: "matrix";
      path: string;
      numColumns: number;
      numRows?: number;
      unavailable?: HierarchicalColumnUnavailability;
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
  unavailable?: HierarchicalColumnUnavailability;
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
