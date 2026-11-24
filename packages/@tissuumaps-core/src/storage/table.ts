import type { TableDataSource } from "../model/table";
import type { TypedArrayOrArray } from "../types/arrays";
import type { ProgressCallback } from "../types/callbacks";
import type { DataProvider, ItemsData } from "./base";

/**
 * A group of the columns of a table to pick from, by the conventions of its
 * format (see `TableData.loadQueryGroups`)
 *
 * A leaf is a set of columns that can be searched as one, e.g. the features
 * of one expression matrix. Its `id` is passed back as `groupId` (see
 * `TableData.suggestColumnQueries`); it is opaque to callers, unique among the
 * table's leaves and stable across reopening the table. A node gathers
 * subgroups that hold the same kind of columns, e.g. the features of AnnData
 * in `X`, a layer and `raw`; a column maps between the leaves below one
 * top-level query group (see `TableData.mapColumn`). Query groups may nest to
 * any depth. Names are for display only: they do not identify query groups or
 * columns.
 */
export type TableColumnQueryGroup =
  | {
      /** The name to show for the leaf, e.g. "X" */
      name: string;
      /** The key of the leaf, passed back as `groupId` */
      id: string;
    }
  | {
      /** The name to show for the node, e.g. "Features" */
      name: string;
      /** The query groups below this one */
      subgroups: TableColumnQueryGroup[];
    };

/** A column query suggested by `TableData.suggestColumnQueries` */
export type TableColumnQuerySuggestion = {
  /** The suggested query, in the provider's own format */
  query: string;

  /**
   * The name to show for the suggested column, if the query is not one (e.g.
   * the gene symbol of a matrix column whose query names its gene ID)
   */
  name?: string;

  /**
   * Whether the query continues into a group of columns, e.g. a path prefix
   *
   * A group suggestion does not resolve to a column; it continues the query,
   * e.g. into the columns of a group or a matrix. Other suggestions are
   * expected to resolve to an exact column name. Unrelated to query groups,
   * whose leaves are identified by their `id` (see
   * `TableData.loadQueryGroups`).
   *
   * @defaultValue false
   */
  group?: boolean;
};

/**
 * The type of a table column's values, as far as coloring and grouping by it
 * are concerned
 *
 * - `"number"`: numeric values on a continuous scale
 * - `"category"`: categories, whatever their values (e.g. strings, or the
 *   numeric categories of an AnnData categorical)
 * - `"boolean"`: two states (possibly with missing values)
 */
export type TableColumnDataType = "number" | "category" | "boolean";

/**
 * Loaded tabular data providing column-wise access
 *
 * The items of a table are its rows, each of which annotates one item of
 * another data object (see `AnnotatedDataSource`); the item IDs and names are
 * those of the annotated items, in row order.
 *
 * A query is what is typed to find a column (partial, in the provider's own
 * format); a column is the exact name that {@link resolveColumnQuery} returns,
 * which is saved and passed to {@link loadValues}.
 */
export interface TableData extends ItemsData {
  /**
   * Returns column query suggestions for the current query
   *
   * The query format is up to the provider. Suggestions matching the current
   * query come first, best matches first, followed by the other suggestions.
   * The list is not capped.
   *
   * Scoped to a leaf query group (see {@link loadQueryGroups}), the
   * suggestions are the columns of that leaf, and the current query is matched
   * against their names (and their queries) rather than continued as a query.
   *
   * @param currentQuery - The partial column query to autocomplete, or the
   * text to search for within a leaf query group
   * @param options - The `id` of the leaf query group to suggest the columns
   * of (`groupId`), if any, and an optional abort signal
   * @returns A list of suggested column queries
   * @throws Error if the table implements {@link loadQueryGroups} and has no
   * query group with the given `id`; tables without query groups ignore the
   * option
   */
  suggestColumnQueries(
    currentQuery: string,
    options?: { groupId?: string; signal?: AbortSignal },
  ): Promise<TableColumnQuerySuggestion[]>;

  /**
   * Resolves a query to an exact column name
   *
   * @param query - The column query
   * @param options - Optional abort signal
   * @returns The resolved column name, or `null` if no match is found
   */
  resolveColumnQuery(
    query: string,
    options?: { signal?: AbortSignal },
  ): Promise<string | null>;

  /**
   * Lists the query groups of the table's columns, by the conventions of its
   * format, for picking columns by group rather than by query
   *
   * Only implemented by table data whose format groups its columns (e.g. the
   * annotations and the features of AnnData); an empty list means that the
   * table has no such groups. How query groups are presented is up to the
   * caller.
   *
   * @param options - Optional abort signal
   * @returns The top-level query groups, in the canonical order of the format
   */
  loadQueryGroups?(options?: {
    signal?: AbortSignal;
  }): Promise<TableColumnQueryGroup[]>;

  /**
   * Describes a column by the leaf query group holding it and its name (see
   * {@link loadQueryGroups})
   *
   * @param column - The column name
   * @param options - Optional abort signal
   * @returns The `id` of the column's leaf query group (`groupId`) and the
   * column's name, or `null` if the column belongs to no query group
   */
  describeColumn?(
    column: string,
    options?: { signal?: AbortSignal },
  ): Promise<{ groupId: string; name: string } | null>;

  /**
   * Maps a column to the column of another leaf query group holding the same
   * entity (e.g. the same gene in another expression matrix), as far as the
   * format tells; names are never used to decide that
   *
   * Columns map only between the leaves below one top-level query group (see
   * {@link loadQueryGroups}): a column of another top-level query group maps
   * to `null`.
   *
   * @param column - The column name
   * @param options - The `id` of the leaf query group to map the column to
   * and an optional abort signal
   * @returns The column of the query group, or `null` if it has none for the
   * entity
   * @throws Error if the table has no query group with the given `id`
   */
  mapColumn?(
    column: string,
    options: { groupId: string; signal?: AbortSignal },
  ): Promise<string | null>;

  /**
   * Loads the type that a column's encoding declares
   *
   * Only implemented by table data whose format declares column types that
   * the loaded values do not reveal (e.g. AnnData categoricals with numeric
   * categories, or nullable booleans loaded as numbers); otherwise, callers
   * derive the type from the values loaded by {@link loadValues} (see
   * `TableUtils.computeColumnDataType`).
   *
   * @param column - The column name
   * @param options - Optional abort signal
   * @returns The declared type, or `undefined` if the format declares none for
   * the column
   */
  loadColumnDataType?(
    column: string,
    options?: { signal?: AbortSignal },
  ): Promise<TableColumnDataType | undefined>;

  /**
   * Loads a column's values
   *
   * Numeric columns are loaded as numeric typed arrays, all other columns as
   * plain arrays (see `TypedArrayOrArray`).
   *
   * @typeParam T - Element type of the returned array
   * @param column - The column name
   * @param options - Optional abort signal and progress callback
   * @returns The column values
   */
  loadValues<T>(
    column: string,
    options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
  ): Promise<TypedArrayOrArray<T>>;

  /**
   * Loads the number of rows per unique value of a column
   *
   * Only implemented by table data that can count the values more cheaply
   * than by loading the column (e.g. from file metadata); otherwise, callers
   * count the values loaded by {@link loadValues}.
   *
   * @typeParam T - Element type of the column
   * @param column - The column name
   * @param options - Optional abort signal and progress callback
   * @returns The row count of every unique column value, keyed by value, in the
   * order the values first appear
   */
  loadUniqueValueCounts?<T>(
    column: string,
    options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
  ): Promise<Map<T, number>>;

  /**
   * Loads a column's minimum and maximum values
   *
   * Only implemented by table data that can determine the range more cheaply
   * than by loading the column (e.g. from file metadata); otherwise, callers
   * compute the range of the values loaded by {@link loadValues}.
   *
   * @param column - The column name
   * @param options - Optional abort signal and progress callback
   * @returns The numeric [min, max] value range of the column, or `undefined` if not numeric
   */
  loadValueRange?(
    column: string,
    options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
  ): Promise<[number, number] | undefined>;
}

/**
 * Data provider for tabular data
 *
 * @typeParam TTableDataSource - The data source type this data provider opens
 * @typeParam TTableData - The {@link TableData} type produced by this data
 * provider
 * @typeParam TNormalizedTableDataSource - The normalized data source type
 * produced by `normalize` and accepted by `load`
 */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface TableDataProvider<
  TTableDataSource extends TableDataSource,
  TTableData extends TableData,
  TNormalizedTableDataSource extends TTableDataSource = TTableDataSource,
> extends DataProvider<
  TTableDataSource,
  TTableData,
  TNormalizedTableDataSource
> {}
