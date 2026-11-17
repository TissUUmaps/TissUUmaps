import type { TableDataSource } from "../model/table";
import type { TypedArrayOrArray } from "../types/arrays";
import type { ProgressCallback } from "../types/callbacks";
import type { DataProvider, ItemsData } from "./base";

/**
 * A group of a table's columns, by the conventions of its format (see
 * `TableData.getQueryGroups`)
 *
 * A query group without subgroups is a leaf: a set of columns searched as one,
 * e.g. the features of one expression matrix. Pass its `id` as
 * `queryGroupId` (see `TableData.suggestColumnQueries`). A query group with
 * subgroups gathers leaves that hold the same kind of columns, e.g. the
 * features of AnnData in `X`, a layer and `raw`. Names are for display only.
 */
export type TableColumnQueryGroup = {
  /**
   * The key of the query group, unique among the table's query groups and
   * stable across reopening the table
   */
  id: string;

  /** The name to show, e.g. "Features" */
  name: string;

  /** The query groups below this one, if it is no leaf */
  subgroups?: TableColumnQueryGroup[];

  /**
   * Why the columns of the leaf cannot be read, if they cannot, in the
   * provider's own words (e.g. `"row-major-sparse"`)
   */
  unavailable?: string;
};

/** A column query suggested by `TableData.suggestColumnQueries` */
export type TableColumnQuerySuggestion = {
  /** The suggested query, in the provider's own format */
  query: string;

  /**
   * The name to show for the suggested column, if the query is not one (e.g.
   * the gene name of a matrix column)
   */
  name?: string;

  /**
   * Why the suggested column cannot be read, if it cannot, in the provider's
   * own words (e.g. `"row-major-sparse"`); reading it fails
   */
  unavailable?: string;

  /**
   * Whether the query continues into a group of columns, e.g. a path prefix
   *
   * A group suggestion does not resolve to a column; it continues the query,
   * e.g. into the columns of a group or a matrix. Other suggestions are
   * expected to resolve to an exact column name. Unrelated to query groups
   * (see `TableData.getQueryGroups`).
   *
   * @defaultValue false
   */
  group?: boolean;
};

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
   * Suggests column queries matching the current query
   *
   * Suggestions are unique by query. Suggestions that resolve to the current
   * query come first, best matches first, followed by the other suggestions.
   * The list is not capped.
   *
   * Within a leaf query group (see {@link getQueryGroups}), the suggestions
   * are the leaf's columns, and the current query is matched against their
   * names.
   *
   * @param currentQuery - The partial column query, or the text to search for
   * within a leaf query group
   * @param options - The `id` of a leaf query group (`queryGroupId`), if any,
   * and an optional abort signal
   * @returns A list of suggested column queries
   * @throws Error if the table has query groups but no leaf with the given
   * `id`; tables without query groups ignore the option
   */
  suggestColumnQueries(
    currentQuery: string,
    options?: { queryGroupId?: string; signal?: AbortSignal },
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
   * Returns the query groups of the table's columns, by the conventions of its
   * format
   *
   * Only implemented by table data whose format groups its columns (e.g. the
   * annotations and the features of AnnData); callers without it pick columns
   * by query (see {@link suggestColumnQueries}).
   *
   * @returns The top-level query groups, in the canonical order of the format
   */
  getQueryGroups?(): TableColumnQueryGroup[];

  /**
   * Loads the leaf query group of a column and the column's name in it (see
   * {@link getQueryGroups})
   *
   * Only implemented by table data with query groups. Async, as naming a column
   * may take a read (e.g. gene names from a column of `var`).
   *
   * @param column - The column name
   * @param options - Optional abort signal
   * @returns The `id` of the column's leaf query group and the column's name,
   * or `null` if the column belongs to no query group
   */
  describeColumn?(
    column: string,
    options?: { signal?: AbortSignal },
  ): Promise<{ queryGroupId: string; name: string } | null>;

  /**
   * Loads the column of another leaf query group that holds the same entity
   * (e.g. the same gene in another expression matrix), as far as the format
   * tells
   *
   * Only implemented by table data with query groups. Columns map only between
   * the leaves below one top-level query group (see {@link getQueryGroups});
   * names are never used to decide that. A column of a query group whose
   * columns cannot be read is still returned.
   *
   * @param column - The column name
   * @param options - The `id` of the leaf query group to map the column to
   * (`queryGroupId`) and an optional abort signal
   * @returns The column of the leaf query group, or `null` if it has none for
   * the entity
   * @throws Error if the table has no leaf query group with the given `id`
   */
  mapColumn?(
    column: string,
    options: { queryGroupId: string; signal?: AbortSignal },
  ): Promise<string | null>;

  /**
   * Returns whether a column holds categories, as its encoding declares
   *
   * Only implemented by table data whose format declares categories that the
   * loaded values do not reveal (e.g. AnnData categoricals with numeric
   * categories); callers without it, or given `undefined`, inspect the values
   * (see `TableUtils.loadIsColumnCategorical`).
   *
   * @param column - The column name
   * @returns Whether the column is categorical, or `undefined` if the format
   * does not say
   */
  isColumnCategorical?(column: string): boolean | undefined;

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
