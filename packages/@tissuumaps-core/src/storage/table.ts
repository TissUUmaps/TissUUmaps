import type { TableDataSource } from "../model/table";
import type { TypedArrayOrArray } from "../types/arrays";
import type { ProgressCallback } from "../types/callbacks";
import type { DataProvider, ItemsData } from "./base";

/** A column query suggested by `TableData.suggestColumnQueries` */
export type TableColumnQuerySuggestion = {
  /** The suggested query, in the provider's own format */
  query: string;

  /**
   * Whether the query continues into a group of columns
   *
   * A group suggestion does not resolve to a column; it continues the query,
   * e.g. into the columns of a group or a matrix. Other suggestions are
   * expected to resolve to an exact column name.
   *
   * @defaultValue false
   */
  group?: boolean;

  /**
   * Whether the suggestion does not match the query
   *
   * A provider can list other columns after the matching ones, e.g. all
   * columns when none match.
   *
   * @defaultValue false
   */
  fallback?: boolean;
};

/**
 * Loaded tabular data providing column-wise access
 *
 * The items of a table are its rows, each of which annotates one item of
 * another data object (see `AnnotatedDataSource`); the item IDs and names are
 * those of the annotated items, in row order.
 */
export interface TableData extends ItemsData {
  /**
   * Returns column query suggestions for the current query
   *
   * The query format is up to the provider. Suggestions matching the current
   * query come first, best matches first, followed by any fallback
   * suggestions. The list is not capped.
   *
   * @param currentQuery - The partial column query to autocomplete
   * @param options - Optional abort signal
   * @returns A list of suggested column queries
   */
  suggestColumnQueries(
    currentQuery: string,
    options?: { signal?: AbortSignal },
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
   * @typeParam T - Element type of the column
   * @param column - The column name
   * @param options - Optional abort signal and progress callback
   * @returns The row count of every unique column value, keyed by value, in the
   * order the values first appear
   */
  loadUniqueValueCounts<T>(
    column: string,
    options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
  ): Promise<Map<T, number>>;

  /**
   * Loads a column's minimum and maximum values
   *
   * @param column - The column name
   * @param options - Optional abort signal and progress callback
   * @returns The numeric [min, max] value range of the column, or `undefined` if not numeric
   */
  loadValueRange(
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
