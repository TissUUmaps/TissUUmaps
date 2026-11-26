import type { TableDataSource } from "../model/table";
import type { TypedArrayOrArray } from "../types/arrays";
import type { ProgressCallback } from "../types/callbacks";
import type { DataProvider, ItemsData } from "./base";

/**
 * A group of a table's columns, by the conventions of its format (see
 * `TableData.getColumnSections`)
 *
 * A section without subsections is a leaf: a set of columns searched as one,
 * e.g. the features of one expression matrix. Pass its `id` as
 * `sectionId` (see `TableData.suggestColumnQueries`). A section with
 * subsections gathers leaves that hold the same kind of columns, e.g. the
 * features of AnnData in `X`, a layer and `raw`. Names are for display only.
 */
export type TableColumnSection = {
  /**
   * The key of the section, unique among the table's sections and
   * stable across reopening the table
   */
  id: string;

  /** The name to show, e.g. "Features" */
  name: string;

  /** The sections below this one, if it is no leaf */
  subsections?: TableColumnSection[];
};

/** How a table column's values are used: as categories, or as numbers */
export type TableColumnType = "categorical" | "numerical";

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
   * Whether the query continues into a group of columns, e.g. a path prefix
   *
   * A group suggestion does not resolve to a column; it continues the query,
   * e.g. into the columns of a group or a matrix. Other suggestions are
   * expected to resolve to an exact column name.
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
   * Within a leaf section (see {@link getColumnSections}), the suggestions
   * are the leaf's columns, and the current query is matched against their
   * names.
   *
   * @param currentQuery - The partial column query, or the text to search for
   * within a leaf section
   * @param options - The `id` of a leaf section from
   * {@link getColumnSections} (`sectionId`), if any, and an optional abort
   * signal
   * @returns A list of suggested column queries
   */
  suggestColumnQueries(
    currentQuery: string,
    options?: { sectionId?: string; signal?: AbortSignal },
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
   * Returns the sections of the table's columns, by the conventions of its
   * format
   *
   * Only implemented by table data whose format groups its columns (e.g. the
   * observations and the features of AnnData); callers without it pick columns
   * by query (see {@link suggestColumnQueries}).
   *
   * @returns The top-level sections, in the canonical order of the format;
   * none if the table has no sections
   */
  getColumnSections?(): TableColumnSection[];

  /**
   * Loads the leaf section of a column and the column's name in it (see
   * {@link getColumnSections})
   *
   * Only implemented by table data that implements {@link getColumnSections}.
   * Async, as naming a column may take a read (e.g. gene names from a column
   * of `var`).
   *
   * @param column - The column name
   * @param options - Optional abort signal
   * @returns The `id` of the column's leaf section and the column's name,
   * or `null` if the column belongs to no section
   */
  describeColumn?(
    column: string,
    options?: { signal?: AbortSignal },
  ): Promise<{ sectionId: string; name: string } | null>;

  /**
   * Loads the column of another leaf section that holds the same entity
   * (e.g. the same gene in another expression matrix), as far as the format
   * tells
   *
   * Only implemented by table data that implements {@link getColumnSections}.
   * Columns map only between the leaves below one top-level section; names
   * are never used to decide that.
   *
   * @param column - The column name
   * @param options - The `id` of the leaf section to map the column to
   * (`sectionId`) and an optional abort signal
   * @returns The column of the leaf section, or `null` if it has none for
   * the entity
   * @throws Error if the table has no leaf section with the given `id`
   */
  mapColumn?(
    column: string,
    options: { sectionId: string; signal?: AbortSignal },
  ): Promise<string | null>;

  /**
   * Returns the type of a column, as its encoding declares
   *
   * Only implemented by table data whose format declares column types that the
   * loaded values do not reveal (e.g. AnnData categoricals with numeric
   * categories); callers without it, or given `undefined`, inspect the values
   * (see `TableUtils.loadColumnType`).
   *
   * @param column - The column name
   * @returns The column type, or `undefined` if the format does not say
   */
  getColumnType?(column: string): TableColumnType | undefined;

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
