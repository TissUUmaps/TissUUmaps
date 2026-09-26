import type { TableColumnQuerySuggestion } from "@tissuumaps/core";

import type { HierarchicalTableColumn } from "./HierarchicalTable";

/** A matrix column, the only kind a bracketed selector can address */
type MatrixColumn = Extract<HierarchicalTableColumn, { kind: "matrix" }>;

/**
 * Column query parsing, suggestion and resolution for hierarchical tables
 *
 * A column query is a path such as `obs/area`, optionally followed by a
 * bracketed selector for matrix columns, such as `obsm/spatial[0]`. The
 * selector is a column index, or one of the column's selectors where the
 * matrix has them, such as `X[CD3]` for an AnnData expression matrix. A
 * leading slash is accepted and ignored.
 *
 * Paths and selectors are matched exactly, or else ignoring case if that
 * matches exactly one of them.
 */
export class ColumnQueryUtils {
  /**
   * A path, optionally with a leading slash and followed by a bracketed
   * selector
   */
  private static readonly _columnQueryPattern =
    /^\/?([^[\]]*?)(?:\[([^[\]]*)\])?$/;

  /** A selector that is a column index rather than a column name */
  private static readonly _indexSelectorPattern = /^\d+$/;

  /** Brackets, which delimit the selector of a column query */
  private static readonly _bracketPattern = /[[\]]/;

  /**
   * Whether a name can be part of a column query, as a path segment or as a
   * selector
   *
   * @param name - The name of a node or of a matrix column
   * @returns Whether the name contains no bracket
   */
  static isQueryableName(name: string): boolean {
    return !ColumnQueryUtils._bracketPattern.test(name);
  }

  /**
   * Derives the selectors of the columns of a matrix from their names
   *
   * A name is its column's selector if it is unique, non-empty, not a number
   * and free of brackets; any other column is selected by its index. A
   * numeric selector then always means an index.
   *
   * @param names - The name of each matrix column
   * @returns The selector of each matrix column
   */
  static getMatrixSelectors(names: string[]): string[] {
    const counts = new Map<string, number>();
    for (const name of names) {
      counts.set(name, (counts.get(name) ?? 0) + 1);
    }
    return names.map((name, i) =>
      name === "" ||
      ColumnQueryUtils._indexSelectorPattern.test(name) ||
      !ColumnQueryUtils.isQueryableName(name) ||
      counts.get(name)! > 1
        ? String(i)
        : name,
    );
  }

  /**
   * Resolves a column query to the column it addresses
   *
   * @param columns - The columns of the table
   * @param query - The column query
   * @returns The column and, for a matrix column, the index of the selected
   * matrix column; `null` if the query addresses no column
   */
  static resolveColumn(
    columns: HierarchicalTableColumn[],
    query: string,
  ): { column: HierarchicalTableColumn; index: number | undefined } | null {
    const match = ColumnQueryUtils._columnQueryPattern.exec(query);
    if (match === null) {
      return null;
    }
    const [, path, selector] = match;
    const column =
      columns[
        ColumnQueryUtils._findMatchIndex(
          columns.map((column) => column.path),
          path!,
        )
      ];
    if (column === undefined) {
      return null;
    }
    if (column.kind === "dataset") {
      return selector === undefined ? { column, index: undefined } : null;
    }
    if (selector === undefined) {
      return null;
    }
    const index = ColumnQueryUtils._resolveMatrixSelector(column, selector);
    return index !== undefined ? { column, index } : null;
  }

  /**
   * Resolves a column query to its canonical form
   *
   * @param columns - The columns of the table
   * @param query - The column query
   * @returns The canonical query (`path` for dataset columns, `path[selector]`
   * for matrix columns), or `null` if the query addresses no column
   */
  static resolveColumnQuery(
    columns: HierarchicalTableColumn[],
    query: string,
  ): string | null {
    const resolved = ColumnQueryUtils.resolveColumn(columns, query);
    if (resolved === null) {
      return null;
    }
    const { column, index } = resolved;
    if (column.kind === "dataset") {
      return column.path;
    }
    return `${column.path}[${column.selectors?.[index!] ?? index}]`;
  }

  /**
   * Suggests column queries matching the current query
   *
   * The children of the query's parent path whose name contains its partial
   * name, ignoring case, are suggested: groups with a trailing slash, dataset
   * columns as their path, and matrix columns as the `path[selector]` queries
   * matching the typed selector.
   *
   * Groups and matrix columns suggested without a selector are group
   * suggestions: a group continues into its children, a matrix column into
   * its bracket.
   *
   * Names equal to the partial name come first, then names starting with it,
   * then names containing it, each in path order.
   *
   * @param columns - The columns of the table
   * @param currentQuery - The partial column query
   * @returns The suggested column queries
   */
  static suggestColumnQueries(
    columns: HierarchicalTableColumn[],
    currentQuery: string,
  ): TableColumnQuerySuggestion[] {
    const query = currentQuery.startsWith("/")
      ? currentQuery.slice(1)
      : currentQuery;
    const bracket = query.indexOf("[");
    const path = bracket >= 0 ? query.slice(0, bracket) : query;
    const partialSelector =
      bracket >= 0 ? query.slice(bracket + 1).replace("]", "") : undefined;
    const lastSlash = path.lastIndexOf("/");
    const prefix = path.slice(0, lastSlash + 1).toLowerCase();
    const partialName = path.slice(lastSlash + 1).toLowerCase();

    // by the rank of their name, see `_rankMatch`
    const rankedSuggestions: TableColumnQuerySuggestion[][] = [[], [], []];
    const seenNames = new Set<string>();
    for (const column of columns) {
      if (!column.path.toLowerCase().startsWith(prefix)) {
        continue;
      }
      const columnPrefix = column.path.slice(0, prefix.length);
      const rest = column.path.slice(prefix.length);
      const slash = rest.indexOf("/");
      const name = slash >= 0 ? rest.slice(0, slash) : rest;
      if (seenNames.has(name)) {
        continue;
      }
      const rank = ColumnQueryUtils._rankMatch(name, partialName);
      if (slash < 0 && column.kind === "matrix" && rank === 0) {
        seenNames.add(name);
        // concatenated rather than spread, as a matrix can have more columns
        // than a call takes arguments
        rankedSuggestions[0] = rankedSuggestions[0]!.concat(
          ColumnQueryUtils._suggestMatrixColumns(column, partialSelector),
        );
      } else if (partialSelector === undefined && rank >= 0) {
        seenNames.add(name);
        // a matrix column only addresses a column with a bracketed selector
        rankedSuggestions[rank]!.push(
          slash >= 0
            ? { query: `${columnPrefix}${name}/`, group: true }
            : column.kind === "matrix"
              ? { query: column.path, group: true }
              : { query: column.path },
        );
      }
    }
    return rankedSuggestions.flat();
  }

  /**
   * Ranks how well a name matches a partial name, ignoring case
   *
   * @param name - The name of a node or the selector of a matrix column
   * @param partialName - The partially typed name, in lower case
   * @returns 0 if the name equals the partial name, 1 if it starts with it, 2
   * if it contains it, or -1 if it does not match
   */
  private static _rankMatch(name: string, partialName: string): number {
    const lowerCaseName = name.toLowerCase();
    if (lowerCaseName === partialName) {
      return 0;
    }
    if (lowerCaseName.startsWith(partialName)) {
      return 1;
    }
    return lowerCaseName.includes(partialName) ? 2 : -1;
  }

  /**
   * Finds a key exactly, or else ignoring case if exactly one key matches that
   * way
   *
   * @param keys - The keys to search
   * @param key - The key to find
   * @returns The index of the matching key, or -1 if there is none
   */
  private static _findMatchIndex(keys: string[], key: string): number {
    const exactIndex = keys.indexOf(key);
    if (exactIndex >= 0) {
      return exactIndex;
    }
    const lowerCaseKey = key.toLowerCase();
    let matchIndex = -1;
    for (let i = 0; i < keys.length; i++) {
      if (keys[i]!.toLowerCase() === lowerCaseKey) {
        if (matchIndex >= 0) {
          return -1;
        }
        matchIndex = i;
      }
    }
    return matchIndex;
  }

  /**
   * Resolves the selector of a matrix column to a column index
   *
   * @param column - The matrix column
   * @param selector - The bracketed selector of the query
   * @returns The column index, or `undefined` if the selector addresses no
   * column of the matrix
   */
  private static _resolveMatrixSelector(
    column: MatrixColumn,
    selector: string,
  ): number | undefined {
    if (ColumnQueryUtils._indexSelectorPattern.test(selector)) {
      const index = Number(selector);
      return index < column.numColumns ? index : undefined;
    }
    const { selectors } = column;
    if (selectors === undefined) {
      return undefined;
    }
    const index = ColumnQueryUtils._findMatchIndex(selectors, selector);
    return index >= 0 ? index : undefined;
  }

  /**
   * Suggests the columns of a matrix
   *
   * Columns with selectors are matched by their selector, ignoring case, and
   * ranked like names (see `_rankMatch`). The others are matched by the digits
   * of their index, in index order.
   *
   * @param column - The matrix column
   * @param partialSelector - The partially typed selector, or `undefined` if
   * the query has no brackets yet
   * @returns The suggested `path[selector]` queries
   */
  private static _suggestMatrixColumns(
    column: MatrixColumn,
    partialSelector: string | undefined,
  ): TableColumnQuerySuggestion[] {
    const { selectors } = column;
    if (selectors !== undefined) {
      const partial = (partialSelector ?? "").toLowerCase();
      const rankedSelectors: string[][] = [[], [], []];
      for (const selector of selectors) {
        const rank = ColumnQueryUtils._rankMatch(selector, partial);
        if (rank >= 0) {
          rankedSelectors[rank]!.push(selector);
        }
      }
      return rankedSelectors
        .flat()
        .map((selector) => ({ query: `${column.path}[${selector}]` }));
    }
    const suggestions: TableColumnQuerySuggestion[] = [];
    for (let i = 0; i < column.numColumns; i++) {
      if (
        partialSelector === undefined ||
        String(i).startsWith(partialSelector)
      ) {
        suggestions.push({ query: `${column.path}[${i}]` });
      }
    }
    return suggestions;
  }
}
