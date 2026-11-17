import type {
  IDArray,
  ProgressCallback,
  TableColumnQueryGroup,
  TableColumnQuerySuggestion,
  TableData,
  TypedArrayOrArray,
} from "@tissuumaps/core";

import { ColumnQueryUtils } from "./ColumnQueryUtils";
import type {
  HierarchicalTable,
  HierarchicalTableColumn,
  HierarchicalTableQueryGroup,
} from "./HierarchicalTable";

/**
 * The {@link TableData} of a hierarchical table; owns the table and closes it
 *
 * Extended by every container format, see
 * {@link HierarchicalTableDataProviderBase.createTableData}.
 *
 * The store's query groups (see {@link getQueryGroups}) come from its format,
 * e.g. AnnData. A leaf is either a group of the store whose dataset columns
 * lie directly below it (e.g. the AnnData `obs`), or a matrix whose columns
 * are named by their selectors (e.g. `X`).
 */
export abstract class HierarchicalTableDataBase implements TableData {
  private readonly _table: HierarchicalTable;
  private readonly _numRows: number;
  private _ids: IDArray | undefined;
  private readonly _names: string[] | undefined;

  /**
   * @param table - The open table, which the data owns and closes
   * @param numRows - The number of rows
   * @param ids - The row IDs, `undefined` for sequential IDs
   * @param names - The row names, if any
   */
  constructor(
    table: HierarchicalTable,
    numRows: number,
    ids: IDArray | undefined,
    names: string[] | undefined,
  ) {
    this._table = table;
    this._numRows = numRows;
    this._ids = ids;
    this._names = names;
  }

  getIds(): IDArray {
    if (this._ids === undefined) {
      console.warn("No ID column specified, using sequential IDs instead");
      this._ids = Uint32Array.from({ length: this.getSize() }, (_, i) => i);
    }
    return this._ids;
  }

  getSize(): number {
    return this._numRows;
  }

  getNames(): string[] | undefined {
    return this._names;
  }

  /**
   * Suggests column queries for the current query, or the columns of a leaf
   * query group
   *
   * Without a query group, the query is continued as a path (see
   * {@link ColumnQueryUtils.suggestColumnQueries}). Within a leaf query group,
   * its columns are suggested with their names, best name matches first (see
   * {@link ColumnQueryUtils.rankByName}).
   *
   * @param currentQuery - The partial column query, or the text to search for
   * within the leaf query group
   * @param options - The `id` of a leaf query group, if any, and an optional
   * abort signal
   * @returns A list of suggested column queries
   * @throws Error if the table has no leaf query group with the given `id`
   */
  suggestColumnQueries(
    currentQuery: string,
    options?: { queryGroupId?: string; signal?: AbortSignal },
  ): Promise<TableColumnQuerySuggestion[]> {
    const { queryGroupId, signal } = options ?? {};
    if (signal?.aborted) {
      return Promise.reject(signal.reason as Error);
    }
    if (queryGroupId === undefined) {
      return Promise.resolve(
        ColumnQueryUtils.suggestColumnQueries(
          this._table.columns,
          currentQuery,
        ),
      );
    }
    const found = this._findLeaf((leaf) => leaf.id === queryGroupId);
    if (found === undefined) {
      return Promise.reject(
        new Error(`Query group '${queryGroupId}' not found`),
      );
    }
    const matrix = this._getMatrix(found.leaf);
    const suggestions =
      matrix !== undefined
        ? this._suggestMatrixColumns(matrix)
        : this._suggestLeafColumns(found.leaf);
    return Promise.resolve(
      ColumnQueryUtils.rankByName(suggestions, currentQuery),
    );
  }

  resolveColumnQuery(
    query: string,
    options?: { signal?: AbortSignal },
  ): Promise<string | null> {
    const { signal } = options ?? {};
    if (signal?.aborted) {
      return Promise.reject(signal.reason as Error);
    }
    return Promise.resolve(
      ColumnQueryUtils.resolveColumnQuery(this._table.columns, query),
    );
  }

  getQueryGroups(): TableColumnQueryGroup[] {
    const toQueryGroup = ({
      id,
      name,
      subgroups,
      unavailable,
    }: HierarchicalTableQueryGroup): TableColumnQueryGroup => ({
      id,
      name,
      ...(subgroups !== undefined && {
        subgroups: subgroups.map(toQueryGroup),
      }),
      ...(unavailable !== undefined && { unavailable }),
    });
    return this._table.queryGroups.map(toQueryGroup);
  }

  describeColumn(
    column: string,
    options?: { signal?: AbortSignal },
  ): Promise<{ queryGroupId: string; name: string } | null> {
    const { signal } = options ?? {};
    if (signal?.aborted) {
      return Promise.reject(signal.reason as Error);
    }
    const resolved = ColumnQueryUtils.resolveColumn(
      this._table.columns,
      column,
    );
    if (resolved === null) {
      return Promise.resolve(null);
    }
    const { column: tableColumn, index } = resolved;
    const found = this._findLeafOfColumn(tableColumn);
    if (found === undefined) {
      return Promise.resolve(null);
    }
    const { leaf } = found;
    return Promise.resolve({
      queryGroupId: leaf.id,
      name:
        tableColumn.kind === "matrix"
          ? (tableColumn.selectors?.[index!] ?? String(index))
          : tableColumn.path.slice(leaf.id.length + 1),
    });
  }

  /**
   * Loads the column of another leaf query group that holds the same entity
   *
   * Within a leaf of dataset columns, a column maps to itself if it belongs
   * to the leaf. Between matrix leaves, a column maps to the column with the
   * same selector: a selector names the same feature in every matrix of a
   * query group (e.g. a gene of the AnnData `var` index in `X`, the layers and
   * `raw/X`). Matrices with the same selectors map by position, so that
   * columns selected by their index map too. Matrices without selectors map
   * to nothing, as nothing tells that their columns hold the same entities.
   *
   * @param column - The column name
   * @param options - The `id` of the leaf query group to map the column to,
   * and an optional abort signal
   * @returns The column of the leaf query group, or `null` if it has none for
   * the entity
   * @throws Error if the table has no leaf query group with the given `id`
   */
  mapColumn(
    column: string,
    options: { queryGroupId: string; signal?: AbortSignal },
  ): Promise<string | null> {
    const { queryGroupId, signal } = options;
    if (signal?.aborted) {
      return Promise.reject(signal.reason as Error);
    }
    const target = this._findLeaf((leaf) => leaf.id === queryGroupId);
    if (target === undefined) {
      return Promise.reject(
        new Error(`Query group '${queryGroupId}' not found`),
      );
    }
    const resolved = ColumnQueryUtils.resolveColumn(
      this._table.columns,
      column,
    );
    if (resolved === null) {
      return Promise.resolve(null);
    }
    const { column: tableColumn, index } = resolved;
    const origin = this._findLeafOfColumn(tableColumn);
    // columns only map between the leaves of one top-level query group
    if (origin === undefined || origin.group !== target.group) {
      return Promise.resolve(null);
    }
    const targetMatrix = this._getMatrix(target.leaf);
    if (tableColumn.kind !== "matrix" || targetMatrix === undefined) {
      return Promise.resolve(
        origin.leaf === target.leaf ? tableColumn.path : null,
      );
    }
    const { selectors } = tableColumn;
    const targetSelectors = targetMatrix.selectors;
    if (selectors === undefined || targetSelectors === undefined) {
      return Promise.resolve(null);
    }
    const isSameSelectors =
      selectors.length === targetSelectors.length &&
      selectors.every((selector, i) => selector === targetSelectors[i]);
    const selector = selectors[index!];
    // a column selected by its index has no name to carry over
    const targetIndex = isSameSelectors
      ? index!
      : selector !== undefined && selector !== String(index)
        ? targetSelectors.indexOf(selector)
        : -1;
    const targetSelector = targetSelectors[targetIndex];
    return Promise.resolve(
      targetSelector !== undefined
        ? `${targetMatrix.path}[${targetSelector}]`
        : null,
    );
  }

  isColumnCategorical(column: string): boolean | undefined {
    const resolved = ColumnQueryUtils.resolveColumn(
      this._table.columns,
      column,
    );
    return resolved?.column.kind === "dataset"
      ? resolved.column.categorical
      : undefined;
  }

  // onProgress is accepted but unused: stores report no byte progress.
  async loadValues<T>(
    column: string,
    options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
  ): Promise<TypedArrayOrArray<T>> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const data = await this._table.readColumn(column, {
      numRows: this._numRows,
      signal,
    });
    return data as TypedArrayOrArray<T>;
  }

  close(): void {
    this._table.close();
  }

  /**
   * @param predicate - Tests a leaf query group
   * @returns The first leaf query group passing the test and the top-level
   * query group above it (or the leaf itself), if any
   */
  private _findLeaf(
    predicate: (leaf: HierarchicalTableQueryGroup) => boolean,
  ):
    | { group: HierarchicalTableQueryGroup; leaf: HierarchicalTableQueryGroup }
    | undefined {
    const getLeaves = (
      queryGroup: HierarchicalTableQueryGroup,
    ): HierarchicalTableQueryGroup[] =>
      queryGroup.subgroups?.flatMap(getLeaves) ?? [queryGroup];
    for (const group of this._table.queryGroups) {
      const leaf = getLeaves(group).find(predicate);
      if (leaf !== undefined) {
        return { group, leaf };
      }
    }
    return undefined;
  }

  /**
   * @param column - A column of the table
   * @returns The leaf query group holding the column and the top-level query
   * group above it, if any
   */
  private _findLeafOfColumn(column: HierarchicalTableColumn) {
    return this._findLeaf((leaf) =>
      column.kind === "matrix"
        ? column.path === leaf.id
        : this._isLeafColumn(column, leaf),
    );
  }

  /**
   * @param leaf - A leaf query group
   * @returns The matrix column of a matrix leaf, `undefined` for a leaf of
   * dataset columns
   */
  private _getMatrix(
    leaf: HierarchicalTableQueryGroup,
  ): Extract<HierarchicalTableColumn, { kind: "matrix" }> | undefined {
    const column = this._table.columns.find(
      (column) => column.path === leaf.id,
    );
    return column?.kind === "matrix" ? column : undefined;
  }

  /**
   * @param column - A column of the table
   * @param leaf - A leaf query group
   * @returns Whether the column is a dataset column directly below the leaf,
   * but for its index
   */
  private _isLeafColumn(
    column: HierarchicalTableColumn,
    leaf: HierarchicalTableQueryGroup,
  ): boolean {
    const prefix = `${leaf.id}/`;
    return (
      column.kind === "dataset" &&
      column.path.startsWith(prefix) &&
      !column.path.slice(prefix.length).includes("/") &&
      column.path !== leaf.indexPath
    );
  }

  /**
   * @param leaf - A leaf query group of dataset columns
   * @returns The leaf's columns, named by their path below the leaf
   */
  private _suggestLeafColumns(
    leaf: HierarchicalTableQueryGroup,
  ): TableColumnQuerySuggestion[] {
    return this._table.columns
      .filter((column) => this._isLeafColumn(column, leaf))
      .map((column) => ({
        query: column.path,
        name: column.path.slice(leaf.id.length + 1),
        ...(column.unavailable !== undefined && {
          unavailable: column.unavailable,
        }),
      }));
  }

  /**
   * @param matrix - The matrix column of a matrix leaf
   * @returns The matrix's columns, named by their selectors
   */
  private _suggestMatrixColumns(
    matrix: Extract<HierarchicalTableColumn, { kind: "matrix" }>,
  ): TableColumnQuerySuggestion[] {
    return Array.from({ length: matrix.numColumns }, (_, i) => {
      const selector = matrix.selectors?.[i] ?? String(i);
      return {
        query: `${matrix.path}[${selector}]`,
        name: selector,
        ...(matrix.unavailable !== undefined && {
          unavailable: matrix.unavailable,
        }),
      };
    });
  }
}
