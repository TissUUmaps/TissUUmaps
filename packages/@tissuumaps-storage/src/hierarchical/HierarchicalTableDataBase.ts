import type {
  IDArray,
  ProgressCallback,
  TableColumnDataType,
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
  HierarchicalTableQueryGroupLeaf,
} from "./HierarchicalTable";

/**
 * The {@link TableData} of a hierarchical table; owns the table and closes it
 *
 * Extended by every container format, see
 * {@link HierarchicalTableDataProviderBase.createTableData}.
 *
 * The store's column groups are offered as query groups (see
 * {@link loadQueryGroups}): a column group with a single leaf is a top-level
 * leaf, named by the column group, and any other is a node of its leaves. A
 * leaf is either a group of the store whose dataset columns lie directly below
 * it (e.g. the AnnData `obs`), or a matrix whose columns are its features
 * (e.g. `X`). The columns of a matrix leaf are named by their selectors.
 * Columns are matched by selector (see {@link mapColumn}).
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
   * Suggests column queries for the current query, or the columns of a query group
   *
   * Without a query group, the query is continued as a path (see
   * {@link ColumnQueryUtils.suggestColumnQueries}). Within a query group, its
   * columns are suggested with their names, best name matches first (see
   * {@link ColumnQueryUtils.rankByName}).
   *
   * @param currentQuery - The partial column query, or the text to search for
   * within the query group
   * @param options - The query of the query group, if any, and an abort signal
   * @returns A list of suggested column queries
   * @throws Error if the query group does not exist
   */
  suggestColumnQueries(
    currentQuery: string,
    options?: { groupId?: string; signal?: AbortSignal },
  ): Promise<TableColumnQuerySuggestion[]> {
    const { groupId, signal } = options ?? {};
    // not async (see below): an aborted signal rejects
    if (signal?.aborted) {
      return Promise.reject(signal.reason as Error);
    }
    if (groupId === undefined) {
      return Promise.resolve(
        ColumnQueryUtils.suggestColumnQueries(
          this._table.columns,
          currentQuery,
        ),
      );
    }
    const found = this._findLeaf(groupId);
    if (found === undefined) {
      return Promise.reject(new Error(`No query group "${groupId}"`));
    }
    const suggestions =
      this._getMatrix(found.leaf) !== undefined
        ? this._suggestMatrixColumns(found.leaf)
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
    // not async, as nothing is awaited: an aborted signal rejects the
    // returned promise instead of throwing synchronously
    if (signal?.aborted) {
      return Promise.reject(signal.reason as Error);
    }
    return Promise.resolve(
      ColumnQueryUtils.resolveColumnQuery(this._table.columns, query),
    );
  }

  loadQueryGroups(options?: {
    signal?: AbortSignal;
  }): Promise<TableColumnQueryGroup[]> {
    const { signal } = options ?? {};
    // not async (see above): an aborted signal rejects
    if (signal?.aborted) {
      return Promise.reject(signal.reason as Error);
    }
    const toLeaf = (
      { id }: HierarchicalTableQueryGroupLeaf,
      name: string,
    ): TableColumnQueryGroup => ({ name, id });
    return Promise.resolve(
      this._table.columnGroups.map(({ name, subgroups }) =>
        subgroups.length === 1
          ? toLeaf(subgroups[0]!, name)
          : {
              name,
              subgroups: subgroups.map((leaf) => toLeaf(leaf, leaf.name)),
            },
      ),
    );
  }

  describeColumn(
    column: string,
    options?: { signal?: AbortSignal },
  ): Promise<{ groupId: string; name: string } | null> {
    const { signal } = options ?? {};
    // not async (see above): an aborted signal rejects
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
      groupId: leaf.id,
      name:
        tableColumn.kind === "matrix"
          ? (tableColumn.selectors?.[index!] ?? String(index))
          : tableColumn.path.slice(leaf.id.length + 1),
    });
  }

  /**
   * Maps a column to the column of another leaf holding the same entity
   *
   * Columns map only between the leaves of one column group. Within a leaf of
   * columns, a column is its own match if it belongs to the leaf. Within a
   * matrix leaf, a matrix column maps to the column of the leaf with the same
   * selector: a selector names the same feature in every matrix of a column
   * group (e.g. a gene of the AnnData `var` index in `X`, the layers and
   * `raw/X`). Matrices without selectors, or columns selected by their index
   * for want of a usable name, map only between matrices whose selectors are
   * the same, i.e. by position over the same features.
   *
   * @param column - The column name
   * @param options - The query of the leaf to map the column to, and an
   * optional abort signal
   * @returns The column of the leaf, or `null` if it has none for the entity
   * @throws Error if the query group does not exist
   */
  mapColumn(
    column: string,
    options: { groupId: string; signal?: AbortSignal },
  ): Promise<string | null> {
    const { groupId, signal } = options;
    // not async (see above): an aborted signal rejects
    if (signal?.aborted) {
      return Promise.reject(signal.reason as Error);
    }
    if (this._findLeaf(groupId) === undefined) {
      return Promise.reject(new Error(`No query group "${groupId}"`));
    }
    return Promise.resolve(this._mapToLeaf(column, groupId));
  }

  loadColumnDataType(
    column: string,
    options?: { signal?: AbortSignal },
  ): Promise<TableColumnDataType | undefined> {
    const { signal } = options ?? {};
    // not async (see above): an aborted signal rejects
    if (signal?.aborted) {
      return Promise.reject(signal.reason as Error);
    }
    const resolved = ColumnQueryUtils.resolveColumn(
      this._table.columns,
      column,
    );
    return Promise.resolve(
      resolved?.column.kind === "dataset"
        ? resolved.column.dataType
        : undefined,
    );
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
   * @param query - The query of a query group
   * @returns The query group and its group, if a group has the query group
   */
  private _findLeaf(id: string):
    | {
        group: HierarchicalTableQueryGroup;
        leaf: HierarchicalTableQueryGroupLeaf;
      }
    | undefined {
    for (const group of this._table.columnGroups) {
      const leaf = group.subgroups.find((leaf) => leaf.id === id);
      if (leaf !== undefined) {
        return { group, leaf };
      }
    }
    return undefined;
  }

  /**
   * @param leaf - A query group
   * @returns The matrix column of a matrix query group, `undefined` for a query group of
   * columns
   */
  private _getMatrix(
    leaf: HierarchicalTableQueryGroupLeaf,
  ): Extract<HierarchicalTableColumn, { kind: "matrix" }> | undefined {
    const column = this._table.columns.find(
      (column) => column.path === leaf.id,
    );
    return column?.kind === "matrix" ? column : undefined;
  }

  /**
   * @param column - A column of the table
   * @returns The query group holding the column and its group, if any
   */
  private _findLeafOfColumn(column: HierarchicalTableColumn):
    | {
        group: HierarchicalTableQueryGroup;
        leaf: HierarchicalTableQueryGroupLeaf;
      }
    | undefined {
    for (const group of this._table.columnGroups) {
      for (const leaf of group.subgroups) {
        const isInQueryGroup =
          column.kind === "matrix"
            ? column.path === leaf.id
            : this._getMatrix(leaf) === undefined &&
              column.path.startsWith(`${leaf.id}/`) &&
              !column.path.slice(leaf.id.length + 1).includes("/") &&
              column.path !== leaf.indexPath;
        if (isInQueryGroup) {
          return { group, leaf };
        }
      }
    }
    return undefined;
  }

  private _mapToLeaf(query: string, groupId: string): string | null {
    const target = this._findLeaf(groupId);
    const resolved = ColumnQueryUtils.resolveColumn(this._table.columns, query);
    if (target === undefined || resolved === null) {
      return null;
    }
    const { column, index } = resolved;
    const origin = this._findLeafOfColumn(column);
    // columns only map between the query groups of one group
    if (origin === undefined || origin.group !== target.group) {
      return null;
    }
    const targetMatrix = this._getMatrix(target.leaf);
    if (column.kind !== "matrix" || targetMatrix === undefined) {
      return origin.leaf === target.leaf ? column.path : null;
    }
    let targetIndex: number | undefined;
    const sameSelectors =
      column.numColumns === targetMatrix.numColumns &&
      (column.selectors === targetMatrix.selectors ||
        (column.selectors !== undefined &&
          targetMatrix.selectors !== undefined &&
          column.selectors.every(
            (selector, i) => selector === targetMatrix.selectors![i],
          )));
    if (sameSelectors) {
      targetIndex = index;
    } else if (
      column.selectors !== undefined &&
      targetMatrix.selectors !== undefined
    ) {
      const selector = column.selectors[index!]!;
      // a column selected by its index has no name to carry over
      if (selector !== String(index)) {
        const i = targetMatrix.selectors.indexOf(selector);
        targetIndex = i >= 0 ? i : undefined;
      }
    }
    return targetIndex !== undefined
      ? `${targetMatrix.path}[${targetMatrix.selectors?.[targetIndex] ?? targetIndex}]`
      : null;
  }

  /**
   * @param leaf - A query group of columns
   * @returns The dataset columns directly below the query group, but for its
   * index, named as such
   */
  private _suggestLeafColumns(
    leaf: HierarchicalTableQueryGroupLeaf,
  ): TableColumnQuerySuggestion[] {
    const prefix = `${leaf.id}/`;
    return this._table.columns
      .filter(
        (column) =>
          column.kind === "dataset" &&
          column.path.startsWith(prefix) &&
          !column.path.slice(prefix.length).includes("/") &&
          column.path !== leaf.indexPath,
      )
      .map((column) => ({
        query: column.path,
        name: column.path.slice(prefix.length),
      }));
  }

  /**
   * @param leaf - A matrix query group
   * @returns The columns of the leaf's matrix, named by their selectors
   */
  private _suggestMatrixColumns(
    leaf: HierarchicalTableQueryGroupLeaf,
  ): TableColumnQuerySuggestion[] {
    const matrix = this._getMatrix(leaf);
    if (matrix === undefined) {
      return [];
    }
    return Array.from({ length: matrix.numColumns }, (_, i) => {
      const selector = matrix.selectors?.[i] ?? String(i);
      return { query: `${matrix.path}[${selector}]`, name: selector };
    });
  }
}
