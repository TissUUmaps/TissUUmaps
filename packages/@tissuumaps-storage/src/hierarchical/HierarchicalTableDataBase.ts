import type {
  IDArray,
  ProgressCallback,
  TableColumnQuerySuggestion,
  TableData,
  TypedArrayOrArray,
} from "@tissuumaps/core";

import { ColumnQueryUtils } from "./ColumnQueryUtils";
import type { HierarchicalTable } from "./HierarchicalTable";

/**
 * The {@link TableData} of a hierarchical table; owns the table and closes it
 *
 * Extended by every container format, see
 * {@link HierarchicalTableDataProviderBase.createTableData}.
 */
export abstract class HierarchicalTableDataBase implements TableData {
  private readonly _table: HierarchicalTable;
  private readonly _numRows: number;
  private _ids: IDArray | undefined;
  private readonly _names: string[] | undefined;

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

  suggestColumnQueries(
    currentQuery: string,
    options?: { signal?: AbortSignal },
  ): Promise<TableColumnQuerySuggestion[]> {
    const { signal } = options ?? {};
    if (signal?.aborted) {
      return Promise.reject(signal.reason as Error);
    }
    return Promise.resolve(
      ColumnQueryUtils.suggestColumnQueries(this._table.columns, currentQuery),
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
}
