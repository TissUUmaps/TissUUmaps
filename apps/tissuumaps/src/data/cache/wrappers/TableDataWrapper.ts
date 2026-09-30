import type {
  IDArray,
  ProgressCallback,
  TableColumnQuerySuggestion,
  TableData,
  TypedArrayOrArray,
} from "@tissuumaps/core";

import { SharedOperation } from "../SharedOperation";
import { DataWrapperBase } from "./DataWrapperBase";

/**
 * Cache wrapper around table data, sharing the loaded columns
 *
 * Values, unique value counts and value ranges are each loaded once per
 * column and then kept for as long as this wrapper lives; column queries are
 * delegated to the wrapped data unchanged. The optional names getter is only
 * provided if the wrapped data provides it.
 */
export class TableDataWrapper
  extends DataWrapperBase<TableData>
  implements TableData
{
  private readonly _loadValuesOps = new Map<
    string,
    SharedOperation<TypedArrayOrArray<unknown>>
  >();
  private readonly _loadUniqueValueCountsOps = new Map<
    string,
    SharedOperation<Map<unknown, number>>
  >();
  private readonly _loadValueRangeOps = new Map<
    string,
    SharedOperation<[number, number] | undefined>
  >();

  readonly getNames?: TableData["getNames"];

  constructor(data: TableData) {
    super(data);
    if (data.getNames !== undefined) {
      this.getNames = () => this.data.getNames!();
    }
  }

  getIds(): IDArray {
    return this.data.getIds();
  }

  getSize(): number {
    return this.data.getSize();
  }

  suggestColumnQueries(
    currentQuery: string,
    options?: { signal?: AbortSignal },
  ): Promise<TableColumnQuerySuggestion[]> {
    return this.data.suggestColumnQueries(currentQuery, options);
  }

  resolveColumnQuery(
    query: string,
    options?: { signal?: AbortSignal },
  ): Promise<string | null> {
    return this.data.resolveColumnQuery(query, options);
  }

  /**
   * Loads a column's values, sharing one load operation per column between all
   * callers
   *
   * @param column - The name of the column to load the values of
   * @param options - Optional abort signal and progress callback
   * @returns A promise that resolves to the column's values, or rejects if the
   * wrapper has been destroyed
   */
  loadValues<T>(
    column: string,
    options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
  ): Promise<TypedArrayOrArray<T>> {
    if (this.destroyed) {
      return Promise.reject(new Error("Data has been destroyed"));
    }
    let op = this._loadValuesOps.get(column);
    if (op === undefined || op.failed) {
      const newOp = new SharedOperation((opts) =>
        this.data.loadValues(column, opts),
      );
      newOp.signal.addEventListener(
        "abort",
        () => {
          if (this._loadValuesOps.get(column) === newOp) {
            this._loadValuesOps.delete(column);
          }
        },
        { once: true },
      );
      this._loadValuesOps.set(column, newOp);
      op = newOp;
    }
    return op.subscribe(options) as Promise<TypedArrayOrArray<T>>;
  }

  /**
   * Loads a column's unique value counts, sharing one load operation per
   * column between all callers
   *
   * @param column - The name of the column to load the unique value counts of
   * @param options - Optional abort signal and progress callback
   * @returns A promise that resolves to the row count of every unique value of
   * the column, or rejects if the wrapper has been destroyed
   */
  loadUniqueValueCounts<T>(
    column: string,
    options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
  ): Promise<Map<T, number>> {
    if (this.destroyed) {
      return Promise.reject(new Error("Data has been destroyed"));
    }
    let op = this._loadUniqueValueCountsOps.get(column);
    if (op === undefined || op.failed) {
      const newOp = new SharedOperation((opts) =>
        this.data.loadUniqueValueCounts(column, opts),
      );
      newOp.signal.addEventListener(
        "abort",
        () => {
          if (this._loadUniqueValueCountsOps.get(column) === newOp) {
            this._loadUniqueValueCountsOps.delete(column);
          }
        },
        { once: true },
      );
      this._loadUniqueValueCountsOps.set(column, newOp);
      op = newOp;
    }
    return op.subscribe(options) as Promise<Map<T, number>>;
  }

  /**
   * Loads the range of a column's numeric values, sharing one load operation
   * per column between all callers
   *
   * @param column - The name of the column to load the value range of
   * @param options - Optional abort signal and progress callback
   * @returns A promise that resolves to the column's minimum and maximum value,
   * to `undefined` if the column is not numeric, or rejects if the wrapper has
   * been destroyed
   */
  loadValueRange(
    column: string,
    options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
  ): Promise<[number, number] | undefined> {
    if (this.destroyed) {
      return Promise.reject(new Error("Data has been destroyed"));
    }
    let op = this._loadValueRangeOps.get(column);
    if (op === undefined || op.failed) {
      const newOp = new SharedOperation((opts) =>
        this.data.loadValueRange(column, opts),
      );
      newOp.signal.addEventListener(
        "abort",
        () => {
          if (this._loadValueRangeOps.get(column) === newOp) {
            this._loadValueRangeOps.delete(column);
          }
        },
        { once: true },
      );
      this._loadValueRangeOps.set(column, newOp);
      op = newOp;
    }
    return op.subscribe(options);
  }

  /**
   * Aborts all pending column load operations and destroys the wrapper
   */
  override destroy(): void {
    for (const op of [
      ...this._loadValuesOps.values(),
      ...this._loadUniqueValueCountsOps.values(),
      ...this._loadValueRangeOps.values(),
    ]) {
      op.abort();
    }
    super.destroy();
  }
}
