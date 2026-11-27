import {
  AsyncUtils,
  type IDArray,
  type ProgressCallback,
  type TableColumnQuerySuggestion,
  type TableColumnType,
  type TableData,
  type TypedArrayOrArray,
} from "@tissuumaps/core";

import type { CoordinateColumn } from "./profiles/GeoParquetUtils";
import { runParquetWorker } from "./runParquetWorker";
import type { ParquetSource } from "./types";

/**
 * A table read from one or more Parquet files, such as the partitions of a
 * Dask dataset, whose rows follow one another in the order of the files
 */
export class ParquetTableData implements TableData {
  private readonly _parts: ParquetSource[];
  private readonly _numRows: number;
  private readonly _columns: string[];
  // The coordinate columns of the file, by name: a column is derived only if
  // the reader said so, never because its name looks derived.
  private readonly _coordinateColumns: Map<string, CoordinateColumn>;
  private _ids: IDArray | undefined;
  private readonly _names: string[] | undefined;
  // the columns of a pandas category dtype, see PandasMetadataUtils
  private readonly _categoricalColumns: Set<string>;
  // Both axes of a point geometry column are decoded in one pass, so the
  // second axis a point cloud reads does not decode the column again.
  private readonly _coordinates = new Map<
    string,
    Promise<{ x: Float32Array; y: Float32Array }>
  >();

  constructor(
    parts: ParquetSource[],
    numRows: number,
    columns: string[],
    coordinateColumns: CoordinateColumn[],
    ids: IDArray | undefined,
    names: string[] | undefined,
    categoricalColumns: string[],
  ) {
    this._parts = parts;
    this._numRows = numRows;
    this._columns = columns;
    this._coordinateColumns = new Map(
      coordinateColumns.map((coordinateColumn) => [
        coordinateColumn.column,
        coordinateColumn,
      ]),
    );
    this._ids = ids;
    this._names = names;
    this._categoricalColumns = new Set(categoricalColumns);
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

  getColumnType(column: string): TableColumnType | undefined {
    return this._categoricalColumns.has(column) ? "categorical" : undefined;
  }

  suggestColumnQueries(
    currentQuery: string,
  ): Promise<TableColumnQuerySuggestion[]> {
    const lowerCaseQuery = currentQuery.toLowerCase();
    const exactMatches: string[] = [];
    const prefixMatches: string[] = [];
    const matches: string[] = [];
    const others: string[] = [];
    for (const column of this._columns) {
      const lowerCaseColumn = column.toLowerCase();
      if (column === currentQuery) {
        exactMatches.push(column);
      } else if (lowerCaseColumn.startsWith(lowerCaseQuery)) {
        prefixMatches.push(column);
      } else if (lowerCaseColumn.includes(lowerCaseQuery)) {
        matches.push(column);
      } else {
        others.push(column);
      }
    }
    return Promise.resolve(
      [...exactMatches, ...prefixMatches, ...matches, ...others].map(
        (query) => ({ query }),
      ),
    );
  }

  resolveColumnQuery(query: string): Promise<string | null> {
    if (this._columns.includes(query)) {
      return Promise.resolve(query);
    }
    const lowerCaseQuery = query.toLowerCase();
    const matches = this._columns.filter(
      (column) => column.toLowerCase() === lowerCaseQuery,
    );
    return Promise.resolve(matches.length === 1 ? matches[0]! : null);
  }

  async loadValues<T>(
    column: string,
    options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
  ): Promise<TypedArrayOrArray<T>> {
    const { signal, onProgress } = options ?? {};
    signal?.throwIfAborted();
    const coordinateColumn = this._coordinateColumns.get(column);
    if (coordinateColumn !== undefined) {
      const { x, y } = await AsyncUtils.raceSignal(
        this._loadCoordinates(coordinateColumn.geometryColumn, { onProgress }),
        { signal },
      );
      return (coordinateColumn.axis === "x" ? x : y) as TypedArrayOrArray<T>;
    }
    const { data } = await runParquetWorker(
      { op: "column", parts: this._parts, column },
      { signal, onProgress },
    );
    return data as TypedArrayOrArray<T>;
  }

  // No signal option: the read is shared by both axes, so one caller must not
  // be able to abort it for the other. Callers race it against their own
  // signal instead, and the progress goes to whoever starts the read.
  private _loadCoordinates(
    geometryColumn: string,
    options?: { onProgress?: ProgressCallback },
  ): Promise<{ x: Float32Array; y: Float32Array }> {
    const { onProgress } = options ?? {};
    let coordinates = this._coordinates.get(geometryColumn);
    if (coordinates === undefined) {
      coordinates = runParquetWorker(
        { op: "coordinates", parts: this._parts, geometryColumn },
        { onProgress },
      ).then(({ x, y }) => ({ x, y }));
      coordinates.catch(() => this._coordinates.delete(geometryColumn));
      this._coordinates.set(geometryColumn, coordinates);
    }
    return coordinates;
  }

  async loadValueRange(
    column: string,
    options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
  ): Promise<[number, number] | undefined> {
    const { signal, onProgress } = options ?? {};
    signal?.throwIfAborted();
    const coordinateColumn = this._coordinateColumns.get(column);
    const { range } = await runParquetWorker(
      {
        op: "range",
        parts: this._parts,
        column: coordinateColumn?.geometryColumn ?? column,
        axis: coordinateColumn?.axis,
      },
      { signal, onProgress },
    );
    return range;
  }

  close(): void {
    this._coordinates.clear();
  }
}
