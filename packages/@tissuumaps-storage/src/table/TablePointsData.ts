import type {
  IDArray,
  PointsData,
  PointsGeometry,
  ProgressCallback,
  TableData,
} from "@tissuumaps/core";

export class TablePointsData implements PointsData {
  private readonly _tableData: TableData;
  private readonly _xColumn: string;
  private readonly _yColumn: string;

  constructor(tableData: TableData, xColumn: string, yColumn: string) {
    this._tableData = tableData;
    this._xColumn = xColumn;
    this._yColumn = yColumn;
  }

  getIds(): IDArray {
    return this._tableData.getIds();
  }

  getSize(): number {
    return this._tableData.getSize();
  }

  getNames(): string[] | undefined {
    return this._tableData.getNames?.();
  }

  async loadGeometry(options?: {
    signal?: AbortSignal;
    onProgress?: ProgressCallback;
  }): Promise<PointsGeometry> {
    const { signal, onProgress } = options ?? {};
    signal?.throwIfAborted();
    const xPromise = this._tableData.loadValues<number>(this._xColumn, {
      signal,
      onProgress,
    });
    const yPromise = this._tableData.loadValues<number>(this._yColumn, {
      signal,
      onProgress,
    });
    let [xs, ys] = await Promise.all([xPromise, yPromise]);
    if (!(xs instanceof Float32Array)) {
      xs = new Float32Array(xs);
    }
    if (!(ys instanceof Float32Array)) {
      ys = new Float32Array(ys);
    }
    return { xs, ys };
  }

  close(): void {}
}
