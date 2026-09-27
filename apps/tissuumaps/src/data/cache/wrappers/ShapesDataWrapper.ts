import type {
  IDArray,
  ProgressCallback,
  ShapesData,
  ShapesGeometry,
} from "@tissuumaps/core";

import { SharedOperation } from "../SharedOperation";
import { DataWrapperBase } from "./DataWrapperBase";

/**
 * Cache wrapper around shapes data, sharing the loaded geometry
 *
 * The optional names getter is only provided if the wrapped data provides it.
 */
export class ShapesDataWrapper
  extends DataWrapperBase<ShapesData>
  implements ShapesData
{
  private _loadGeometryOp?: SharedOperation<ShapesGeometry>;

  readonly getNames?: ShapesData["getNames"];

  constructor(data: ShapesData) {
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

  /**
   * Loads the shapes' geometry, sharing one load operation between all callers
   *
   * The geometry is loaded once and then kept for as long as this wrapper
   * lives. An operation that failed or was abandoned - see
   * {@link SharedOperation.subscribe} - is replaced on the next call.
   *
   * @param options - Optional abort signal and progress callback
   * @returns A promise that resolves to the shapes' geometry, or rejects if the
   * wrapper has been destroyed
   */
  loadGeometry(options?: {
    signal?: AbortSignal;
    onProgress?: ProgressCallback;
  }): Promise<ShapesGeometry> {
    if (this.destroyed) {
      return Promise.reject(new Error("Data has been destroyed"));
    }
    if (this._loadGeometryOp === undefined || this._loadGeometryOp.failed) {
      const newOp = new SharedOperation((opts) => this.data.loadGeometry(opts));
      newOp.signal.addEventListener(
        "abort",
        () => {
          if (this._loadGeometryOp === newOp) {
            this._loadGeometryOp = undefined;
          }
        },
        { once: true },
      );
      this._loadGeometryOp = newOp;
    }
    return this._loadGeometryOp.subscribe(options);
  }

  /**
   * Aborts the pending geometry load operation, if any, and destroys the wrapper
   */
  override destroy(): void {
    this._loadGeometryOp?.abort();
    super.destroy();
  }
}
