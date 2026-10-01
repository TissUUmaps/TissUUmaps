import type {
  CustomTileSource,
  LabelsData,
  TileSourceConfig,
} from "@tissuumaps/core";

import { DataWrapperBase } from "./DataWrapperBase";

/**
 * Cache wrapper around labels data, delegating to the wrapped data
 */
export class LabelsDataWrapper
  extends DataWrapperBase<LabelsData>
  implements LabelsData
{
  getTileSource(): string | TileSourceConfig | CustomTileSource {
    // caching is handled by renderers
    return this.data.getTileSource();
  }

  getTileData(
    event: Parameters<LabelsData["getTileData"]>[0],
  ): ReturnType<LabelsData["getTileData"]> {
    return this.data.getTileData(event);
  }

  loadLabelId(
    x: number,
    y: number,
    options?: { signal?: AbortSignal },
  ): Promise<number | undefined> {
    return this.data.loadLabelId(x, y, options);
  }
}
