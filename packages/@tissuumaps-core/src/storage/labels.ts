import type OpenSeadragon from "openseadragon";

import type { LabelsDataSource } from "../model/labels";
import type { IntOrUintArray } from "../types/arrays";
import type { AnnotatedDataProvider, RasterData } from "./base";

/**
 * Loaded label image data providing a tiled, multi-resolution integer raster
 *
 * Each pixel value represents a label (segment) ID. Unlike points and shapes, a
 * label image does not enumerate its labels: their IDs are only known per
 * tile, as the tiles are read, so that arbitrarily large label images can be
 * opened without scanning them.
 */
export interface LabelsData extends RasterData {
  /**
   * Extracts the label IDs of an invalidated tile
   *
   * Label images always carry values rather than colors, so unlike for
   * {@link RasterData.getTileData} this is not optional, and the values are
   * integers (signed or unsigned, of up to 32 bits).
   *
   * @param event - The tile invalidation event
   * @returns The label IDs of the invalidated tile, one per raster pixel in
   * row-major order, along with the width and height of the raster in pixels
   * @throws Error if the event does not contain label image data
   */
  getTileData(event: OpenSeadragon.TileInvalidatedEvent): Promise<{
    values: IntOrUintArray;
    width: number;
    height: number;
  }>;
}

/**
 * Data provider for label images
 *
 * @typeParam TLabelsDataSource - The data source type this data provider opens
 * @typeParam TLabelsData - The {@link LabelsData} type produced by this data
 * provider
 * @typeParam TNormalizedLabelsDataSource - The normalized data source type
 * produced by `normalize` and accepted by `load`
 */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface LabelsDataProvider<
  TLabelsDataSource extends LabelsDataSource,
  TLabelsData extends LabelsData,
  TNormalizedLabelsDataSource extends TLabelsDataSource = TLabelsDataSource,
> extends AnnotatedDataProvider<
  TLabelsDataSource,
  TLabelsData,
  TNormalizedLabelsDataSource
> {}
