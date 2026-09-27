import type { OMEZarrTileData, OMEZarrTileSource } from "omezarr-tilesource";
import type OpenSeadragon from "openseadragon";

import type {
  CustomTileSource,
  IntOrUintArray,
  LabelsData,
  TileSourceConfig,
} from "@tissuumaps/core";

/**
 * Loaded OME-Zarr label image data
 *
 * Provides a single tile source whose tiles hold label IDs as signed or
 * unsigned integers (see {@link OMEZarrLabelsData.getTileData}). Label IDs are
 * not enumerated up front: they are read per tile, so that arbitrarily large
 * label images can be opened without scanning them.
 */
export class OMEZarrLabelsData implements LabelsData {
  private readonly _tileSource: OMEZarrTileSource;

  /**
   * @param tileSource - The ready tile source of the label image, rendering a
   * single channel
   */
  constructor(tileSource: OMEZarrTileSource) {
    this._tileSource = tileSource;
  }

  /** Returns the tile source of the label image */
  getTileSource(): string | TileSourceConfig | CustomTileSource {
    return this._tileSource;
  }

  /**
   * Extracts the label IDs of an invalidated tile from its raw OME-Zarr data
   *
   * The tile has to belong to an `OMEZarrTileSource` rendering a single
   * channel, whose tile data (`chunks`) holds exactly one two-dimensional
   * (height x width) tile read from the Zarr array (with the channel, z-slice
   * and timepoint already selected) instead of a rendered image.
   *
   * Only integer tiles of up to 32 bits (signed or unsigned) are accepted, as
   * label IDs have to be integers and the renderer resolves them as such;
   * 64-bit integers cannot be represented without loss.
   *
   * @param event - The tile invalidation event
   * @returns The label IDs of the invalidated tile, one per raster pixel in
   * row-major order, along with the width and height of the raster in pixels
   * @throws Error if the tile data does not hold exactly one tile, or if the
   * tile is not an 8-, 16- or 32-bit integer array
   */
  async getTileData(
    event: OpenSeadragon.TileInvalidatedEvent,
  ): Promise<{ values: IntOrUintArray; width: number; height: number }> {
    const data = (await event.getData("ome-zarr")) as OMEZarrTileData;
    if (data.chunks.length !== 1) {
      throw new Error(`Expected a single tile, got ${data.chunks.length}`);
    }
    const tile = data.chunks[0]!;
    if (
      tile.data instanceof Int8Array ||
      tile.data instanceof Int16Array ||
      tile.data instanceof Int32Array ||
      tile.data instanceof Uint8Array ||
      tile.data instanceof Uint16Array ||
      tile.data instanceof Uint32Array
    ) {
      return {
        values: tile.data,
        width: tile.shape[1]!,
        height: tile.shape[0]!,
      };
    }
    throw new Error(`Unsupported data type: ${tile.data.constructor.name}`);
  }

  close(): void {}
}
