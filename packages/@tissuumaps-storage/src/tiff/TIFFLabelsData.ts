import type { TiffRaster } from "geotiff-tilesource";
import type OpenSeadragon from "openseadragon";

import type {
  CustomTileSource,
  IntOrUintArray,
  LabelsData,
  TileSourceConfig,
} from "@tissuumaps/core";

import { tiffRasterType } from "./installTIFFTileSource";

/**
 * The loaded label mask of a TIFF file
 *
 * The tiles carry the label IDs as they are stored in the file, which the
 * renderer colors by looking each ID up in the table annotating the mask, if
 * any, and resolves as they are drawn otherwise.
 */
export class TIFFLabelsData implements LabelsData {
  private readonly _tileSource: OpenSeadragon.TileSource;

  /**
   * @param tileSource - The tile source of the label mask
   */
  constructor(tileSource: OpenSeadragon.TileSource) {
    this._tileSource = tileSource;
  }

  /** Returns the tile source of the label mask */
  getTileSource(): string | TileSourceConfig | CustomTileSource {
    return this._tileSource;
  }

  /**
   * Extracts the label IDs of a tile from a tile invalidation event
   *
   * The IDs are returned as they are, in the raster's own size (see
   * {@link TiffRaster}), not cropped to the tile's bounds. They are the cached
   * raster's own band rather than a copy (see `copyRasters` in
   * `installTIFFTileSource`), so callers must not modify them.
   *
   * @param event - The tile invalidation event
   * @returns The label IDs of the invalidated tile, one per raster pixel in
   * row-major order, along with the width and height of the raster
   * @throws Error if the tile's raster has no band
   */
  async getTileData(event: OpenSeadragon.TileInvalidatedEvent): Promise<{
    values: IntOrUintArray;
    width: number;
    height: number;
  }> {
    const raster = (await event.getData(tiffRasterType)) as TiffRaster;
    const band = raster.bands[0];
    if (band === undefined) {
      throw new Error("The tile's raster has no bands");
    }
    // the provider opens integer files of at most 32 bits only, whose bands
    // geotiff.js decodes into a signed or unsigned integer array
    return {
      values: band as IntOrUintArray,
      width: raster.width,
      height: raster.height,
    };
  }

  /** Closing does nothing: the decoder pool is shared, and the file is read on demand */
  close(): void {
    // nothing to release
  }
}
