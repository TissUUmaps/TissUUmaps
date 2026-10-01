import type { GeoTIFFImage } from "geotiff";
import type { TiffRaster } from "geotiff-tilesource";
import type OpenSeadragon from "openseadragon";

import type {
  CustomTileSource,
  IntOrUintArray,
  LabelsData,
  TileSourceConfig,
} from "@tissuumaps/core";

import { installTIFFTileSource, tiffRasterType } from "./installTIFFTileSource";

/**
 * The loaded label mask of a TIFF file
 *
 * The tiles carry the label IDs as they are stored in the file, which the
 * renderer colors by looking each ID up in the table annotating the mask, if
 * any, and resolves as they are drawn otherwise.
 */
export class TIFFLabelsData implements LabelsData {
  private readonly _tileSource: OpenSeadragon.TileSource;
  private readonly _image: GeoTIFFImage;

  /**
   * @param tileSource - The tile source of the label mask
   * @param image - The full-resolution image of the label mask
   */
  constructor(tileSource: OpenSeadragon.TileSource, image: GeoTIFFImage) {
    this._tileSource = tileSource;
    this._image = image;
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

  /**
   * Loads the label ID at a pixel of the full-resolution label mask
   *
   * Decodes the tile or strip holding the pixel on the shared decoder pool.
   *
   * @param x - Horizontal pixel position; pixel i spans [i, i + 1)
   * @param y - Vertical pixel position; pixel i spans [i, i + 1)
   * @param options - Optional abort signal
   * @returns The label ID at the pixel, or `undefined` if the position lies
   * outside the label mask
   * @throws Error if the label mask has no band
   */
  async loadLabelId(
    x: number,
    y: number,
    options?: { signal?: AbortSignal },
  ): Promise<number | undefined> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const column = Math.floor(x);
    const row = Math.floor(y);
    if (
      column < 0 ||
      row < 0 ||
      column >= this._image.getWidth() ||
      row >= this._image.getHeight()
    ) {
      return undefined;
    }
    const { pool } = installTIFFTileSource();
    const rasters = await this._image.readRasters({
      window: [column, row, column + 1, row + 1],
      pool,
      signal,
    });
    signal?.throwIfAborted(); // the decoder pool does not abort decoding
    const band = rasters[0];
    if (band === undefined) {
      throw new Error("The label mask has no bands");
    }
    return band[0];
  }

  /** Closing does nothing: the decoder pool is shared, and the file is read on demand */
  close(): void {
    // nothing to release
  }
}
