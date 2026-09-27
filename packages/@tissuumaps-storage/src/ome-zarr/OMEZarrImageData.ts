import type { OMEZarrTileData, OMEZarrTileSource } from "omezarr-tilesource";
import type OpenSeadragon from "openseadragon";

import {
  type Color,
  type CustomTileSource,
  type ImageChannelHistogram,
  type ImageData,
  ImageUtils,
  type TileSourceConfig,
  type TypedArray,
} from "@tissuumaps/core";

/**
 * Loaded OME-Zarr image data
 *
 * Images with a channel axis are multi-channel: they provide one tile source
 * per channel, addressed by channel index, along with the per-channel rendering
 * metadata (name, visibility, color, contrast limits) that the tile sources
 * resolve from the image's `omero` metadata. Images without a channel axis
 * provide a single tile source that is not addressed by channel, and no
 * per-channel metadata: the channel accessors reject them.
 *
 * Extracts one sample per pixel from the raw OME-Zarr data of its tiles,
 * which the renderer contrast-stretches and colorizes. Integer tiles of up to
 * 32 bits and floating-point tiles are passed through as they are; 64-bit
 * integer tiles are rejected.
 *
 * Multi-channel image data carries one precomputed value histogram per
 * channel, computed at load time from a downsampled resolution level (see
 * `OMEZarrImageDataProvider.load`), from which the renderer derives default
 * contrast limits for channels whose `omero` window is incomplete, except for
 * `uint8` channels, which fall back to the full `[0, 255]` range instead (see
 * {@link OMEZarrImageData.getChannelContrastLimits}).
 */
export class OMEZarrImageData implements ImageData {
  private readonly _tileSources: OMEZarrTileSource | OMEZarrTileSource[];
  private readonly _histograms:
    (ImageChannelHistogram | undefined)[] | undefined;

  /**
   * @param tileSources - One ready tile source per channel, in channel order,
   * each rendering only its channel (`c`), for images with a channel axis; the
   * single ready tile source for images without one
   * @param histograms - One precomputed value histogram per channel, in
   * channel order (`undefined` for channels without one), for images with a
   * channel axis; `undefined` for images without one
   */
  constructor(
    tileSources: OMEZarrTileSource | OMEZarrTileSource[],
    histograms?: (ImageChannelHistogram | undefined)[],
  ) {
    this._tileSources = tileSources;
    this._histograms = histograms;
  }

  /**
   * Returns the number of channels, or `undefined` for images without a
   * channel axis
   */
  getSizeC(): number | undefined {
    return Array.isArray(this._tileSources)
      ? this._tileSources.length
      : undefined;
  }

  /**
   * Returns the tile source of a channel, or the only tile source of an image
   * without a channel axis
   *
   * @param c - The channel index (0-based), required for images with a channel
   * axis and to be omitted otherwise
   * @returns The tile source
   * @throws Error if `c` is omitted for an image with a channel axis, passed
   * for an image without one, or out of bounds
   */
  getTileSource(c?: number): string | TileSourceConfig | CustomTileSource {
    if (Array.isArray(this._tileSources)) {
      if (c === undefined) {
        throw new Error("Not a single-channel image");
      }
      return this._getChannelTileSource(c);
    }
    if (c !== undefined) {
      throw new Error("Not a multi-channel image");
    }
    return this._tileSources;
  }

  /**
   * Extracts the samples of an invalidated tile from its raw OME-Zarr data
   *
   * The tile has to belong to an `OMEZarrTileSource` rendering a single
   * channel, whose tile data (`chunks`) holds exactly one two-dimensional
   * (height x width) tile read from the Zarr array (with the channel, z-slice
   * and timepoint already selected) instead of a rendered image. 64-bit
   * integer tiles are rejected (see {@link OMEZarrImageData}).
   *
   * @param event - The tile invalidation event
   * @returns The samples of the invalidated tile, one per raster pixel in
   * row-major order, along with the width and height of the raster in pixels
   * @throws Error if the tile data does not hold exactly one tile, or if the
   * tile holds 64-bit integers
   */
  async getTileData(
    event: OpenSeadragon.TileInvalidatedEvent,
  ): Promise<{ values: TypedArray; width: number; height: number }> {
    const data = (await event.getData("ome-zarr")) as OMEZarrTileData;
    if (data.chunks.length !== 1) {
      throw new Error(`Expected a single tile, got ${data.chunks.length}`);
    }
    const tile = data.chunks[0]!; // single channel -> first tile only
    if (
      tile.data instanceof BigInt64Array ||
      tile.data instanceof BigUint64Array
    ) {
      throw new Error("64-bit integer data is not supported");
    }
    return {
      values: tile.data,
      width: tile.shape[1]!,
      height: tile.shape[0]!,
    };
  }

  /**
   * Returns the label of a channel from the image's `omero` metadata
   *
   * @param c - The channel index (0-based)
   * @returns The channel's label, or `undefined` if it has none
   * @throws Error if the image has no channel axis, or if `c` is out of bounds
   */
  getChannelName(c: number): string | undefined {
    const tileSource = this._getChannelTileSource(c);
    const label = tileSource.channels?.[0]?.label;
    return typeof label === "string" ? label : undefined;
  }

  /**
   * Returns whether a channel is active in the image's `omero` metadata
   *
   * @param c - The channel index (0-based)
   * @returns The channel's active flag, or `undefined` if it has none
   * @throws Error if the image has no channel axis, or if `c` is out of bounds
   */
  getChannelVisibility(c: number): boolean | undefined {
    const tileSource = this._getChannelTileSource(c);
    return tileSource.channels?.[0]?.active;
  }

  /**
   * Returns the color of a channel from the image's `omero` metadata
   *
   * Channel colors are 6-digit hex strings with an optional `#`, which the
   * tile source validates and parses. Channels without a valid color have no
   * color, leaving the renderer to derive one from the channel index, even if
   * other channels of the image have one.
   *
   * @param c - The channel index (0-based)
   * @returns The channel's color, or `undefined` if it has no valid one
   * @throws Error if the image has no channel axis, or if `c` is out of bounds
   */
  getChannelColor(c: number): Color | undefined {
    const tileSource = this._getChannelTileSource(c);
    const color = tileSource.colors?.[0];
    if (color !== undefined) {
      const [r, g, b] = color;
      return { r, g, b };
    }
    return undefined;
  }

  /**
   * Returns the precomputed value histogram of a channel
   *
   * The histogram is computed once at load time from a downsampled resolution
   * level of the channel's plane (see `OMEZarrImageDataProvider.load`) and
   * spans the channel's actual value range at that level, so that the default
   * contrast limits the renderer derives from it are as precise as its bins.
   *
   * @param c - The channel index (0-based)
   * @returns The channel's histogram, as bin counts and the value range the
   * bins span, or `undefined` if none was computed (64-bit integer data, or
   * planes with fewer than two distinct finite values)
   * @throws Error if the image has no channel axis, or if `c` is out of bounds
   */
  getChannelHistogram(c: number): ImageChannelHistogram | undefined {
    this._getChannelTileSource(c); // check channel index
    return this._histograms?.[c];
  }

  /**
   * Returns the range the `dtype` of a channel can hold
   *
   * @param c - The channel index (0-based)
   * @returns The full range of an integer `dtype` of at most 32 bits,
   * `undefined` for every other `dtype`
   * @throws Error if the image has no channel axis, or if `c` is out of bounds
   */
  getChannelDataTypeRange(c: number): [number, number] | undefined {
    const { dtype } = this._getChannelTileSource(c).loaded.arrays[0]!;
    const match = /^(u?)int(8|16|32)$/.exec(dtype);
    if (match === null) {
      return undefined;
    }
    const [, unsignedPrefix, bits] = match;
    const signed = unsignedPrefix === "";
    return ImageUtils.getIntegerTypeRange(Number(bits), signed);
  }

  /**
   * Returns the rendering window of a channel from the image's `omero` metadata
   *
   * `uint8` channels without a complete window fall back to the full
   * `[0, 255]` range, like other viewers show them, rather than to the
   * quantile-based limits the renderer would otherwise derive from their
   * histogram (see {@link OMEZarrImageData.getChannelHistogram}).
   *
   * @param c - The channel index (0-based)
   * @returns The channel's window `[start, end]` widened to a non-empty range
   * if necessary; `[0, 255]` for `uint8` channels without a start or an end;
   * or `undefined` for other channels without a start or an end
   * @throws Error if the image has no channel axis, or if `c` is out of bounds
   */
  getChannelContrastLimits(c: number): [number, number] | undefined {
    const tileSource = this._getChannelTileSource(c);
    const range = tileSource.ranges?.[0];
    if (range !== undefined) {
      const [start, end] = range;
      return start < end ? [start, end] : [start, start + 1];
    }
    if (tileSource.loaded.arrays[0]!.dtype === "uint8") {
      return [0, 255];
    }
    return undefined;
  }

  close(): void {}

  /**
   * Returns the tile source rendering a channel
   *
   * @param c - The channel index (0-based)
   * @returns The tile source
   * @throws Error if the image has no channel axis, or if `c` is out of bounds
   */
  private _getChannelTileSource(c: number): OMEZarrTileSource {
    if (!Array.isArray(this._tileSources)) {
      throw new Error("Not a multi-channel image");
    }
    const tileSource = this._tileSources[c];
    if (tileSource === undefined) {
      throw new Error(`Channel index ${c} is out of bounds`);
    }
    return tileSource;
  }
}
