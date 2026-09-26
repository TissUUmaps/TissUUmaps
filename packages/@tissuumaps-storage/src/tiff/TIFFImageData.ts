import type { TiffRaster } from "geotiff-tilesource";
import type OpenSeadragon from "openseadragon";

import type {
  Color,
  CustomTileSource,
  ImageChannelHistogram,
  ImageData,
  TileSourceConfig,
  TypedArray,
} from "@tissuumaps/core";

import { tiffRasterType } from "./installTIFFTileSource";

/** What a parser reads about a channel of a multi-channel TIFF file */
export type TIFFChannelMetadata = {
  name?: string;
  color?: Color;
};

/** A channel of a loaded multi-channel TIFF file */
export type TIFFChannel = TIFFChannelMetadata & {
  histogram?: ImageChannelHistogram;
  dataTypeRange?: [number, number];
  contrastLimits?: [number, number];
};

/**
 * The loaded image of a TIFF file
 *
 * Multi-channel files provide one tile source per channel, whose tiles carry
 * the raw samples that the renderer contrast-stretches and colorizes, using the
 * channel names and colors read from the file and the value histogram read from
 * its pixels. Files that are drawn in their own colors (RGB, palette or
 * white-is-zero) provide a single tile source instead.
 *
 * Channel visibility is not provided: neither OME-XML nor the QPTIFF
 * description records it.
 */
export class TIFFImageData implements ImageData {
  private readonly _tileSources: OpenSeadragon.TileSource[];
  private readonly _channels: TIFFChannel[] | undefined;

  /**
   * @param tileSources - One per channel, or a single one for RGB images
   * @param channels - The channels, or `undefined` for RGB images
   */
  constructor(
    tileSources: OpenSeadragon.TileSource[],
    channels: TIFFChannel[] | undefined,
  ) {
    this._tileSources = tileSources;
    this._channels = channels;
  }

  /** Returns the number of channels, or `undefined` for a file drawn in its own colors */
  getSizeC(): number | undefined {
    return this._channels?.length;
  }

  /**
   * Returns the tile source of a channel, or the only tile source of a file
   * drawn in its own colors
   *
   * @param c - The channel index (0-based), required for multi-channel files
   * and to be omitted otherwise
   * @returns The tile source
   * @throws Error if `c` is given for a file drawn in its own colors, omitted
   * for a multi-channel file, or out of bounds
   */
  getTileSource(c?: number): string | TileSourceConfig | CustomTileSource {
    if (this._channels === undefined) {
      if (c !== undefined) {
        throw new Error("Not a multi-channel image");
      }
      return this._tileSources[0]!;
    }
    if (c === undefined) {
      throw new Error("Not a single-channel image");
    }
    if (c < 0 || c >= this._tileSources.length) {
      throw new Error(`Channel index ${c} is out of bounds`);
    }
    return this._tileSources[c]!;
  }

  /**
   * Extracts the samples of a channel tile from a tile invalidation event
   *
   * The samples are returned as they are, in the raster's own size (see
   * {@link TiffRaster}), not cropped to the tile's bounds. They are the cached
   * raster's own band rather than a copy (see `copyRasters` in
   * `installTIFFTileSource`), so callers must not modify them.
   *
   * @param event - The tile invalidation event
   * @returns The samples of the invalidated tile, one per raster pixel in
   * row-major order, along with the width and height of the raster
   * @throws Error if the file is drawn in its own colors, or if the tile's
   * raster has no band
   */
  async getTileData(
    event: OpenSeadragon.TileInvalidatedEvent,
  ): Promise<{ values: TypedArray; width: number; height: number }> {
    if (this._channels === undefined) {
      throw new Error("Not a multi-channel image");
    }
    const raster = (await event.getData(tiffRasterType)) as TiffRaster;
    const band = raster.bands[0];
    if (band === undefined) {
      throw new Error("The tile's raster has no bands");
    }
    return { values: band, width: raster.width, height: raster.height };
  }

  /**
   * Returns the name of a channel, as the OME-XML or the QPTIFF description
   * records it
   *
   * @param c - The channel index (0-based)
   * @returns The channel's name, or `undefined` if the file records none
   * @throws Error if `c` is out of bounds
   */
  getChannelName(c: number): string | undefined {
    return this._getChannel(c).name;
  }

  /**
   * Returns the color of a channel, as the OME-XML or the QPTIFF description
   * records it
   *
   * Channels the file leaves uncolored are left to the renderer, which gives
   * them a default color.
   *
   * @param c - The channel index (0-based)
   * @returns The channel's color, or `undefined` if the file records none
   * @throws Error if `c` is out of bounds
   */
  getChannelColor(c: number): Color | undefined {
    return this._getChannel(c).color;
  }

  /**
   * Returns the value histogram read from the pixels of a channel, which the
   * renderer derives its contrast limits from
   *
   * @param c - The channel index (0-based)
   * @returns The channel's histogram, or `undefined` for a channel that holds
   * fewer than two distinct values
   * @throws Error if `c` is out of bounds
   */
  getChannelHistogram(c: number): ImageChannelHistogram | undefined {
    return this._getChannel(c).histogram;
  }

  /**
   * Returns the range the sample type of a channel can hold
   *
   * @param c - The channel index (0-based)
   * @returns The full range of an integer channel of at most 32 bits,
   * `undefined` for every other channel
   * @throws Error if `c` is out of bounds
   */
  getChannelDataTypeRange(c: number): [number, number] | undefined {
    return this._getChannel(c).dataTypeRange;
  }

  /**
   * Returns the contrast limits an 8-bit channel is shown over
   *
   * TIFF records no display range. 8-bit channels are shown over their full
   * range, like other viewers show them, rather than over the quantile-based
   * limits the renderer would otherwise derive from their histogram (see
   * {@link TIFFImageData.getChannelHistogram}).
   *
   * @param c - The channel index (0-based)
   * @returns `[0, 255]` for an 8-bit channel, `undefined` for every other
   * channel
   * @throws Error if `c` is out of bounds
   */
  getChannelContrastLimits(c: number): [number, number] | undefined {
    return this._getChannel(c).contrastLimits;
  }

  /** Closing does nothing: the decoder pool is shared, and the file is read on demand */
  close(): void {
    // nothing to release
  }

  /**
   * Returns the channel at an index
   *
   * @param c - The channel index (0-based)
   * @returns The channel
   * @throws Error if the file is drawn in its own colors, or if `c` is out of
   * bounds
   */
  private _getChannel(c: number): TIFFChannel {
    if (this._channels === undefined) {
      throw new Error("Not a multi-channel image");
    }
    const channel = this._channels[c];
    if (channel === undefined) {
      throw new Error(`Channel index ${c} is out of bounds`);
    }
    return channel;
  }
}
