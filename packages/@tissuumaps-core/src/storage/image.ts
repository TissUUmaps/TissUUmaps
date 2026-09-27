import type { ImageDataSource } from "../model/image";
import type { Color } from "../model/primitives";
import type {
  CustomTileSource,
  TileSourceConfig,
} from "../types/openseadragon";
import type { DataProvider, RasterData } from "./base";

/** The value histogram of a channel, in raw pixel values */
export type ImageChannelHistogram = {
  /**
   * The number of samples per bin, bin `i` of `n` counting the values around
   * `vmin + i / (n - 1) * (vmax - vmin)` (see
   * {@link MathUtils.computeHistogram})
   */
  hist: number[];

  /** The values the first and the last bin are centered on, as `[vmin, vmax]` */
  range: [number, number];
};

/**
 * Loaded image data providing one or more OpenSeadragon-compatible tile sources
 *
 * Image data is either multi-channel, in which case it provides one tile source
 * per channel, addressed by channel index; or it is not, in which case it
 * provides a single tile source that is not addressed by channel.
 * {@link ImageData.getSizeC} returns `undefined` for image data that is not multi-channel.
 *
 * Multi-channel image data that implements {@link RasterData.getTileData}
 * provides a single sample ("grayscale value") per pixel and channel, which the
 * renderer contrast-stretches and colorizes (see
 * {@link ImageData.getChannelContrastLimits} and
 * {@link ImageData.getChannelColor}). Multi-channel image data that does not
 * implement it is free to provide grayscale or RGB tile sources per channel,
 * which are drawn as they are.
 */
export interface ImageData extends RasterData {
  /** Returns the number of channels in the image, or undefined if not multi-channel */
  getSizeC(): number | undefined;

  /**
   * Returns the tile source of a channel, or the only tile source of image data that is not multi-channel
   *
   * @param c - The channel index (0-based), required for multi-channel image
   * data and to be omitted otherwise
   * @returns The tile source, which can be a URL string, a TileSourceConfig
   * object, or a CustomTileSource object
   * @throws Error if `c` is omitted for multi-channel image data, if `c` is
   * passed for image data that is not multi-channel, or if `c` is out of bounds
   */
  getTileSource(c?: number): string | TileSourceConfig | CustomTileSource;

  /** Returns the name of a specific channel, or undefined if not available */
  getChannelName?: (c: number) => string | undefined;

  /** Returns the visibility of a specific channel, or undefined if not available */
  getChannelVisibility?: (c: number) => boolean | undefined;

  /** Returns the opacity of a specific channel, or undefined if not available */
  getChannelOpacity?: (c: number) => number | undefined;

  /**
   * Returns the color of a specific channel, or undefined if not available
   *
   * Channels without a color (neither configured on the image nor returned
   * here) are colorized with a default color derived from the channel index
   * (see {@link ImageUtils.getDefaultChannelColor}), so that multi-channel image data
   * without any color information renders as distinguishable channels rather
   * than as a single white blob; the only channel of single-channel image data
   * is colorized white instead. Data providers should only return colors that
   * their image metadata actually specifies, and `undefined` for all other
   * channels, leaving the choice of default colors to the renderer.
   *
   * @param c - The channel index (0-based)
   * @returns The channel's color, or `undefined` if not available
   */
  getChannelColor?: (c: number) => Color | undefined;

  /**
   * Returns the value histogram of a specific channel, or undefined if not available
   *
   * `hist[i]` counts the channel values that fall into bin `i`, with bins
   * spread evenly over `range`: the first bin corresponds to the range's lower
   * bound, the last bin to its upper bound. The histogram may be computed from
   * a downsampled resolution level. The renderer derives default contrast
   * limits from the histogram for channels without contrast limits (neither
   * configured on the image nor returned by
   * {@link ImageData.getChannelContrastLimits}, see
   * {@link ImageUtils.getDefaultContrastLimits}). As the renderer calls this on every
   * synchronization, implementations should return a precomputed histogram
   * rather than compute one on each call.
   *
   * @param c - The channel index (0-based)
   * @returns The channel's histogram, as bin counts and the value range the
   * bins span, or `undefined` if not available
   */
  getChannelHistogram?: (c: number) => ImageChannelHistogram | undefined;

  /**
   * Returns the range the data type of a specific channel can hold, or undefined if not available
   *
   * The full range of integer channel data of at most 32 bits, e.g.
   * `[0, 65535]` for unsigned 16-bit samples. Other channel data has no such
   * range and returns `undefined`.
   *
   * @param c - The channel index (0-based)
   * @returns The data type range, as `[min, max]`, or `undefined` if not
   * available
   */
  getChannelDataTypeRange?: (c: number) => [number, number] | undefined;

  /**
   * Returns the contrast limits of a specific channel, or undefined if not available
   *
   * The renderer scales each channel value linearly between the contrast
   * limits, clamps the result to `[0, 1]` and multiplies it with the channel's
   * color. Channels without contrast limits (neither configured on the image
   * nor returned here) are stretched between quantile-based limits derived
   * from the channel's histogram, if {@link ImageData.getChannelHistogram}
   * provides one (see {@link ImageUtils.getDefaultContrastLimits}), then over
   * the data type range declared by {@link ImageData.getChannelDataTypeRange},
   * and otherwise over the value range that the data type of their channel data
   * can hold, as returned by {@link ImageUtils.getDataTypeRange} (the full
   * integer range for integer typed arrays, `[0, 1]` for floating-point typed
   * arrays).
   *
   * @param c - The channel index (0-based)
   * @returns The channel's contrast limits, in the channel's value range, or
   * `undefined` if not available
   */
  getChannelContrastLimits?: (c: number) => [number, number] | undefined;
}

/**
 * Data provider for raster images
 *
 * @typeParam TImageDataSource - The data source type this data provider opens
 * @typeParam TImageData - The {@link ImageData} type produced by this data
 * provider
 * @typeParam TNormalizedImageDataSource - The normalized data source type
 * produced by `normalize` and accepted by `load`
 */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface ImageDataProvider<
  TImageDataSource extends ImageDataSource,
  TImageData extends ImageData,
  TNormalizedImageDataSource extends TImageDataSource = TImageDataSource,
> extends DataProvider<
  TImageDataSource,
  TImageData,
  TNormalizedImageDataSource
> {}
