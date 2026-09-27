import type { Image } from "../model/image";
import type { Color } from "../model/primitives";
import type { ImageChannelHistogram, ImageData } from "../storage/image";
import type { TypedArray } from "../types/arrays";
import { ColorUtils } from "./ColorUtils";
import { MathUtils } from "./MathUtils";

/** The integer typed arrays, with the bits and signedness of their type */
const integerArrayTypes = [
  [Uint8Array, 8, false],
  [Uint8ClampedArray, 8, false],
  [Uint16Array, 16, false],
  [Uint32Array, 32, false],
  [Int8Array, 8, true],
  [Int16Array, 16, true],
  [Int32Array, 32, true],
] as const;

/** Utility methods for resolving and defaulting image channel settings */
export class ImageUtils {
  /**
   * Returns the color of a channel
   *
   * The color configured on the image wins, then the one its data states, and
   * the only channel of single-channel image data is white, as a grayscale
   * image rather than in the red the index-based default would give it (see
   * {@link ImageUtils.getDefaultChannelColor}).
   *
   * @param image - The image
   * @param data - The image data providing the channel
   * @param c - The channel index (0-based)
   * @returns The channel's color
   */
  static getChannelColor(image: Image, data: ImageData, c: number): Color {
    return (
      image.channels?.[c]?.color ??
      data.getChannelColor?.(c) ??
      (data.getSizeC() === 1
        ? { r: 255, g: 255, b: 255 }
        : ImageUtils.getDefaultChannelColor(c))
    );
  }

  /**
   * Returns the contrast limits of a channel
   *
   * The limits configured on the image win over those its data states.
   * Channels with neither are stretched between quantile-based limits derived
   * from their histogram (see {@link ImageUtils.getDefaultContrastLimits}), if
   * the data provides one, and otherwise over the range the data declares for
   * their data type.
   *
   * @param image - The image
   * @param data - The image data providing the channel
   * @param c - The channel index (0-based)
   * @returns The channel's contrast limits, or `undefined` if the data
   * provides neither limits, a histogram nor a data type range
   */
  static getChannelContrastLimits(
    image: Image,
    data: ImageData,
    c: number,
  ): [number, number] | undefined {
    const histogram = data.getChannelHistogram?.(c);
    return (
      image.channels?.[c]?.contrastLimits ??
      data.getChannelContrastLimits?.(c) ??
      (histogram !== undefined
        ? ImageUtils.getDefaultContrastLimits(histogram)
        : undefined) ??
      data.getChannelDataTypeRange?.(c)
    );
  }

  /**
   * Returns the visibility of a channel
   *
   * The visibility configured on the image wins over the one its data states,
   * and channels with neither are visible.
   *
   * @param image - The image
   * @param data - The image data providing the channel
   * @param c - The channel index (0-based)
   * @returns Whether the channel is visible
   */
  static getChannelVisibility(
    image: Image,
    data: ImageData,
    c: number,
  ): boolean {
    return (
      image.channels?.[c]?.visibility ?? data.getChannelVisibility?.(c) ?? true
    );
  }

  /**
   * Returns the opacity of a channel
   *
   * The opacity configured on the image wins over the one its data states,
   * and channels with neither are fully opaque.
   *
   * @param image - The image
   * @param data - The image data providing the channel
   * @param c - The channel index (0-based)
   * @returns The channel's opacity, in `[0, 1]`
   */
  static getChannelOpacity(image: Image, data: ImageData, c: number): number {
    return image.channels?.[c]?.opacity ?? data.getChannelOpacity?.(c) ?? 1;
  }

  /**
   * Returns the channel an image shows in the single-channel view modes
   *
   * The image's active channel is bounded to the channels the data has, so
   * that an image kept from data with more channels still shows one.
   *
   * @param image - The image
   * @param sizeC - The number of channels the image data has
   * @returns The channel index (0-based)
   */
  static getActiveChannel(image: Image, sizeC: number): number {
    return MathUtils.clamp(image.activeChannel, 0, sizeC - 1);
  }

  /**
   * Returns a default color for a channel, for use when no channel colors are
   * known
   *
   * The first six channels map to red, green, blue, yellow, cyan, and magenta.
   * Further channels are assigned hues 128 degrees apart, with saturation and
   * brightness decreasing by 0.05 every ten channels. Channel indices wrap
   * around after 100, so that saturation and brightness stay above 0.5.
   * Colors are not guaranteed to be unique.
   *
   * Resembles `ImageChannel.getDefaultChannelColor` in QuPath v0.7.0, except
   * that yellow, cyan, and magenta have been replaced by pure versions (255
   * for the two highest components, 0 for the lowest), and that QuPath wraps
   * after 360 channels instead, yielding dim colors and eventually black.
   *
   * @param c - The channel index, a non-negative integer
   * @returns The default color for the channel
   */
  static getDefaultChannelColor(c: number): Color {
    c = c % 100;
    switch (c) {
      case 0:
        return { r: 255, g: 0, b: 0 }; // red
      case 1:
        return { r: 0, g: 255, b: 0 }; // green
      case 2:
        return { r: 0, g: 0, b: 255 }; // blue
      case 3:
        return { r: 255, g: 255, b: 0 }; // yellow
      case 4:
        return { r: 0, g: 255, b: 255 }; // cyan
      case 5:
        return { r: 255, g: 0, b: 255 }; // magenta
      default: {
        const hue = ((c * 128) % 360) / 360;
        const level = 1 - Math.floor(c / 10) / 20;
        return ColorUtils.fromHSB(hue, level, level);
      }
    }
  }

  /**
   * Returns quantile-based contrast limits derived from a channel histogram,
   * for use as default contrast limits when no contrast limits are known
   *
   * The histogram's `hist[i]` counts the values that fall into bin `i`, with
   * bins spread evenly over `range`: bin `0` maps to the range's lower bound,
   * the last bin to its upper bound, and each bin in between to
   * `vmin + i / (n - 1) * (vmax - vmin)` (see
   * {@link MathUtils.computeHistogram}). The lower limit is the value of the
   * first bin at which the cumulative count from the bottom reaches the
   * fraction `qlow` of the total count, the upper limit is the value of the
   * first bin at which the cumulative count from the top reaches the fraction
   * `1 - qhigh`; i.e., the fraction `qlow` of the values is clipped at the
   * bottom and the fraction `1 - qhigh` at the top. Only non-empty bins are
   * considered, so quantiles of `0` and `1` yield the first and last non-empty
   * bin, respectively. Values that fall into the same bin cannot be told
   * apart, so the limits are only as precise as the bins.
   *
   * Histograms with fewer than two bins, histograms whose counts sum to zero
   * and degenerate ranges return `range` as is. So do histograms whose clipped
   * limits would not be ascending (upper bin at or below lower bin), which
   * happens if more than `qhigh - qlow` of the values fall into a single bin,
   * e.g. for sparse channels that are mostly background, or if `qhigh` is
   * below `qlow`, and a `qlow` above `1`, which no bin can reach.
   *
   * @param histogram - The channel histogram, as bin counts and value range
   * @param qlow - The quantile of the lower limit, in `[0, 1]`
   * @param qhigh - The quantile of the upper limit, in `[0, 1]`, above `qlow`
   * @returns The contrast limits, as `[low, high]` in the histogram's value
   * range, with `low < high` unless the range is degenerate
   */
  static getDefaultContrastLimits(
    histogram: ImageChannelHistogram,
    qlow: number = 0.01,
    qhigh: number = 0.999,
  ): [number, number] {
    const {
      hist,
      range: [vmin, vmax],
    } = histogram;
    const n = hist.length;
    if (n < 2 || vmin === vmax) {
      return [vmin, vmax];
    }
    const total = hist.reduce((sum, count) => sum + count, 0);
    if (total === 0) {
      return [vmin, vmax];
    }
    let binLeft: number | undefined;
    let binRight: number | undefined;
    let cumulativeLeft = 0;
    let cumulativeRight = 0;
    const targetLeft = total * qlow;
    const targetRight = total - total * qhigh; // avoid potential rounding errors of total * (1 - qhigh)
    for (let i = 0; i < n; i++) {
      const countLeft = hist[i]!;
      if (countLeft > 0) {
        cumulativeLeft += countLeft;
        if (binLeft === undefined && cumulativeLeft >= targetLeft) {
          binLeft = i;
        }
      }
      const countRight = hist[n - 1 - i]!;
      if (countRight > 0) {
        cumulativeRight += countRight;
        if (binRight === undefined && cumulativeRight >= targetRight) {
          binRight = n - 1 - i;
        }
      }
      if (binLeft !== undefined && binRight !== undefined) {
        break;
      }
    }
    if (
      binLeft === undefined ||
      binRight === undefined ||
      binLeft >= binRight
    ) {
      return [vmin, vmax];
    }
    return [
      vmin + (binLeft / (n - 1)) * (vmax - vmin),
      vmin + (binRight / (n - 1)) * (vmax - vmin),
    ];
  }

  /**
   * Returns the value range that an integer type can hold
   *
   * @param bits - The number of bits per value, at most 32
   * @param signed - Whether the type is signed
   * @returns The value range, as `[min, max]`
   */
  static getIntegerTypeRange(bits: number, signed: boolean): [number, number] {
    return signed
      ? [-(2 ** (bits - 1)), 2 ** (bits - 1) - 1]
      : [0, 2 ** bits - 1];
  }

  /**
   * Returns the value range that the type of the given array can hold, for use
   * as default contrast limits of image channel data
   *
   * Integer typed arrays span their full integer range, floating-point typed
   * arrays are taken to hold normalized values in `[0, 1]`.
   *
   * @param values - The array whose value range to return
   * @returns The value range, as `[min, max]`
   */
  static getDataTypeRange(values: TypedArray): [number, number] {
    for (const [arrayType, bits, signed] of integerArrayTypes) {
      if (values instanceof arrayType) {
        return ImageUtils.getIntegerTypeRange(bits, signed);
      }
    }
    return [0, 1];
  }
}
