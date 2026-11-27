import {
  type DataSource,
  type RawDataSource,
  type RawRenderedRasterDataObject,
  type RenderedRasterDataObject,
  createDataSource,
  createRenderedRasterDataObject,
  omitUndefined,
} from "./base";
import type { Color } from "./primitives";

/**
 * How the channels of multi-channel image data are displayed
 *
 * - `composite`: all visible channels, blended in their colors
 * - `grayscale`: the active channel alone, in gray
 * - `color`: the active channel alone, in its color
 */
export const ImageChannelViewMode = {
  composite: "composite",
  grayscale: "grayscale",
  color: "color",
} as const;

/** One of the channel view modes of the {@link ImageChannelViewMode} object */
export type ImageChannelViewMode =
  (typeof ImageChannelViewMode)[keyof typeof ImageChannelViewMode];

/**
 * Default values for {@link RawImage}
 */
export const imageDefaults = {
  channelViewMode: ImageChannelViewMode.composite,
  activeChannel: 0,
} as const satisfies Partial<RawImage>;

/**
 * A channel of a two-dimensional raster image
 *
 * Channels are only applied to multi-channel image data, as reported by the
 * image's data provider.
 */
export type ImageChannel = {
  /**
   * Channel name
   *
   * Overrides the channel name provided by the image data provider, if any.
   *
   * @defaultValue The name reported by the image data provider, or `undefined`
   */
  name?: string;

  /**
   * Channel visibility
   *
   * Overrides the channel visibility provided by the image data provider, if any.
   *
   * @defaultValue The visibility reported by the image data provider, or `true`
   */
  visibility?: boolean;

  /**
   * Channel opacity, in the range [0, 1]
   *
   * Overrides the channel opacity provided by the image data provider, if any.
   *
   * @defaultValue The opacity reported by the image data provider, or `1`
   */
  opacity?: number;

  /**
   * Channel color
   *
   * Overrides the channel color provided by the image data provider, if any.
   *
   * @defaultValue The color reported by the image data provider, or white for
   * single-channel image data, or a default color for the channel index
   * otherwise (see {@link ImageUtils.getDefaultChannelColor})
   */
  color?: Color;

  /**
   * Channel contrast limits, in the channel's image data value range
   *
   * Overrides the channel contrast limits provided by the image data provider, if any.
   *
   * @defaultValue The contrast limits reported by the image data provider, or
   * limits derived from the channel histogram reported by the image data
   * provider (see {@link ImageUtils.getDefaultContrastLimits}), or the value
   * range of the channel's data type
   */
  contrastLimits?: [number, number];
};

/**
 * A two-dimensional raster image
 */
export interface RawImage extends RawRenderedRasterDataObject<
  RawImageDataSource<string>
> {
  /**
   * The channels of the image, indexed by channel
   *
   * Channels beyond the end of the array, and channels of images whose data is
   * not multi-channel, use the default values of {@link ImageChannel}.
   */
  channels?: ImageChannel[];

  /**
   * How the channels are displayed
   *
   * In the single-channel modes, only {@link RawImage.activeChannel} is shown,
   * whatever the visibility of the channels; the channel colors are kept, and
   * ignored in grayscale.
   *
   * @defaultValue {@link imageDefaults.channelViewMode}
   */
  channelViewMode?: ImageChannelViewMode;

  /**
   * The channel shown in the single-channel view modes (0-based)
   *
   * @defaultValue {@link imageDefaults.activeChannel}
   */
  activeChannel?: number;
}

/**
 * A {@link RawImage} with {@link imageDefaults} applied
 */
export type Image = RenderedRasterDataObject<ImageDataSource<string>> &
  Required<Pick<RawImage, keyof typeof imageDefaults>> &
  Omit<RawImage, keyof typeof imageDefaults>;

/**
 * Creates an {@link Image} from a {@link RawImage} by applying {@link imageDefaults}
 *
 * @param rawImage - The raw image
 * @returns The complete image with the default values of missing or `undefined` properties
 */
export function createImage(rawImage: RawImage): Image {
  return {
    ...createRenderedRasterDataObject(rawImage),
    ...structuredClone(imageDefaults),
    ...structuredClone(omitUndefined(rawImage)),
    dataSource: createImageDataSource(rawImage.dataSource),
  };
}

/**
 * Default values for {@link RawImageDataSource}
 */
export const imageDataSourceDefaults = {} as const satisfies Partial<
  RawImageDataSource<string>
>;

/**
 * A data source for two-dimensional raster images
 */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface RawImageDataSource<
  TType extends string = string,
> extends RawDataSource<TType> {}

/**
 * A {@link RawImageDataSource} with {@link imageDataSourceDefaults} applied
 */
export type ImageDataSource<TType extends string = string> = DataSource<TType> &
  Required<
    Pick<RawImageDataSource<TType>, keyof typeof imageDataSourceDefaults>
  > &
  Omit<RawImageDataSource<TType>, keyof typeof imageDataSourceDefaults>;

/**
 * Creates an {@link ImageDataSource} from a {@link RawImageDataSource} by applying {@link imageDataSourceDefaults}
 *
 * @param rawImageDataSource - The raw image data source
 * @returns The complete image data source with the default values of missing or `undefined` properties
 */
export function createImageDataSource<TType extends string>(
  rawImageDataSource: RawImageDataSource<TType>,
): ImageDataSource<TType> {
  return {
    ...createDataSource(rawImageDataSource),
    ...structuredClone(imageDataSourceDefaults),
    ...structuredClone(omitUndefined(rawImageDataSource)),
  };
}
