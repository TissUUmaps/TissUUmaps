import { deepEqual } from "fast-equals";

import {
  type Color,
  ColorUtils,
  type CustomTileSource,
  type Image,
  ImageChannelViewMode,
  type ImageData,
  ImageUtils,
  MathUtils,
  type TileSourceConfig,
} from "@tissuumaps/core";

import type { DataTransfer } from "./OpenSeadragonContext";
import {
  type ObjectRef,
  OpenSeadragonRendererBase,
} from "./OpenSeadragonRendererBase";

export type OpenSeadragonImageSyncContext = {
  loadObject: (
    image: Image,
    options?: { signal?: AbortSignal },
  ) => Promise<ImageData>;
};

/** The color that channels are colorized with in the grayscale view mode */
const white: Color = { r: 255, g: 255, b: 255 };

/** The color and contrast limits a channel's tiles are drawn with */
type ChannelTransferState = {
  color: Color;
  contrastLimits: [number, number] | undefined;
};

/**
 * Renderer for the tiled images of {@link Image} data objects
 *
 * Multi-channel image data provides one tiled image per channel, whose tiles
 * carry the channel's values rather than colors. Each channel is recolored by a
 * data transfer (see {@link OpenSeadragonContext.updateTiledImageDataTransfer})
 * that scales the values between the channel's contrast limits and multiplies
 * them with the channel's color. Color and contrast limits are resolved by
 * {@link ImageUtils.getChannelColor} and
 * {@link ImageUtils.getChannelContrastLimits}; channels without contrast
 * limits are stretched over the value range of each tile's data type. In the
 * grayscale view mode, every channel is colorized white instead of its color.
 * Channels whose data does not provide values are drawn as they are, as is
 * image data that is not multi-channel. Channel visibility and opacity are not
 * part of the transfer: like layer and image opacity, OpenSeadragon applies
 * them when drawing the tiled image, which is also where the single-channel
 * view modes hide all but the image's active channel (see
 * {@link getTiledImageOpacity}).
 *
 * Channel colors, contrast limits and the view mode need no synchronization:
 * like visibility and opacity, they are read from the current model whenever
 * the image's tiled images are updated (see
 * {@link resolveTiledImageDataTransfer}), so changing them only requires
 * {@link setModel}. As data transfers are compared by identity, each channel's
 * data transfer is kept until the image's data, or the channel's resolved color
 * (white in grayscale) or contrast limits change, so that tiles are only
 * recolored when needed.
 */
export class OpenSeadragonImageRenderer extends OpenSeadragonRendererBase<
  Image,
  ImageData,
  OpenSeadragonImageSyncContext
> {
  private readonly _renderedMultichannelImages = new Map<
    string,
    {
      data: ImageData;
      channels: {
        state: ChannelTransferState;
        dataTransfer: DataTransfer | undefined;
      }[];
    }
  >();

  /**
   * Returns the state of an image that a synchronization depends on
   *
   * Blanks out the channel view mode and active channel, and the name,
   * visibility, opacity, color and contrast limits of every channel on top of
   * what the base class blanks out: they are applied from the current model
   * when updating the channel's tiled image (see {@link getTiledImageOpacity}
   * and {@link resolveTiledImageDataTransfer}), except for the name, which is
   * cosmetic.
   *
   * @param image - The image to return the state of
   * @returns The image without the properties that are applied by
   * {@link setModel}, and without the cosmetic ones
   */
  protected override getObjectSyncState(image: Image): object {
    return {
      ...super.getObjectSyncState(image),
      channelViewMode: undefined,
      activeChannel: undefined,
      channels: image.channels?.map((channel) => ({
        ...channel,
        name: undefined,
        visibility: undefined,
        opacity: undefined,
        color: undefined,
        contrastLimits: undefined,
      })),
    };
  }

  /**
   * Drops the data transfers of all images other than the given ones
   *
   * @param images - The images about to be displayed
   */
  protected override retainObjects(images: Image[]): void {
    for (const imageId of this._renderedMultichannelImages.keys()) {
      if (!images.some((image) => image.id === imageId)) {
        this._renderedMultichannelImages.delete(imageId);
      }
    }
  }

  /**
   * Returns the tile sources for the given image data
   *
   * Multi-channel image data provides one tile source per channel, in channel
   * order; all other image data provides a single tile source.
   *
   * @param data - The image data for which to retrieve the tile sources
   * @returns The tile sources, one per channel for multi-channel image data and
   * a single one otherwise
   */
  protected override getTileSources(
    data: ImageData,
  ): (string | TileSourceConfig | CustomTileSource)[] {
    const tileSources: (string | TileSourceConfig | CustomTileSource)[] = [];
    const n = data.getSizeC();
    if (n !== undefined) {
      for (let c = 0; c < n; c++) {
        tileSources.push(data.getTileSource(c));
      }
    } else {
      tileSources.push(data.getTileSource());
    }
    return tileSources;
  }

  /**
   * Returns whether the channels of the given image data are blended additively
   *
   * Multi-channel image data blends additively, so that its channels add up
   * rather than hide one another. Image data that is not multi-channel has a
   * single tile source, so there is nothing to add up, and it keeps
   * OpenSeadragon's default compositing, which preserves the transparency of its
   * tiles.
   *
   * @param data - The image data to check
   * @returns Whether the image's channels are blended additively
   */
  protected override usesAdditiveBlending(data: ImageData): boolean {
    return data.getSizeC() !== undefined;
  }

  /**
   * Computes the effective opacity for one of an image's tiled images
   *
   * Multiplies the layer and image opacity computed by the base class with the
   * visibility and opacity of the channel that the tiled image renders. Channels
   * are only applied to multi-channel image data, and channels that the image
   * does not define are visible at full opacity. In the single-channel view
   * modes, only the image's active channel is visible (see
   * {@link ImageUtils.getActiveChannel}). Without a channel index, i.e. for the
   * image's backdrop, the layer and image opacity is returned as is.
   *
   * @param ref - The image reference for which to compute the opacity
   * @param index - The index of the tiled image (channel), or `null` for the image's backdrop
   * @returns The effective opacity for the tiled image
   */
  protected override getTiledImageOpacity(
    ref: ObjectRef<Image, ImageData>,
    index: number | null,
  ): number {
    let alpha = super.getTiledImageOpacity(ref, index);
    const sizeC = ref.data.getSizeC();
    if (alpha > 0 && index !== null && sizeC !== undefined) {
      const channelVisibility =
        ref.object.channelViewMode !== ImageChannelViewMode.composite
          ? index === ImageUtils.getActiveChannel(ref.object, sizeC)
          : ImageUtils.getChannelVisibility(ref.object, ref.data, index);
      const channelOpacity = ImageUtils.getChannelOpacity(
        ref.object,
        ref.data,
        index,
      );
      alpha *= channelVisibility ? channelOpacity : 0;
    }
    return alpha;
  }

  /**
   * Resolves the data transfer for one of an image's tiled images, unless it is up to date
   *
   * Only the tiled images of the channels of multi-channel image data are
   * recolored, never the backdrop. A channel's color and contrast limits are
   * resolved (see {@link ImageUtils.getChannelColor} and
   * {@link ImageUtils.getChannelContrastLimits}), with every channel white in
   * the grayscale view mode. The channel's data transfer is kept as long as the
   * image's data and the resolved values are unchanged, and is created anew
   * otherwise (see {@link _createDataTransfer}). Channels are resolved
   * independently, so that changing one channel only recolors the tiles of that
   * channel.
   *
   * @param ref - The image reference for which to get the data transfer, holding
   * the image as it currently is in the model
   * @param index - The index of the tiled image (channel), or `null` for the image's backdrop
   * @returns The channel's data transfer, or `undefined` if the channel has none
   */
  protected override resolveTiledImageDataTransfer(
    ref: ObjectRef<Image, ImageData>,
    index: number | null,
  ): DataTransfer | undefined {
    if (ref.data.getSizeC() === undefined) {
      this._renderedMultichannelImages.delete(ref.object.id);
      return undefined;
    }
    if (index === null) {
      return undefined;
    }
    let renderedMultichannelImage = this._renderedMultichannelImages.get(
      ref.object.id,
    );
    if (
      renderedMultichannelImage === undefined ||
      renderedMultichannelImage.data !== ref.data
    ) {
      renderedMultichannelImage = { data: ref.data, channels: [] };
      this._renderedMultichannelImages.set(
        ref.object.id,
        renderedMultichannelImage,
      );
    }
    const renderedMultichannelImageChannel =
      renderedMultichannelImage.channels[index];
    const newRenderedMultichannelImageChannelState = structuredClone({
      color:
        ref.object.channelViewMode === ImageChannelViewMode.grayscale
          ? white
          : ImageUtils.getChannelColor(ref.object, ref.data, index),
      contrastLimits: ImageUtils.getChannelContrastLimits(
        ref.object,
        ref.data,
        index,
      ),
    });
    if (
      renderedMultichannelImageChannel !== undefined &&
      deepEqual(
        renderedMultichannelImageChannel.state,
        newRenderedMultichannelImageChannelState,
      )
    ) {
      return renderedMultichannelImageChannel.dataTransfer;
    }
    const dataTransfer = OpenSeadragonImageRenderer._createDataTransfer(
      ref.data,
      newRenderedMultichannelImageChannelState,
    );
    renderedMultichannelImage.channels[index] = {
      state: newRenderedMultichannelImageChannelState,
      dataTransfer,
    };
    return dataTransfer;
  }

  /**
   * Creates the data transfer of an image channel
   *
   * The transfer scales each value linearly between the contrast limits,
   * clamped to `[0, 1]`, and multiplies the result with the channel's color,
   * which {@link resolveTiledImageDataTransfer} has already resolved. Without
   * contrast limits, the value range of the data type of each tile's data is
   * used (see {@link ImageUtils.getDataTypeRange}); contrast limits that are not
   * ascending render every value black. The resulting pixels are opaque;
   * channel visibility and opacity are applied by OpenSeadragon when drawing
   * the tiled image.
   *
   * The scaled color is not computed per pixel: the transfer packs a ramp of
   * 256 RGBA colors once, with `ColorUtils.packColor` and
   * `ColorUtils.withAlpha`, and maps each value to the nearest ramp entry. As
   * no color component exceeds 255, the ramp holds every color the channel can
   * take.
   *
   * @param data - The image data providing the channel's values
   * @param channel - The resolved color and contrast limits of the channel
   * @returns The data transfer, or `undefined` if the data provides no channel
   * values
   */
  private static _createDataTransfer(
    data: ImageData,
    channel: ChannelTransferState,
  ): DataTransfer | undefined {
    if (data.getTileData !== undefined) {
      const { r, g, b } = channel.color;
      const ramp = new Uint32Array(256);
      for (let i = 0; i < ramp.length; i++) {
        const scale = i / (ramp.length - 1);
        ramp[i] = ColorUtils.withAlpha(
          ColorUtils.packColor({ r: r * scale, g: g * scale, b: b * scale }),
          1,
          255,
        );
      }
      return {
        getTileData: (event) => data.getTileData!(event),
        transferValues: (values, pixelBuffer) => {
          const [vmin, vmax] =
            channel.contrastLimits ?? ImageUtils.getDataTypeRange(values);
          const rampScale = vmax > vmin ? (ramp.length - 1) / (vmax - vmin) : 0;
          for (let i = 0; i < values.length; i++) {
            const value = values[i]!;
            const rampIndex = Math.round(
              MathUtils.clamp((value - vmin) * rampScale, 0, ramp.length - 1),
            );
            pixelBuffer[i] = ramp[rampIndex]!;
          }
        },
      };
    }
    return undefined;
  }
}
