import type { GeoTIFFImage, Pool } from "geotiff";

import {
  type DataProviderLoadOptions,
  type ImageChannelHistogram,
  type ImageDataProvider,
  ImageUtils,
  MathUtils,
  SourceUtils,
} from "@tissuumaps/core";

import { type TIFFChannel, TIFFImageData } from "./TIFFImageData";
import {
  type NormalizedTIFFImageDataSource,
  type TIFFImageDataSource,
  tiffImageDataSourceDefaults,
} from "./TIFFImageDataSource";
import { TIFFUtils } from "./TIFFUtils";
import { installTIFFTileSource } from "./installTIFFTileSource";
import { openTIFF } from "./openTIFF";

/**
 * Data provider for images stored in TIFF files
 *
 * Opens a {@link TIFFImageDataSource} as {@link TIFFImageData}, with one tile
 * source per channel for multi-channel files and a single one for files drawn
 * in their own colors. The format is recognized from the file's metadata (see
 * `findTIFFParser`), which provides the channel names and colors; the
 * histograms the renderer stretches the channels over are read from the
 * pixels.
 */
export class TIFFImageDataProvider implements ImageDataProvider<
  TIFFImageDataSource,
  TIFFImageData,
  NormalizedTIFFImageDataSource
> {
  /**
   * The pixels a histogram is built from (see
   * {@link TIFFImageDataProvider._computeChannelHistogram}); more do not make
   * the quantiles the renderer derives from them more stable. A level needs at
   * least this many pixels to be read from, and this many of its values are
   * sampled
   */
  private static readonly _numHistogramPixels = 512 * 512;

  readonly name = "TIFF";

  readonly schema = {
    type: "object",
    properties: {
      source: {
        type: "string",
      },
      z: {
        type: "integer",
        minimum: 0,
      },
      t: {
        type: "integer",
        minimum: 0,
      },
    },
    required: ["source"],
  };

  readonly uischema = {
    type: "VerticalLayout",
    elements: [
      {
        type: "Control",
        scope: "#/properties/source",
        label: "Source",
      },
      {
        type: "HorizontalLayout",
        elements: [
          {
            type: "Control",
            scope: "#/properties/z",
            label: "Z-slice",
          },
          {
            type: "Control",
            scope: "#/properties/t",
            label: "Timepoint",
          },
        ],
      },
    ],
  };

  /**
   * Returns the data source with {@link tiffImageDataSourceDefaults} applied
   * and its source normalized (see `SourceUtils.normalizeSource`)
   *
   * @param dataSource - The data source to normalize
   * @param workspace - The directory handle of the open workspace, if any
   * @param projectSource - Where the project was loaded from, if anywhere
   * @returns The normalized data source
   */
  normalize(
    dataSource: TIFFImageDataSource,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
  ): NormalizedTIFFImageDataSource {
    return {
      ...tiffImageDataSourceDefaults,
      ...dataSource,
      source: SourceUtils.normalizeSource(
        dataSource.source,
        workspace,
        projectSource,
      ),
    };
  }

  /**
   * Opens a TIFF image data source and returns the loaded image data
   *
   * The file and its structure are read with {@link openTIFF}. The channel
   * histograms are read afterwards, a few channels at a time (see
   * {@link TIFFImageDataProvider._computeChannelHistograms}).
   *
   * @param normalizedDataSource - The normalized data source to open
   * @param options - See `DataProviderLoadOptions`; `workspace` is required
   * for workspace-relative sources
   * @returns A promise that resolves to the loaded image data
   * @throws Error if the source is workspace-relative while no workspace is
   * open, or if the file holds a TIFF no parser recognizes
   */
  async load(
    normalizedDataSource: NormalizedTIFFImageDataSource,
    options?: DataProviderLoadOptions,
  ): Promise<TIFFImageData> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();

    const { z, t } = normalizedDataSource;
    const { tiff, pyramids, channels } = await openTIFF(
      normalizedDataSource.source,
      { ...options, z, t },
    );

    const { GeoTIFFTileSource, pool, poolSize } = installTIFFTileSource();
    let channelsWithHistograms: TIFFChannel[] | undefined;
    if (channels !== undefined) {
      // the histograms only seed the contrast limits, which the renderer can
      // fall back to the data type range for, so a file whose pixels cannot be
      // decoded still opens
      let histograms: (ImageChannelHistogram | undefined)[] = [];
      try {
        histograms = await TIFFImageDataProvider._computeChannelHistograms(
          pyramids,
          { pool, concurrency: poolSize, signal },
        );
      } catch (error) {
        signal?.throwIfAborted();
        console.error("Failed to read the TIFF channel histograms:", error);
      }
      channelsWithHistograms = channels.map((channel, c) => {
        const sampleType = TIFFUtils.getIntegerSampleType(pyramids[c]![0]!);
        const dataTypeRange =
          sampleType !== undefined
            ? ImageUtils.getIntegerTypeRange(sampleType.bits, sampleType.signed)
            : undefined;
        const isUint8 = sampleType?.bits === 8 && !sampleType.signed;
        return {
          ...channel,
          histogram: histograms[c],
          dataTypeRange,
          contrastLimits: isUint8 ? dataTypeRange : undefined,
        };
      });
    }
    const tileSources = pyramids.map(
      (images) =>
        new GeoTIFFTileSource({ GeoTIFF: tiff, GeoTIFFImages: images }),
    );
    return new TIFFImageData(tileSources, channelsWithHistograms);
  }

  /**
   * Reads the value histogram of every channel of a file
   *
   * At most `concurrency` channels are read at a time, so that a file with
   * many channels does not start every read at once. The decode jobs of a read
   * are spread over the whole pool, so this bounds the reads in flight, not
   * the workers each of them uses.
   *
   * @param pyramids - The images of every channel, largest first
   * @param options - The decoder pool (`null` for the main thread), the number
   * of channels to read at a time (default `1`), and an abort signal
   * @returns One histogram per channel, in channel order, each as returned by
   * {@link TIFFImageDataProvider._computeChannelHistogram}
   * @throws Error if a channel has no pyramid level
   */
  private static async _computeChannelHistograms(
    pyramids: GeoTIFFImage[][],
    options?: {
      pool?: Pool | null;
      concurrency?: number;
      signal?: AbortSignal;
    },
  ): Promise<(ImageChannelHistogram | undefined)[]> {
    const { pool = null, concurrency = 1, signal } = options ?? {};
    signal?.throwIfAborted();
    const histograms: (ImageChannelHistogram | undefined)[] = [];
    let next = 0;
    await Promise.all(
      Array.from(
        { length: Math.min(concurrency, pyramids.length) },
        async () => {
          for (let c = next++; c < pyramids.length; c = next++) {
            histograms[c] =
              await TIFFImageDataProvider._computeChannelHistogram(
                pyramids[c]!,
                { pool, signal },
              );
          }
        },
      ),
    );
    return histograms;
  }

  /**
   * Computes the value histogram of a channel, which the renderer stretches
   * the channel over, since TIFF stores no display range
   *
   * Reads the smallest pyramid level that still holds at least
   * {@link TIFFImageDataProvider._numHistogramPixels} pixels, or the largest
   * level of images smaller than that, and bins the level's values over their
   * actual range (see {@link MathUtils.computeRange} and
   * {@link MathUtils.computeHistogram}), sampled down to that many values, so
   * that the histogram is as fine as its bins allow regardless of the data
   * type's range.
   *
   * @param pyramid - The images of the channel, largest first
   * @param options - The decoder pool (`null` for the main thread) and an
   * abort signal
   * @returns The histogram, as bin counts and the value range they span, or
   * `undefined` for a channel that holds fewer than two distinct values
   * @throws Error if `pyramid` is empty
   */
  private static async _computeChannelHistogram(
    pyramid: GeoTIFFImage[],
    options?: { pool?: Pool | null; signal?: AbortSignal },
  ): Promise<ImageChannelHistogram | undefined> {
    const { pool = null, signal } = options ?? {};
    signal?.throwIfAborted();
    if (pyramid.length === 0) {
      throw new Error("The channel has no pyramid level.");
    }
    const largeEnough = pyramid.filter(
      (image) =>
        image.getWidth() * image.getHeight() >=
        TIFFImageDataProvider._numHistogramPixels,
    );
    const image = largeEnough[largeEnough.length - 1] ?? pyramid[0]!;
    // a single sample, interleaved, is read as one typed array
    const values = await image.readRasters({
      samples: [0],
      interleave: true,
      pool,
      signal,
    });
    const range = await MathUtils.computeRange(values, { signal });
    // a channel without finite values, or with a single one, has no range to
    // spread bins over; the renderer falls back to its data type range
    if (!(range[1] > range[0])) {
      return undefined;
    }
    return await MathUtils.computeHistogram(values, range, {
      signal,
      sample: TIFFImageDataProvider._numHistogramPixels,
    });
  }
}
