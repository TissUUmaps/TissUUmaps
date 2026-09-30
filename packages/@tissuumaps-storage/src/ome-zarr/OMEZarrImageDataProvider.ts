import { OMEZarrTileSource } from "omezarr-tilesource";

import {
  type DataProviderLoadOptions,
  type ImageChannelHistogram,
  type ImageDataProvider,
  MathUtils,
  SourceUtils,
} from "@tissuumaps/core";

import { OMEZarrImageData } from "./OMEZarrImageData";
import {
  type NormalizedOMEZarrImageDataSource,
  type OMEZarrImageDataSource,
  omeZarrImageDataSourceDefaults,
} from "./OMEZarrImageDataSource";
import { openOMEZarr } from "./openOMEZarr";

/**
 * Data provider for OME-Zarr images
 *
 * Opens an {@link OMEZarrImageDataSource} as {@link OMEZarrImageData} with one
 * tile source and one precomputed value histogram per channel if the image has
 * a channel axis (even one of length one), and with a single tile source
 * otherwise.
 */
export class OMEZarrImageDataProvider implements ImageDataProvider<
  OMEZarrImageDataSource,
  OMEZarrImageData,
  NormalizedOMEZarrImageDataSource
> {
  /**
   * The minimum number of pixels of the resolution level that channel
   * histograms are computed from (see
   * {@link OMEZarrImageDataProvider._computeChannelHistogram}); more do not
   * make the quantiles the renderer derives from them more stable
   */
  private static readonly _numHistogramPixels = 512 * 512;

  readonly name = "OME-Zarr";

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
   * Returns the data source with {@link omeZarrImageDataSourceDefaults} applied
   * and its source normalized (see `SourceUtils.normalizeSource`)
   *
   * @param dataSource - The data source to normalize
   * @param workspace - The directory handle of the open workspace, if any
   * @param projectSource - Where the project was loaded from, if anywhere
   * @returns The normalized data source
   */
  normalize(
    dataSource: OMEZarrImageDataSource,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
  ): NormalizedOMEZarrImageDataSource {
    return {
      ...omeZarrImageDataSourceDefaults,
      ...dataSource,
      source: SourceUtils.normalizeSource(
        dataSource.source,
        workspace,
        projectSource,
      ),
    };
  }

  /**
   * Opens an OME-Zarr image data source and returns the loaded image data
   *
   * The OME-Zarr image and its arrays are loaded once with {@link openOMEZarr}
   * and shared between the tile sources opened for it: one per channel for
   * images with a channel axis, or a single one for images without one. Every
   * tile source renders exactly one channel (`c`), so that its tile data holds
   * a single tile (see {@link OMEZarrImageData.getTileData}) and its resolved
   * `omero` metadata is that channel's. The `z` and `t` of the data source
   * select the plane to open.
   *
   * For images with a channel axis, the channels are opened concurrently, each
   * computing the value histogram of its plane from a downsampled resolution
   * level as soon as its tile source has opened (see
   * {@link OMEZarrImageDataProvider._computeChannelHistogram}), so that
   * {@link OMEZarrImageData.getChannelHistogram} can return it without
   * computing anything.
   *
   * @param normalizedDataSource - The normalized data source to open
   * @param options - See `DataProviderLoadOptions`; `workspace` is required
   * for workspace-relative sources
   * @returns A promise that resolves to the loaded image data
   * @throws Error if the source is workspace-relative while no workspace is
   * open
   */
  async load(
    normalizedDataSource: NormalizedOMEZarrImageDataSource,
    options?: DataProviderLoadOptions,
  ): Promise<OMEZarrImageData> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const { loaded, url, zip } = await openOMEZarr(
      normalizedDataSource.source,
      options,
    );
    const { t, z } = normalizedDataSource;
    const cAxis = loaded.image.getAxesNames().indexOf("c");
    if (cAxis >= 0) {
      const sizeC = loaded.arrays[0]!.shape[cAxis]!;
      const channels = await Promise.all(
        Array.from({ length: sizeC }, async (_, c) => {
          const tileSource = await OMEZarrTileSource.open(
            { url, zip, t, z, c },
            loaded,
            { signal },
          );
          let histogram;
          try {
            histogram = await OMEZarrImageDataProvider._computeChannelHistogram(
              tileSource,
              { signal },
            );
          } catch (error) {
            if (signal?.aborted) {
              throw error;
            }
            console.warn(
              `Failed to compute histogram for channel ${c}:`,
              error,
            );
          }
          return { tileSource, histogram };
        }),
      );
      return new OMEZarrImageData(
        channels.map((channel) => channel.tileSource),
        channels.map((channel) => channel.histogram),
      );
    }
    const tileSource = await OMEZarrTileSource.open(
      // c: 0 selects the only (implicit) channel: single-tile tile data, and no
      // "active channels" default (which would reject an inactive sole channel)
      { url, zip, t, z, c: 0 },
      loaded,
      { signal },
    );
    return new OMEZarrImageData(tileSource);
  }

  /**
   * Computes the value histogram of a channel from a downsampled resolution
   * level
   *
   * 64-bit integers get no histogram, and no values are read for them, as
   * {@link OMEZarrImageData.getTileData} rejects their tiles anyway. For all
   * other data types, loads the plane that the given tile source displays (its
   * channel, z-slice and timepoint, see `OMEZarrTileSource.loadChunks`) from
   * the lowest resolution level that still holds at least
   * {@link OMEZarrImageDataProvider._numHistogramPixels} pixels, or from the
   * full-resolution level if no level does (images smaller than that, and
   * images without a multiscale pyramid), and bins the plane's values over
   * their actual range (see {@link MathUtils.computeRange} and
   * {@link MathUtils.computeHistogram}), so that the histogram is as fine as
   * its bins allow regardless of the data type's range. Aborting the signal
   * rejects with its reason.
   *
   * @param tileSource - The opened tile source of the channel
   * @param options - Optional abort signal
   * @returns A promise that resolves to the histogram, or to `undefined` for
   * 64-bit integer planes and planes with fewer than two distinct finite
   * values (which have no range to spread bins over; the renderer falls back
   * to the data type range for them)
   * @throws Error if the tile source does not load exactly one plane, i.e.
   * does not render exactly one channel
   */
  private static async _computeChannelHistogram(
    tileSource: OMEZarrTileSource,
    options?: { signal?: AbortSignal },
  ): Promise<ImageChannelHistogram | undefined> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const { dtype } = tileSource.loaded.arrays[0]!;
    if (dtype === "int64" || dtype === "uint64") {
      return undefined;
    }
    let level = 0; // OpenSeadragon level 0 is the lowest resolution
    while (
      level < tileSource.maxLevel &&
      tileSource.getWidth(level) * tileSource.getHeight(level) <
        OMEZarrImageDataProvider._numHistogramPixels
    ) {
      level++;
    }
    const planes = await tileSource.loadChunks(level, undefined, { signal });
    if (planes.length !== 1) {
      throw new Error(`Expected a single plane, got ${planes.length}`);
    }
    const plane = planes[0]!;
    if (
      plane.data instanceof BigInt64Array ||
      plane.data instanceof BigUint64Array
    ) {
      return undefined; // not reached for the dtypes checked above; narrows the type
    }
    const [vmin, vmax] = await MathUtils.computeRange(plane.data, { signal });
    if (vmin >= vmax) {
      return undefined; // no finite values, or a single one
    }
    return MathUtils.computeHistogram(plane.data, [vmin, vmax], {
      signal,
      sample: OMEZarrImageDataProvider._numHistogramPixels,
    });
  }
}
