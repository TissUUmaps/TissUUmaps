import type { ImageDataSource } from "@tissuumaps/core";

/** Type discriminator of OME-Zarr image data sources */
export const omeZarrImageDataSourceType = "ome-zarr";

/** Default values for {@link OMEZarrImageDataSource} */
export const omeZarrImageDataSourceDefaults = {};

/**
 * Data source for OME-Zarr images
 *
 * The `source` refers to a remote OME-Zarr store (a URL), a remote zipped
 * OME-Zarr file (a URL with the zipped OME-Zarr extension, in any case, see
 * `OMEZarrUtils.isZipSource`), or an OME-Zarr store (a directory) or zipped
 * OME-Zarr file in the open workspace (a workspace-relative or
 * project-relative path).
 *
 * Images with a channel axis (even one of length one) are opened as
 * multi-channel image data with one tile source per channel; images without
 * a channel axis as single-channel image data.
 *
 * Images with a `z` or `t` axis are opened as a single plane: `z` and `t`
 * select the plane, and default to the image's `omero` defaults (or to the
 * middle of the axis without them).
 */
export interface OMEZarrImageDataSource extends ImageDataSource<
  typeof omeZarrImageDataSourceType
> {
  /** URL or path of the OME-Zarr image (see `SourceUtils`) */
  source: string;

  /** Timepoint index (0-based) to open, for images with a `t` axis */
  t?: number;

  /** Z-slice index (0-based) to open, for images with a `z` axis */
  z?: number;
}

/**
 * An {@link OMEZarrImageDataSource} with {@link omeZarrImageDataSourceDefaults}
 * applied and its source normalized
 */
export type NormalizedOMEZarrImageDataSource = Required<
  Pick<OMEZarrImageDataSource, keyof typeof omeZarrImageDataSourceDefaults>
> &
  Omit<OMEZarrImageDataSource, keyof typeof omeZarrImageDataSourceDefaults>;
