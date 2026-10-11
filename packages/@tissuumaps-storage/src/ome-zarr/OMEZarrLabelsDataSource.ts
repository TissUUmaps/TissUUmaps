import type { LabelsDataSource } from "@tissuumaps/core";

/** Type discriminator of OME-Zarr labels data sources */
export const omeZarrLabelsDataSourceType = "ome-zarr";

/** Default values for {@link OMEZarrLabelsDataSource} */
export const omeZarrLabelsDataSourceDefaults = {};

/**
 * Data source for OME-Zarr label images
 *
 * The `source` refers to a remote OME-Zarr store (a URL), a remote zipped
 * OME-Zarr file (a URL with the zipped OME-Zarr extension, in any case, see
 * `OMEZarrUtils.isZipSource`), or an OME-Zarr store (a directory) or zipped
 * OME-Zarr file in the open workspace (a workspace-relative or
 * project-relative path).
 *
 * Label images have to hold signed or unsigned integers of up to 32 bits.
 * Their `image-label` metadata is not read (see `LabelsDataSource` for the
 * annotation table reference).
 *
 * Label images with a `z` or `t` axis are opened as a single plane: `z` and
 * `t` select the plane, and default to the image's `omero` defaults (or to the
 * middle of the axis without them).
 */
export interface OMEZarrLabelsDataSource extends LabelsDataSource<
  typeof omeZarrLabelsDataSourceType
> {
  /** URL or path of the OME-Zarr label image (see `SourceUtils`) */
  source: string;

  /** Timepoint index (0-based) to open, for label images with a `t` axis */
  t?: number;

  /** Z-slice index (0-based) to open, for label images with a `z` axis */
  z?: number;
}

/**
 * An {@link OMEZarrLabelsDataSource} with
 * {@link omeZarrLabelsDataSourceDefaults} applied and its source normalized
 */
export type NormalizedOMEZarrLabelsDataSource = Required<
  Pick<OMEZarrLabelsDataSource, keyof typeof omeZarrLabelsDataSourceDefaults>
> &
  Omit<OMEZarrLabelsDataSource, keyof typeof omeZarrLabelsDataSourceDefaults>;
