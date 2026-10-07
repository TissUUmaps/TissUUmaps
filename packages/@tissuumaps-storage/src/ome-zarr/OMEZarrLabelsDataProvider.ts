import { OMEZarrTileSource } from "omezarr-tilesource";

import {
  type DataProviderLoadOptions,
  type LabelsDataProvider,
  SourceUtils,
} from "@tissuumaps/core";

import { OMEZarrLabelsData } from "./OMEZarrLabelsData";
import {
  type NormalizedOMEZarrLabelsDataSource,
  type OMEZarrLabelsDataSource,
  omeZarrLabelsDataSourceDefaults,
} from "./OMEZarrLabelsDataSource";
import { OMEZarrUtils } from "./OMEZarrUtils";
import { openOMEZarr } from "./openOMEZarr";

/**
 * Data provider for OME-Zarr label images
 *
 * Opens an {@link OMEZarrLabelsDataSource} as {@link OMEZarrLabelsData} with a
 * single tile source rendering the first channel; the image's channel axis, if
 * any, is not iterated, and `image-label` metadata is not read.
 */
export class OMEZarrLabelsDataProvider implements LabelsDataProvider<
  OMEZarrLabelsDataSource,
  OMEZarrLabelsData,
  NormalizedOMEZarrLabelsDataSource
> {
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
      table: {
        type: "string",
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
      {
        type: "Control",
        scope: "#/properties/table",
        label: "Table",
      },
    ],
  };

  /**
   * Returns the data source with {@link omeZarrLabelsDataSourceDefaults}
   * applied and its source normalized (see `SourceUtils.normalizeSource`)
   *
   * @param dataSource - The data source to normalize
   * @param workspace - The directory handle of the open workspace, if any
   * @param projectSource - Where the project was loaded from, if anywhere
   * @returns The normalized data source
   */
  normalize(
    dataSource: OMEZarrLabelsDataSource,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
  ): NormalizedOMEZarrLabelsDataSource {
    return {
      ...omeZarrLabelsDataSourceDefaults,
      ...dataSource,
      source: SourceUtils.normalizeSource(
        dataSource.source,
        workspace,
        projectSource,
      ),
    };
  }

  /**
   * Returns whether a source is an OME-Zarr image, which may be a label image
   * even without `image-label` attributes (see
   * {@link OMEZarrUtils.inspectImage})
   *
   * @param normalizedSource - The normalized source to check
   * @param workspace - The directory handle of the open workspace, if any
   * @param options - Optional abort signal
   * @returns A promise that resolves to whether the source is supported
   * @throws See {@link OMEZarrUtils.inspectImage}
   */
  async supports(
    normalizedSource: string,
    workspace: FileSystemDirectoryHandle | null,
    options?: { signal?: AbortSignal },
  ): Promise<boolean> {
    const image = await OMEZarrUtils.inspectImage(
      normalizedSource,
      workspace,
      options,
    );
    return image !== undefined;
  }

  /**
   * Reads the image's own name from its metadata (see
   * {@link OMEZarrUtils.readImageName})
   *
   * @param normalizedSource - The normalized source
   * @param workspace - The directory handle of the open workspace, if any
   * @param options - Optional abort signal
   * @returns A promise that resolves to the name, or to `undefined` if the
   *   image has none
   * @throws See {@link OMEZarrUtils.readImageName}
   */
  readName(
    normalizedSource: string,
    workspace: FileSystemDirectoryHandle | null,
    options?: { signal?: AbortSignal },
  ): Promise<string | undefined> {
    return OMEZarrUtils.readImageName(normalizedSource, workspace, options);
  }

  /**
   * Opens an OME-Zarr labels data source and returns the loaded labels data
   *
   * The OME-Zarr label image and its arrays are loaded with
   * {@link openOMEZarr}, and a single tile source rendering its first channel
   * is opened for them. The `z` and `t` of the data source select the plane to
   * open.
   *
   * @param normalizedDataSource - The normalized data source to open
   * @param options - See `DataProviderLoadOptions`; `workspace` is required
   * for workspace-relative sources
   * @returns A promise that resolves to the loaded labels data
   * @throws Error if the source is workspace-relative while no workspace is
   * open
   */
  async load(
    normalizedDataSource: NormalizedOMEZarrLabelsDataSource,
    options?: DataProviderLoadOptions,
  ): Promise<OMEZarrLabelsData> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const { loaded, url, zip } = await openOMEZarr(
      normalizedDataSource.source,
      options,
    );
    const { t, z } = normalizedDataSource;
    const tileSource = await OMEZarrTileSource.open(
      { url, zip, t, z, c: 0 },
      loaded,
      { signal },
    );
    return new OMEZarrLabelsData(tileSource);
  }
}
