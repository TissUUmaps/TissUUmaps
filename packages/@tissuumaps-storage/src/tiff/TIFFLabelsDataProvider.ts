import type { GeoTIFFImage } from "geotiff";

import {
  type DataProviderLoadOptions,
  type LabelsDataProvider,
  SourceUtils,
} from "@tissuumaps/core";

import { TIFFLabelsData } from "./TIFFLabelsData";
import {
  type NormalizedTIFFLabelsDataSource,
  type TIFFLabelsDataSource,
  tiffLabelsDataSourceDefaults,
} from "./TIFFLabelsDataSource";
import { type TIFFStructure, findTIFFParser } from "./TIFFParser";
import { TIFFUtils } from "./TIFFUtils";
import { installTIFFTileSource } from "./installTIFFTileSource";
import { openTIFF } from "./openTIFF";
import { OMETIFFParser } from "./profiles/OMETIFFParser";

/**
 * Data provider for label masks stored in TIFF files
 *
 * The file is read like any other TIFF (see `TIFFImageDataProvider`), but its
 * pixels are label IDs rather than intensities: it has to hold a single
 * channel of integers, which the renderer colors per ID instead of
 * contrast-stretching them.
 *
 * Which labels the mask holds is not recorded in the file, and is not read
 * from its pixels: the renderer resolves each label as it is first drawn.
 */
export class TIFFLabelsDataProvider implements LabelsDataProvider<
  TIFFLabelsDataSource,
  TIFFLabelsData,
  NormalizedTIFFLabelsDataSource
> {
  /** The file extensions of the sources this data provider supports */
  private static readonly _extensions = new Set([".tif", ".tiff", ".qptiff"]);

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
   * Returns the data source with {@link tiffLabelsDataSourceDefaults} applied
   * and its source normalized (see `SourceUtils.normalizeSource`)
   *
   * @param dataSource - The data source to normalize
   * @param workspace - The directory handle of the open workspace, if any
   * @param projectSource - Where the project was loaded from, if anywhere
   * @returns The normalized data source
   */
  normalize(
    dataSource: TIFFLabelsDataSource,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
  ): NormalizedTIFFLabelsDataSource {
    return {
      ...tiffLabelsDataSourceDefaults,
      ...dataSource,
      source: SourceUtils.normalizeSource(
        dataSource.source,
        workspace,
        projectSource,
      ),
    };
  }

  /**
   * Returns whether a source has a TIFF file extension (see
   * {@link TIFFLabelsDataProvider._extensions})
   *
   * @param normalizedSource - The normalized source to check
   * @returns A promise that resolves to whether the source is supported
   */
  supports(normalizedSource: string): Promise<boolean> {
    return Promise.resolve(
      TIFFLabelsDataProvider._extensions.has(
        SourceUtils.getExtension(normalizedSource),
      ),
    );
  }

  /**
   * Reads the image's own name for a data object backed by a source, which
   * only OME-TIFF files have (see {@link OMETIFFParser.readImageName})
   *
   * @param normalizedSource - The normalized source
   * @param workspace - The directory handle of the open workspace, if any
   * @param options - Optional abort signal
   * @returns A promise that resolves to the name, if any
   * @throws Error if the source is workspace-relative while no workspace is
   * open, or if the file cannot be read
   */
  async readName(
    normalizedSource: string,
    workspace: FileSystemDirectoryHandle | null,
    options?: { signal?: AbortSignal },
  ): Promise<string | undefined> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const tiff = await openTIFF(normalizedSource, { workspace, signal });
    return await OMETIFFParser.readImageName(tiff, { signal });
  }

  /**
   * Opens a TIFF labels data source and returns the loaded label mask
   *
   * The file is opened with {@link openTIFF}, and its structure read by the
   * parser of its format (see {@link findTIFFParser}), with `z` and `t`
   * selecting the plane. The pixels are not read: the renderer resolves the
   * labels as it draws them.
   *
   * @param normalizedDataSource - The normalized data source to open
   * @param options - See `DataProviderLoadOptions`; `workspace` is required
   * for workspace-relative sources
   * @returns A promise that resolves to the loaded label mask
   * @throws Error if the source is workspace-relative while no workspace is
   * open, or if the file is not a single channel of integers of at most 32
   * bits
   */
  async load(
    normalizedDataSource: NormalizedTIFFLabelsDataSource,
    options?: DataProviderLoadOptions,
  ): Promise<TIFFLabelsData> {
    const { signal, workspace } = options ?? {};
    signal?.throwIfAborted();

    const { z, t } = normalizedDataSource;
    const tiff = await openTIFF(normalizedDataSource.source, {
      signal,
      workspace,
    });
    const parser = await findTIFFParser(tiff, { signal });
    const structure = await parser.load(tiff, { z, t, signal });
    const levels = getLabelLevels(structure);

    const { GeoTIFFTileSource } = installTIFFTileSource();
    const tileSource = new GeoTIFFTileSource({
      GeoTIFF: tiff,
      GeoTIFFImages: levels,
    });
    return new TIFFLabelsData(tileSource);
  }
}

/**
 * Returns the pyramid levels of a structure that holds a label mask
 *
 * @param structure - The structure of the file
 * @returns The images of the mask, one per level, largest first
 * @throws Error if the file is drawn in its own colors, holds more than one
 * channel, or holds samples that are not signed or unsigned integers of at
 * most 32 bits, none of which can be read as label IDs
 */
export function getLabelLevels(structure: TIFFStructure): GeoTIFFImage[] {
  const { channels, pyramids } = structure;
  if (channels === undefined) {
    throw new Error(
      "The file holds an image drawn in its own colors, whose pixels are colors rather than label IDs.",
    );
  }
  if (channels.length !== 1) {
    throw new Error(
      `The file holds ${channels.length} channels; a label mask has a single one.`,
    );
  }
  const levels = pyramids[0]!;
  const image = levels[0]!;
  if (TIFFUtils.getIntegerSampleType(image) === undefined) {
    throw new Error(
      `The image holds ${image.getBitsPerSample(0)}-bit samples of TIFF sample format ${image.getSampleFormat(0)}; label IDs are integers of at most 32 bits.`,
    );
  }
  return levels;
}
