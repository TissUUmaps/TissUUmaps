import {
  type DataProviderLoadOptions,
  type ImageDataProvider,
  SourceUtils,
} from "@tissuumaps/core";

import { OpenSeadragonImageData } from "./OpenSeadragonImageData";
import {
  type NormalizedOpenSeadragonImageDataSource,
  type OpenSeadragonImageDataSource,
  openSeadragonImageDataSourceDefaults,
} from "./OpenSeadragonImageDataSource";

export class OpenSeadragonImageDataProvider implements ImageDataProvider<
  OpenSeadragonImageDataSource,
  OpenSeadragonImageData,
  NormalizedOpenSeadragonImageDataSource
> {
  /**
   * The extensions of tile source descriptor files: DZI, and the XML and JSON
   * that DZI, IIIF and legacy image pyramid descriptors are served as
   */
  private static readonly _descriptorFileExtensions = new Set([
    ".dzi",
    ".json",
    ".xml",
  ]);

  /**
   * The extensions of plain image files: JPEG, PNG, WebP
   */
  private static readonly _imageFileExtensions = new Set([
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
  ]);

  /**
   * The prefix of data URLs that are images, which OpenSeadragon can open as
   * tile sources
   */
  private static readonly _imageDataUrlPrefix = "data:image/";

  readonly name = "Image (e.g. PNG, JPEG, DZI, IIIF)";

  readonly schema = {
    type: "object",
    properties: {
      source: { type: "string" },
      tileSource: { type: "object" },
    },
    anyOf: [{ required: ["source"] }, { required: ["tileSource"] }],
  };

  readonly uischema = {
    type: "VerticalLayout",
    elements: [
      // tileSource is not available through the UI for now
    ],
  };

  normalize(
    dataSource: OpenSeadragonImageDataSource,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
  ): NormalizedOpenSeadragonImageDataSource {
    if (dataSource.tileSource !== undefined) {
      return {
        ...openSeadragonImageDataSourceDefaults,
        ...dataSource,
        source: undefined,
      };
    }
    if (dataSource.source === undefined) {
      throw new Error("Either source or tileSource must be specified.");
    }
    const source = SourceUtils.normalizeSource(
      dataSource.source,
      workspace,
      projectSource,
    );
    if (
      !OpenSeadragonImageDataProvider._isImageSource(source) &&
      SourceUtils.isWorkspacePath(source)
    ) {
      throw new Error(
        `Tile sources cannot be opened from the workspace: ${source}`,
      );
    }
    return {
      ...openSeadragonImageDataSourceDefaults,
      ...dataSource,
      source,
    };
  }

  /**
   * Returns whether a source is a plain image or a tile source descriptor,
   * judged by its extension (see
   * {@link OpenSeadragonImageDataProvider._imageFileExtensions} and
   * {@link OpenSeadragonImageDataProvider._descriptorFileExtensions})
   *
   * Descriptors are only supported as URLs, as they cannot be opened from the
   * workspace (see {@link OpenSeadragonImageDataProvider.normalize}).
   *
   * @param normalizedSource - The normalized source to check
   * @returns A promise that resolves to whether the source is supported
   */
  supports(normalizedSource: string): Promise<boolean> {
    return Promise.resolve(
      OpenSeadragonImageDataProvider._isImageSource(normalizedSource) ||
        (!SourceUtils.isWorkspacePath(normalizedSource) &&
          OpenSeadragonImageDataProvider._descriptorFileExtensions.has(
            SourceUtils.getExtension(normalizedSource),
          )),
    );
  }

  async load(
    normalizedDataSource: NormalizedOpenSeadragonImageDataSource,
    options?: DataProviderLoadOptions,
  ): Promise<OpenSeadragonImageData> {
    const { signal, workspace = null } = options ?? {};
    signal?.throwIfAborted();
    const { source, tileSource } = normalizedDataSource;
    if (tileSource !== undefined) {
      return new OpenSeadragonImageData(tileSource);
    }
    if (source === undefined) {
      throw new Error("Either source or tileSource must be specified.");
    }
    if (!OpenSeadragonImageDataProvider._isImageSource(source)) {
      return new OpenSeadragonImageData(source);
    }
    const { url, file } = await SourceUtils.openSourceFile(source, workspace, {
      signal,
    });
    const objectUrl = url === undefined ? URL.createObjectURL(file) : undefined;
    return new OpenSeadragonImageData(
      {
        type: "image",
        url: url ?? objectUrl,
        crossOriginPolicy: "Anonymous",
        ajaxWithCredentials: false,
      },
      objectUrl,
    );
  }

  private static _isImageSource(normalizedSource: string): boolean {
    if (normalizedSource.startsWith("data:")) {
      return normalizedSource.startsWith(
        OpenSeadragonImageDataProvider._imageDataUrlPrefix,
      );
    }
    return OpenSeadragonImageDataProvider._imageFileExtensions.has(
      SourceUtils.getExtension(normalizedSource),
    );
  }
}
