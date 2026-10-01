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
  readonly name = "OpenSeadragon";

  readonly schema = {
    type: "object",
    properties: {
      source: {
        type: "string",
      },
      // TODO tileSourceConfig
    },
    required: ["source"], // TODO ... or tileSourceConfig
  };

  readonly uischema = {
    type: "VerticalLayout",
    elements: [
      {
        type: "Control",
        scope: "#/properties/source",
        label: "Source",
      },
      // TODO tileSourceConfig
    ],
  };

  normalize(
    dataSource: OpenSeadragonImageDataSource,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
  ): NormalizedOpenSeadragonImageDataSource {
    let { source } = dataSource;
    if (source !== undefined) {
      source = SourceUtils.normalizeSource(source, workspace, projectSource);
    }
    return {
      ...openSeadragonImageDataSourceDefaults,
      ...dataSource,
      source,
    };
  }

  async load(
    normalizedDataSource: NormalizedOpenSeadragonImageDataSource,
    options?: DataProviderLoadOptions,
  ): Promise<OpenSeadragonImageData> {
    const { signal, workspace = null } = options ?? {};
    signal?.throwIfAborted();

    if (normalizedDataSource.tileSourceConfig !== undefined) {
      if (normalizedDataSource.source !== undefined) {
        throw new Error(
          "Specify either a tile source configuration or a source, not both.",
        );
      }
      return new OpenSeadragonImageData(normalizedDataSource.tileSourceConfig);
    }
    if (normalizedDataSource.source === undefined) {
      throw new Error(
        "A tile source configuration or a source is required to load data.",
      );
    }
    const source = await SourceUtils.openSourceFile(
      normalizedDataSource.source,
      workspace,
      { signal },
    );
    if (source.url !== undefined) {
      return new OpenSeadragonImageData(
        isImageURL(source.url)
          ? { type: "image", url: source.url }
          : source.url,
      );
    }
    const objectUrl = URL.createObjectURL(source.file);
    return new OpenSeadragonImageData(
      source.file.type.startsWith("image/")
        ? { type: "image", url: objectUrl }
        : objectUrl,
      objectUrl,
    );
  }
}

/** File extensions of plain images, which OpenSeadragon opens as a single tile */
const imageExtensionPattern = /\.(jpe?g|png|gif|webp|bmp|avif|svg)$/i;

/**
 * Returns whether an absolute URL points to a plain image rather than to a
 * tile source descriptor
 *
 * @param url - The absolute URL
 * @returns `true` if the URL's path has an image file extension
 */
export function isImageURL(url: string): boolean {
  return imageExtensionPattern.test(new URL(url).pathname);
}
