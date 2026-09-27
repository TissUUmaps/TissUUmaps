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
      return new OpenSeadragonImageData(source.url);
    }
    const objectUrl = URL.createObjectURL(source.file);
    return new OpenSeadragonImageData(objectUrl, objectUrl);
  }
}
