import {
  type DataProviderLoadOptions,
  type ShapesDataProvider,
  SourceUtils,
} from "@tissuumaps/core";

import { GeoJSONShapesData } from "./GeoJSONShapesData";
import {
  type GeoJSONShapesDataSource,
  type NormalizedGeoJSONShapesDataSource,
  geoJSONShapesDataSourceDefaults,
} from "./GeoJSONShapesDataSource";
import { runGeoJSONWorker } from "./runGeoJSONWorker";

export class GeoJSONShapesDataProvider implements ShapesDataProvider<
  GeoJSONShapesDataSource,
  GeoJSONShapesData,
  NormalizedGeoJSONShapesDataSource
> {
  readonly name = "GeoJSON";

  readonly schema = {
    type: "object",
    properties: {
      source: {
        type: "string",
      },
      idProperty: {
        type: "string",
      },
      nameProperty: {
        type: "string",
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
        type: "Control",
        scope: "#/properties/idProperty",
        label: "ID Property",
      },
      {
        type: "Control",
        scope: "#/properties/nameProperty",
        label: "Name Property",
      },
      {
        type: "Control",
        scope: "#/properties/table",
        label: "Table",
      },
    ],
  };

  normalize(
    dataSource: GeoJSONShapesDataSource,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
  ): NormalizedGeoJSONShapesDataSource {
    return {
      ...geoJSONShapesDataSourceDefaults,
      ...dataSource,
      source: SourceUtils.normalizeSource(
        dataSource.source,
        workspace,
        projectSource,
      ),
    };
  }

  async load(
    normalizedDataSource: NormalizedGeoJSONShapesDataSource,
    options?: DataProviderLoadOptions,
  ): Promise<GeoJSONShapesData> {
    const { signal, onProgress, workspace = null } = options ?? {};
    signal?.throwIfAborted();
    const { file, url } = await SourceUtils.openSourceFile(
      normalizedDataSource.source,
      workspace,
      { signal },
    );
    const { idProperty, nameProperty } = normalizedDataSource;
    const { ids, names, geometry } = await runGeoJSONWorker(
      { op: "file", file, url, idProperty, nameProperty },
      { signal, onProgress },
    );
    return new GeoJSONShapesData(geometry, ids, names);
  }
}
