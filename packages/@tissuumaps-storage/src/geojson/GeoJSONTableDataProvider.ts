import {
  type DataProviderLoadOptions,
  SourceUtils,
  type TableDataProvider,
} from "@tissuumaps/core";

import { GeoJSONTableData } from "./GeoJSONTableData";
import {
  type GeoJSONTableDataSource,
  type NormalizedGeoJSONTableDataSource,
  geoJSONTableDataSourceDefaults,
} from "./GeoJSONTableDataSource";
import { runGeoJSONWorker } from "./runGeoJSONWorker";

/**
 * Reads the feature properties of a GeoJSON file as a table
 *
 * The table is meant to annotate the shapes of the same file, which add it
 * themselves (see `GeoJSONShapesDataProvider.prepareDataSource`); it does not
 * claim GeoJSON files when tables are added, so that they keep being added as
 * shapes.
 */
export class GeoJSONTableDataProvider implements TableDataProvider<
  GeoJSONTableDataSource,
  GeoJSONTableData,
  NormalizedGeoJSONTableDataSource
> {
  readonly name = "GeoJSON properties";

  readonly schema = {
    type: "object",
    properties: {
      source: {
        type: "string",
      },
      idProperty: {
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
        scope: "#/properties/idProperty",
        label: "ID Property",
      },
    ],
  };

  normalize(
    dataSource: GeoJSONTableDataSource,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
  ): NormalizedGeoJSONTableDataSource {
    return {
      ...geoJSONTableDataSourceDefaults,
      ...dataSource,
      source: SourceUtils.normalizeSource(
        dataSource.source,
        workspace,
        projectSource,
      ),
    };
  }

  async load(
    normalizedDataSource: NormalizedGeoJSONTableDataSource,
    options?: DataProviderLoadOptions,
  ): Promise<GeoJSONTableData> {
    const { signal, onProgress, workspace = null } = options ?? {};
    signal?.throwIfAborted();
    const { file, url } = await SourceUtils.openSourceFile(
      normalizedDataSource.source,
      workspace,
      { signal },
    );
    const { ids, columns, columnValues } = await runGeoJSONWorker(
      {
        op: "table",
        file,
        url,
        idProperty: normalizedDataSource.idProperty,
      },
      { signal, onProgress },
    );
    return new GeoJSONTableData(
      ids.length,
      ids,
      undefined,
      columns,
      columnValues,
    );
  }
}
