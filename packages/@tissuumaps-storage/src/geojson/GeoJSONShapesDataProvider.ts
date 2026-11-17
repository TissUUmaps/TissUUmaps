import {
  type DataProviderLoadOptions,
  type ShapesDataProvider,
  SourceUtils,
  type TableDataSource,
} from "@tissuumaps/core";

import { GeoJSONShapesData } from "./GeoJSONShapesData";
import {
  type GeoJSONShapesDataSource,
  type NormalizedGeoJSONShapesDataSource,
  geoJSONShapesDataSourceDefaults,
} from "./GeoJSONShapesDataSource";
import {
  type GeoJSONTableDataSource,
  geoJSONTableDataSourceType,
} from "./GeoJSONTableDataSource";
import { runGeoJSONWorker } from "./runGeoJSONWorker";

export class GeoJSONShapesDataProvider implements ShapesDataProvider<
  GeoJSONShapesDataSource,
  GeoJSONShapesData,
  NormalizedGeoJSONShapesDataSource
> {
  /** The file extensions of the sources this data provider supports */
  private static readonly _extensions = new Set([".geojson"]);

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

  private readonly _addTable:
    | ((
        dataSource: TableDataSource,
        options?: { signal?: AbortSignal },
      ) => Promise<string>)
    | undefined;

  /**
   * Creates the data provider
   *
   * @param options - `addTable` adds a table backed by a data source to the
   *   project and resolves to its ID; without it, the feature properties are
   *   not added as a table (see
   *   {@link GeoJSONShapesDataProvider.prepareDataSource})
   */
  constructor(options?: {
    addTable?: (
      dataSource: TableDataSource,
      options?: { signal?: AbortSignal },
    ) => Promise<string>;
  }) {
    this._addTable = options?.addTable;
  }

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

  /**
   * Returns whether a source has a GeoJSON file extension (see
   * {@link GeoJSONShapesDataProvider._extensions})
   *
   * @param normalizedSource - The normalized source to check
   * @returns A promise that resolves to whether the source is supported
   */
  supports(normalizedSource: string): Promise<boolean> {
    return Promise.resolve(
      GeoJSONShapesDataProvider._extensions.has(
        SourceUtils.getExtension(normalizedSource),
      ),
    );
  }

  /**
   * Adds the feature properties of the file as the table of the shapes,
   * unless the data source references a table already
   *
   * The table is keyed by the same IDs as the shapes (see
   * `GeoJSONTableDataProvider`), so that its rows annotate them by ID rather
   * than by order. The file is parsed again for the table.
   *
   * @param dataSource - The data source to prepare
   * @param _workspace - Unused: the table is normalized when it is loaded
   * @param _projectSource - Unused, as `_workspace`
   * @param options - Optional abort signal
   * @returns A promise that resolves to the data source referencing the table
   * @throws Error if the operation is aborted
   */
  async prepareDataSource(
    dataSource: GeoJSONShapesDataSource,
    _workspace: FileSystemDirectoryHandle | null,
    _projectSource: string | null,
    options?: { signal?: AbortSignal },
  ): Promise<GeoJSONShapesDataSource> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    if (dataSource.table !== undefined || this._addTable === undefined) {
      return dataSource;
    }
    const tableDataSource: GeoJSONTableDataSource = {
      type: geoJSONTableDataSourceType,
      source: dataSource.source,
      ...(dataSource.idProperty !== undefined && {
        idProperty: dataSource.idProperty,
      }),
    };
    const table = await this._addTable(tableDataSource, { signal });
    return { ...dataSource, table };
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
