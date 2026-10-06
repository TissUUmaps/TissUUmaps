import {
  type DataProviderLoadOptions,
  type ShapesDataProvider,
  SourceUtils,
} from "@tissuumaps/core";

import { ParquetShapesData } from "./ParquetShapesData";
import {
  type NormalizedParquetShapesDataSource,
  type ParquetShapesDataSource,
  parquetShapesDataSourceDefaults,
} from "./ParquetShapesDataSource";
import { runParquetWorker } from "./runParquetWorker";

export class ParquetShapesDataProvider implements ShapesDataProvider<
  ParquetShapesDataSource,
  ParquetShapesData,
  NormalizedParquetShapesDataSource
> {
  /** The file extensions of the (Geo)Parquet sources to check for shapes */
  private static readonly _extensions = new Set([".parquet", ".geoparquet"]);

  readonly name = "Parquet";

  readonly schema = {
    type: "object",
    properties: {
      source: {
        type: "string",
      },
      geometryColumn: {
        type: "string",
      },
      idColumn: {
        type: "string",
      },
      nameColumn: {
        type: "string",
      },
      table: {
        type: "string",
      },
      requestHeaders: {
        type: "object",
        additionalProperties: { type: "string" },
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
        scope: "#/properties/geometryColumn",
        label: "Geometry Column",
      },
      {
        type: "Control",
        scope: "#/properties/idColumn",
        label: "ID Column",
      },
      {
        type: "Control",
        scope: "#/properties/nameColumn",
        label: "Name Column",
      },
      {
        type: "Control",
        scope: "#/properties/table",
        label: "Table",
      },
    ],
  };

  normalize(
    dataSource: ParquetShapesDataSource,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
  ): NormalizedParquetShapesDataSource {
    return {
      ...parquetShapesDataSourceDefaults,
      ...dataSource,
      source: SourceUtils.normalizeSource(
        dataSource.source,
        workspace,
        projectSource,
      ),
    };
  }

  /**
   * Returns whether a source is a GeoParquet file of shapes
   *
   * A file with a (Geo)Parquet extension (see
   * {@link ParquetShapesDataProvider._extensions}) has to have a primary
   * geometry column that is not a point column, which is read from its
   * metadata without reading any data.
   *
   * @param normalizedSource - The normalized source to check
   * @param workspace - The directory handle of the open workspace, if any
   * @param options - Optional abort signal
   * @returns A promise that resolves to whether the source is supported
   * @throws Error if the source cannot be read, or if the operation is
   * aborted
   */
  async supports(
    normalizedSource: string,
    workspace: FileSystemDirectoryHandle | null,
    options?: { signal?: AbortSignal },
  ): Promise<boolean> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    if (
      !ParquetShapesDataProvider._extensions.has(
        SourceUtils.getExtension(normalizedSource),
      )
    ) {
      return false;
    }
    const { file, url } = await SourceUtils.openSourceFile(
      normalizedSource,
      workspace,
      { signal },
    );
    const { hasShapesColumn } = await runParquetWorker(
      { op: "geo", source: { file, url } },
      { signal },
    );
    return hasShapesColumn;
  }

  async load(
    normalizedDataSource: NormalizedParquetShapesDataSource,
    options?: DataProviderLoadOptions,
  ): Promise<ParquetShapesData> {
    const { signal, onProgress, workspace = null } = options ?? {};
    signal?.throwIfAborted();
    const { file, url } = await SourceUtils.openSourceFile(
      normalizedDataSource.source,
      workspace,
      { signal },
    );
    const parquetSource = {
      file,
      url,
      headers: normalizedDataSource.requestHeaders,
    };
    const { geometryColumn, idColumn, nameColumn } = normalizedDataSource;
    const { geometry, ids, names } = await runParquetWorker(
      {
        op: "shapes",
        source: parquetSource,
        geometryColumn,
        idColumn,
        nameColumn,
      },
      { signal, onProgress },
    );
    return new ParquetShapesData(geometry, ids, names);
  }
}
