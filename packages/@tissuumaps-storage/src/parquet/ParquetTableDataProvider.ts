import {
  type DataProviderLoadOptions,
  SourceUtils,
  type TableDataProvider,
} from "@tissuumaps/core";

import { ParquetTableData } from "./ParquetTableData";
import {
  type NormalizedParquetTableDataSource,
  type ParquetTableDataSource,
  parquetTableDataSourceDefaults,
} from "./ParquetTableDataSource";
import { runParquetWorker } from "./runParquetWorker";

export class ParquetTableDataProvider implements TableDataProvider<
  ParquetTableDataSource,
  ParquetTableData,
  NormalizedParquetTableDataSource
> {
  readonly name = "Parquet";

  readonly schema = {
    type: "object",
    properties: {
      source: {
        type: "string",
      },
      idColumn: {
        type: "string",
      },
      nameColumn: {
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
        scope: "#/properties/idColumn",
        label: "ID Column",
      },
      {
        type: "Control",
        scope: "#/properties/nameColumn",
        label: "Name Column",
      },
    ],
  };

  normalize(
    dataSource: ParquetTableDataSource,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
  ): NormalizedParquetTableDataSource {
    return {
      ...parquetTableDataSourceDefaults,
      ...dataSource,
      source: SourceUtils.normalizeSource(
        dataSource.source,
        workspace,
        projectSource,
      ),
    };
  }

  async load(
    normalizedDataSource: NormalizedParquetTableDataSource,
    options?: DataProviderLoadOptions,
  ): Promise<ParquetTableData> {
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
    const { idColumn, nameColumn } = normalizedDataSource;
    const { numRows, columns, coordinateColumns, ids, names } =
      await runParquetWorker(
        { op: "file", source: parquetSource, idColumn, nameColumn },
        { signal, onProgress },
      );
    return new ParquetTableData(
      parquetSource,
      numRows,
      columns,
      coordinateColumns,
      ids,
      names,
    );
  }
}
