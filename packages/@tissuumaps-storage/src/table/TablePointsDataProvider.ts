import {
  type AnnotatedDataProviderLoadOptions,
  AsyncUtils,
  type PointsDataProvider,
  SourceUtils,
  type TableData,
  type TableDataProvider,
  type TableDataSource,
} from "@tissuumaps/core";

import { TablePointsData } from "./TablePointsData";
import {
  type NormalizedTablePointsDataSource,
  type TablePointsDataSource,
  tablePointsDataSourceDefaults,
} from "./TablePointsDataSource";

export class TablePointsDataProvider implements PointsDataProvider<
  TablePointsDataSource,
  TablePointsData,
  NormalizedTablePointsDataSource
> {
  readonly name = "Table";

  readonly schema = {
    type: "object",
    properties: {
      source: {
        type: "string",
      },
      table: {
        type: "string",
        description: "Ignored if a source is given, which is added as a table",
      },
      x: {
        type: "string",
        default: tablePointsDataSourceDefaults.x,
      },
      y: {
        type: "string",
        default: tablePointsDataSourceDefaults.y,
      },
    },
    anyOf: [{ required: ["source"] }, { required: ["table"] }],
  };

  readonly uischema = {
    type: "VerticalLayout",
    elements: [
      {
        type: "Control",
        scope: "#/properties/table",
        label: "Table",
      },
      {
        type: "Control",
        scope: "#/properties/x",
        label: "X column",
      },
      {
        type: "Control",
        scope: "#/properties/y",
        label: "Y column",
      },
    ],
  };

  private readonly _getTableDataProviders: () => Map<
    string,
    TableDataProvider<TableDataSource, TableData>
  >;

  private readonly _addTable: (
    dataSource: TableDataSource,
    options?: { signal?: AbortSignal },
  ) => Promise<string>;

  /**
   * Creates the data provider
   *
   * @param options - `getTableDataProviders` returns the table data providers
   *   registered with the application, by data source type; it is called
   *   whenever they are needed, as providers may be registered at any time.
   *   `addTable` adds a table backed by a data source to the project and
   *   resolves to its ID (see {@link TablePointsDataProvider.prepareDataSource})
   */
  constructor(options: {
    getTableDataProviders: () => Map<
      string,
      TableDataProvider<TableDataSource, TableData>
    >;
    addTable: (
      dataSource: TableDataSource,
      options?: { signal?: AbortSignal },
    ) => Promise<string>;
  }) {
    const { getTableDataProviders, addTable } = options;
    this._getTableDataProviders = getTableDataProviders;
    this._addTable = addTable;
  }

  normalize(
    dataSource: TablePointsDataSource,
  ): NormalizedTablePointsDataSource {
    return { ...tablePointsDataSourceDefaults, ...dataSource };
  }

  /**
   * Returns whether a source is a table that any of the table data providers
   * supports
   *
   * @param normalizedSource - The normalized source to check
   * @param workspace - The directory handle of the open workspace, if any
   * @param options - Optional abort signal
   * @returns A promise that resolves to whether the source is supported
   * @throws Error if the operation is aborted
   */
  async supports(
    normalizedSource: string,
    workspace: FileSystemDirectoryHandle | null,
    options?: { signal?: AbortSignal },
  ): Promise<boolean> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    let supported: boolean;
    try {
      supported = await Promise.any(
        Array.from(
          this._getTableDataProviders().values(),
          async (tableDataProvider) => {
            signal?.throwIfAborted();
            if (tableDataProvider.supports !== undefined) {
              const tableSupported = await tableDataProvider.supports(
                normalizedSource,
                workspace,
                options,
              );
              if (tableSupported) {
                return true;
              }
            }
            throw new Error("Unsupported");
          },
        ),
      );
    } catch {
      signal?.throwIfAborted();
      supported = false;
    }
    return supported;
  }

  /**
   * Adds the table file a data source names as a new table, and returns the
   * data source referencing that table instead
   *
   * The table is backed by the first table data provider, in registration
   * order, that supports the file. A data source without a source is returned
   * as it is.
   *
   * @param dataSource - The data source to prepare
   * @param workspace - The directory handle of the open workspace, if any
   * @param projectSource - Where the project was loaded from, if anywhere
   * @param options - Optional abort signal
   * @returns A promise that resolves to the data source to add, without a
   *   source
   * @throws Error if no table data provider supports the source, if the
   *   source cannot be normalized, or if the operation is aborted
   */
  async prepareDataSource(
    dataSource: TablePointsDataSource,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
    options?: { signal?: AbortSignal },
  ): Promise<TablePointsDataSource> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const { source, ...dataSourceWithoutSource } = dataSource;
    if (source === undefined) {
      return dataSource;
    }
    const normalizedSource = SourceUtils.normalizeSource(
      source,
      workspace,
      projectSource,
    );
    const checks = Array.from(
      this._getTableDataProviders(),
      async ([type, provider]) => {
        let supported = false;
        if (provider.supports !== undefined) {
          try {
            supported = await provider.supports(normalizedSource, workspace, {
              signal,
            });
          } catch {
            // ignored intentionally
          }
        }
        return { type, supported };
      },
    );
    let type: string | undefined;
    for (const check of checks) {
      const checkResult = await check;
      signal?.throwIfAborted();
      if (checkResult.supported) {
        type = checkResult.type;
        break;
      }
    }
    if (type === undefined) {
      throw new Error(`No table data provider supports ${source}`);
    }
    const table = await this._addTable({ type, source }, { signal });
    return { ...dataSourceWithoutSource, table };
  }

  async load(
    normalizedDataSource: NormalizedTablePointsDataSource,
    options?: AnnotatedDataProviderLoadOptions,
  ): Promise<TablePointsData> {
    const { signal, tableDataPromise } = options ?? {};
    signal?.throwIfAborted();
    if (tableDataPromise === undefined) {
      throw new Error("Table data must be provided");
    }
    const tableData = await AsyncUtils.raceSignal(tableDataPromise, { signal });
    return new TablePointsData(
      tableData,
      normalizedDataSource.x,
      normalizedDataSource.y,
    );
  }
}
