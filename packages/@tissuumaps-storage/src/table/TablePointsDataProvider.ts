import {
  type AnnotatedDataProviderLoadOptions,
  AsyncUtils,
  type PointsDataProvider,
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
      table: {
        type: "string",
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
    required: ["table"],
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

  /**
   * Creates the data provider
   *
   * @param options - `getTableDataProviders` returns the table data providers
   *   registered with the application, by data source type; it is called
   *   whenever they are needed, as providers may be registered at any time
   */
  constructor(options: {
    getTableDataProviders: () => Map<
      string,
      TableDataProvider<TableDataSource, TableData>
    >;
  }) {
    const { getTableDataProviders } = options;
    this._getTableDataProviders = getTableDataProviders;
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
      supported = false;
    }
    signal?.throwIfAborted();
    return supported;
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
