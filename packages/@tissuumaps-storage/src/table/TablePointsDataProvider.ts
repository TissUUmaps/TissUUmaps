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
  /**
   * Coordinate column names of common spatial omics exports, tried in order
   * when the table suggests no coordinate columns itself: generic `x`/`y`
   * (also scanpy's `obs.to_csv()` of spatial coordinates), 10x Xenium
   * transcripts (`x_location`) and cells (`x_centroid`), Vizgen MERSCOPE
   * transcripts (`global_x`) and cells (`center_x`), and NanoString CosMx
   * (`x_global_px`)
   */
  private static readonly _coordinateColumnNames = [
    ["x", "y"],
    ["x_location", "y_location"],
    ["x_centroid", "y_centroid"],
    ["global_x", "global_y"],
    ["center_x", "center_y"],
    ["x_global_px", "y_global_px"],
  ] as const;

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
      // no defaults here: unset coordinates are detected from the table
      // (see prepareDataSource)
      x: {
        type: "string",
      },
      y: {
        type: "string",
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
        options: {
          tableColumn: true,
          coordinate: "x",
          placeholder: "Detected automatically",
        },
      },
      {
        type: "Control",
        scope: "#/properties/y",
        label: "Y column",
        options: {
          tableColumn: true,
          coordinate: "y",
          placeholder: "Detected automatically",
        },
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

  private readonly _loadTableData: (
    tableId: string,
    options?: { signal?: AbortSignal },
  ) => Promise<TableData>;

  /**
   * Creates the data provider
   *
   * @param options - `getTableDataProviders` returns the table data providers
   *   registered with the application, by data source type; it is called
   *   whenever they are needed, as providers may be registered at any time.
   *   `addTable` adds a table backed by a data source to the project and
   *   resolves to its ID (see {@link TablePointsDataProvider.prepareDataSource});
   *   `loadTableData` loads the data of a table of the project, through
   *   whatever shares and owns it (e.g. a cache), as this provider must not
   *   open tables itself
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
    loadTableData: (
      tableId: string,
      options?: { signal?: AbortSignal },
    ) => Promise<TableData>;
  }) {
    const { getTableDataProviders, addTable, loadTableData } = options;
    this._getTableDataProviders = getTableDataProviders;
    this._addTable = addTable;
    this._loadTableData = loadTableData;
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
   * Unset coordinate columns are detected from the added table: the first
   * pair the table records (see `TableData.getCoordinateColumns`), else
   * the first pair of common column names that the table resolves (see
   * {@link TablePointsDataProvider._coordinateColumnNames}), else the
   * defaults (see {@link tablePointsDataSourceDefaults}). Coordinate columns
   * that are set are kept.
   *
   * @param dataSource - The data source to prepare
   * @param workspace - The directory handle of the open workspace, if any
   * @param projectSource - Where the project was loaded from, if anywhere
   * @param options - Optional abort signal
   * @returns A promise that resolves to the data source to add, without a
   *   source
   * @throws Error if no table data provider supports the source, if the
   *   source cannot be normalized, if the added table cannot be loaded, or if
   *   the operation is aborted
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
      throw new Error(`No table data provider supports '${source}'`);
    }
    const table = await this._addTable({ type, source }, { signal });
    if (dataSource.x !== undefined && dataSource.y !== undefined) {
      return { ...dataSourceWithoutSource, table };
    }
    const tableData = await this._loadTableData(table, { signal });
    const coordinateColumns = await TablePointsDataProvider._detectCoordinates(
      tableData,
      { signal },
    );
    return {
      ...dataSourceWithoutSource,
      table,
      x: dataSource.x ?? coordinateColumns.x,
      y: dataSource.y ?? coordinateColumns.y,
    };
  }

  /**
   * Detects the coordinate columns of a table, see
   * {@link TablePointsDataProvider.prepareDataSource}
   *
   * @param tableData - The table data
   * @param options - Optional abort signal
   * @returns The coordinate column queries
   */
  private static async _detectCoordinates(
    tableData: TableData,
    options?: { signal?: AbortSignal },
  ): Promise<{ x: string; y: string }> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const pair = tableData.getCoordinateColumns?.()[0];
    if (pair !== undefined) {
      return { x: pair.x, y: pair.y };
    }
    for (const [
      xName,
      yName,
    ] of TablePointsDataProvider._coordinateColumnNames) {
      const [x, y] = await Promise.all([
        tableData.resolveColumnQuery(xName, { signal }),
        tableData.resolveColumnQuery(yName, { signal }),
      ]);
      if (x !== null && y !== null) {
        return { x, y };
      }
    }
    return { ...tablePointsDataSourceDefaults };
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
