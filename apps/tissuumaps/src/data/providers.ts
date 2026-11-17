import { SourceUtils, type TableDataSource } from "@tissuumaps/core";
import {
  CSVTableDataProvider,
  GeoJSONShapesDataProvider,
  GeoJSONTableDataProvider,
  HDF5TableDataProvider,
  OMEZarrImageDataProvider,
  OMEZarrLabelsDataProvider,
  OpenSeadragonImageDataProvider,
  ParquetShapesDataProvider,
  ParquetTableDataProvider,
  TIFFImageDataProvider,
  TIFFLabelsDataProvider,
  TablePointsDataProvider,
  ZarrTableDataProvider,
  csvTableDataSourceType,
  geoJSONShapesDataSourceType,
  geoJSONTableDataSourceType,
  hdf5TableDataSourceType,
  omeZarrImageDataSourceType,
  omeZarrLabelsDataSourceType,
  openSeadragonImageDataSourceType,
  parquetShapesDataSourceType,
  parquetTableDataSourceType,
  tablePointsDataSourceType,
  tiffImageDataSourceType,
  tiffLabelsDataSourceType,
  zarrTableDataSourceType,
} from "@tissuumaps/storage";

import { addTableDataObject } from "@/data/io/dataObject";
import { appStore } from "@/stores/app";
import { projectStore } from "@/stores/project";

/**
 * Adds a table backed by a data source to the project, for data providers
 * that add the tables their data sources reference (see
 * `DataProvider.prepareDataSource`)
 *
 * The table is named after its source, and its own data source is prepared
 * first.
 *
 * @param dataSource - The data source of the table
 * @param options - Optional abort signal
 * @returns The ID of the added table
 * @throws Error if no table data provider is registered for the type, or if
 * preparing the data source fails
 */
async function addTable(
  dataSource: TableDataSource,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const { tableDataProviders, workspace } = appStore.getState();
  const tableDataProvider = tableDataProviders.get(dataSource.type);
  if (tableDataProvider === undefined) {
    throw new Error(`No table data provider for type ${dataSource.type}.`);
  }
  const projectSource = projectStore.getState().source;
  const normalizedSource =
    dataSource.source !== undefined
      ? SourceUtils.normalizeSource(dataSource.source, workspace, projectSource)
      : undefined;
  let name: string | undefined;
  if (normalizedSource !== undefined) {
    if (tableDataProvider.readName !== undefined) {
      try {
        name = await tableDataProvider.readName(normalizedSource, workspace, {
          signal,
        });
      } catch {
        signal?.throwIfAborted();
      }
    }
    name ??= SourceUtils.getStem(normalizedSource);
  }
  let preparedDataSource = dataSource;
  if (tableDataProvider.prepareDataSource !== undefined) {
    preparedDataSource = await tableDataProvider.prepareDataSource(
      dataSource,
      workspace,
      projectSource,
      { signal },
    );
  }
  return addTableDataObject(
    name?.trim() || "Untitled",
    dataSource.source,
    preparedDataSource,
  );
}

/**
 * Registers the data providers shipped with TissUUmaps with the app store
 *
 * Called once during application startup, before any project is loaded.
 */
export function enableBuiltInDataProviders(): void {
  const appStoreState = appStore.getState();

  appStoreState.registerImageDataProvider(
    openSeadragonImageDataSourceType,
    new OpenSeadragonImageDataProvider(),
  );

  appStoreState.registerImageDataProvider(
    omeZarrImageDataSourceType,
    new OMEZarrImageDataProvider(),
  );

  appStoreState.registerLabelsDataProvider(
    omeZarrLabelsDataSourceType,
    new OMEZarrLabelsDataProvider(),
  );

  appStoreState.registerImageDataProvider(
    tiffImageDataSourceType,
    new TIFFImageDataProvider(),
  );

  appStoreState.registerLabelsDataProvider(
    tiffLabelsDataSourceType,
    new TIFFLabelsDataProvider(),
  );

  appStoreState.registerPointsDataProvider(
    tablePointsDataSourceType,
    new TablePointsDataProvider({
      getTableDataProviders: () => appStore.getState().tableDataProviders,
      addTable,
    }),
  );

  appStoreState.registerShapesDataProvider(
    geoJSONShapesDataSourceType,
    new GeoJSONShapesDataProvider({ addTable }),
  );
  appStoreState.registerShapesDataProvider(
    parquetShapesDataSourceType,
    new ParquetShapesDataProvider({ addTable }),
  );

  appStoreState.registerTableDataProvider(
    csvTableDataSourceType,
    new CSVTableDataProvider(),
  );
  appStoreState.registerTableDataProvider(
    parquetTableDataSourceType,
    new ParquetTableDataProvider(),
  );
  appStoreState.registerTableDataProvider(
    hdf5TableDataSourceType,
    new HDF5TableDataProvider(),
  );
  appStoreState.registerTableDataProvider(
    zarrTableDataSourceType,
    new ZarrTableDataProvider(),
  );
  appStoreState.registerTableDataProvider(
    geoJSONTableDataSourceType,
    new GeoJSONTableDataProvider(),
  );
}
