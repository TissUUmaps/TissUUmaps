import { SourceUtils, createTable } from "@tissuumaps/core";
import {
  CSVTableDataProvider,
  GeoJSONShapesDataProvider,
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

import { createDataObjectID } from "@/data/io/dataObject";
import { appStore } from "@/stores/app";
import { projectStore } from "@/stores/project";

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
      addTable: async (dataSource, options) => {
        const { signal } = options ?? {};
        signal?.throwIfAborted();
        const { tableDataProviders, workspace } = appStore.getState();
        const tableDataProvider = tableDataProviders.get(dataSource.type);
        if (tableDataProvider === undefined) {
          throw new Error(
            `No table data provider for type ${dataSource.type}.`,
          );
        }
        const projectSource = projectStore.getState().source;
        const normalizedSource =
          dataSource.source !== undefined
            ? SourceUtils.normalizeSource(
                dataSource.source,
                workspace,
                projectSource,
              )
            : undefined;
        let name: string | undefined;
        if (normalizedSource !== undefined) {
          if (tableDataProvider.readName !== undefined) {
            try {
              name = await tableDataProvider.readName(
                normalizedSource,
                workspace,
                { signal },
              );
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
        // created right before adding, after the last await, so that the ID
        // is still unique when the table is added
        const id = createDataObjectID(
          normalizedSource,
          projectStore.getState().tables.map((table) => table.id),
        );
        projectStore.getState().addTable(
          createTable({
            id,
            name: name ?? "Untitled",
            dataSource: preparedDataSource,
          }),
        );
        return id;
      },
    }),
  );

  appStoreState.registerShapesDataProvider(
    geoJSONShapesDataSourceType,
    new GeoJSONShapesDataProvider(),
  );
  appStoreState.registerShapesDataProvider(
    parquetShapesDataSourceType,
    new ParquetShapesDataProvider(),
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
}
