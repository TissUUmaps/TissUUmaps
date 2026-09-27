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

import { appStore } from "@/stores/app";

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
    new TablePointsDataProvider(),
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
