import type { TableDataSource } from "@tissuumaps/core";

export const geoJSONTableDataSourceType = "geojson";

export const geoJSONTableDataSourceDefaults = {};

/**
 * The feature properties of a GeoJSON file, as a table whose rows are keyed
 * by the IDs of the shapes the file holds (see `GeoJSONShapesDataSource`)
 */
export interface GeoJSONTableDataSource extends TableDataSource<
  typeof geoJSONTableDataSourceType
> {
  source: string;
  /** The property holding the feature IDs, as for the shapes of the file */
  idProperty?: string;
}

export type NormalizedGeoJSONTableDataSource = Required<
  Pick<GeoJSONTableDataSource, keyof typeof geoJSONTableDataSourceDefaults>
> &
  Omit<GeoJSONTableDataSource, keyof typeof geoJSONTableDataSourceDefaults>;
