import type { PointsDataSource } from "@tissuumaps/core";

export const tablePointsDataSourceType = "table";

export const tablePointsDataSourceDefaults = {
  x: "x",
  y: "y",
};

export interface TablePointsDataSource extends PointsDataSource<
  typeof tablePointsDataSourceType
> {
  /**
   * A table file to add as a new table, which the data source then references
   * instead (see `TablePointsDataProvider.prepareDataSource`); never persisted
   */
  source?: string;
  table?: string;
  x?: string;
  y?: string;
}

export type NormalizedTablePointsDataSource = Required<
  Pick<TablePointsDataSource, keyof typeof tablePointsDataSourceDefaults>
> &
  Omit<TablePointsDataSource, keyof typeof tablePointsDataSourceDefaults>;
