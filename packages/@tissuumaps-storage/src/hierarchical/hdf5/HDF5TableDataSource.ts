import type { HierarchicalTableDataSource } from "../HierarchicalTableDataSource";

export const hdf5TableDataSourceType = "hdf5";

export const hdf5TableDataSourceDefaults = {};

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface HDF5TableDataSource extends HierarchicalTableDataSource<
  typeof hdf5TableDataSourceType
> {}

export type NormalizedHDF5TableDataSource = Required<
  Pick<HDF5TableDataSource, keyof typeof hdf5TableDataSourceDefaults>
> &
  Omit<HDF5TableDataSource, keyof typeof hdf5TableDataSourceDefaults>;
