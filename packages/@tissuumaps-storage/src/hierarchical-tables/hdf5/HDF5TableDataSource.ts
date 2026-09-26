import type { HierarchicalTableDataSource } from "../HierarchicalTableDataSource";

export const hdf5TableDataSourceType = "hdf5";

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface HDF5TableDataSource extends HierarchicalTableDataSource<
  typeof hdf5TableDataSourceType
> {}
