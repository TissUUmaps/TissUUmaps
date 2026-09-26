import type { HierarchicalTableDataSource } from "../HierarchicalTableDataSource";

export const zarrTableDataSourceType = "zarr";

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface ZarrTableDataSource extends HierarchicalTableDataSource<
  typeof zarrTableDataSourceType
> {}
