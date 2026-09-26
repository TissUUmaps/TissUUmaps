import type { HierarchicalTableDataSource } from "../HierarchicalTableDataSource";

export const zarrTableDataSourceType = "zarr";

export const zarrTableDataSourceDefaults = {};

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface ZarrTableDataSource extends HierarchicalTableDataSource<
  typeof zarrTableDataSourceType
> {}

export type NormalizedZarrTableDataSource = Required<
  Pick<ZarrTableDataSource, keyof typeof zarrTableDataSourceDefaults>
> &
  Omit<ZarrTableDataSource, keyof typeof zarrTableDataSourceDefaults>;
