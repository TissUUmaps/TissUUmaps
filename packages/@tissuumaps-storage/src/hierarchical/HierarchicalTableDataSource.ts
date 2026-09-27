import type { TableDataSource } from "@tissuumaps/core";

/**
 * A data source read by a {@link HierarchicalTableDataProviderBase}
 *
 * @typeParam TType - The data source type of the container format
 */
export interface HierarchicalTableDataSource<
  TType extends string = string,
> extends TableDataSource<TType> {
  source: string;
  /** Column query of the row IDs; sequential IDs are used if omitted */
  idColumn?: string;
  /** Column query of the row names */
  nameColumn?: string;
}
