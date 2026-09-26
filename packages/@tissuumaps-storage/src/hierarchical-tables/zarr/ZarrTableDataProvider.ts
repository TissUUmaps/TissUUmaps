import type { IDArray } from "@tissuumaps/core";

import type { HierarchicalTable } from "../HierarchicalTable";
import { HierarchicalTableDataProviderBase } from "../HierarchicalTableDataProviderBase";
import { HierarchicalTableReader } from "../HierarchicalTableReader";
import { ZarrTableData } from "./ZarrTableData";
import type { ZarrTableDataSource } from "./ZarrTableDataSource";
import { openZarrStore } from "./openZarrStore";

/**
 * Reads tables from Zarr stores, including the AnnData tables of SpatialData
 * stores
 *
 * Zarr reads are asynchronous fetches, so no Web Worker is needed.
 */
export class ZarrTableDataProvider extends HierarchicalTableDataProviderBase<
  ZarrTableDataSource,
  ZarrTableData
> {
  readonly name = "Zarr";

  protected override async openHierarchicalTable(
    normalizedSource: string,
    options: {
      signal?: AbortSignal;
      workspace: FileSystemDirectoryHandle | null;
    },
  ): Promise<HierarchicalTable> {
    const { signal, workspace } = options;
    signal?.throwIfAborted();
    const store = await openZarrStore(normalizedSource, { signal, workspace });
    return await HierarchicalTableReader.open(store, { signal });
  }

  protected override createTableData(
    table: HierarchicalTable,
    numRows: number,
    ids: IDArray | undefined,
    names: string[] | undefined,
  ): ZarrTableData {
    return new ZarrTableData(table, numRows, ids, names);
  }
}
