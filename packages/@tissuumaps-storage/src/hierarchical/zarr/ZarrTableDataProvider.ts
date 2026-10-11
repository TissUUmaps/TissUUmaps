import { type IDArray, SourceUtils } from "@tissuumaps/core";

import type { HierarchicalTable } from "../HierarchicalTable";
import { HierarchicalTableDataProviderBase } from "../HierarchicalTableDataProviderBase";
import { HierarchicalTableReader } from "../HierarchicalTableReader";
import { ZarrTableData } from "./ZarrTableData";
import {
  type NormalizedZarrTableDataSource,
  type ZarrTableDataSource,
  zarrTableDataSourceDefaults,
} from "./ZarrTableDataSource";
import { openZarr } from "./openZarr";

/**
 * Reads tables from Zarr stores, including the AnnData tables of SpatialData
 * stores
 *
 * Zarr reads are asynchronous fetches, so no Web Worker is needed.
 */
export class ZarrTableDataProvider extends HierarchicalTableDataProviderBase<
  ZarrTableDataSource,
  ZarrTableData,
  NormalizedZarrTableDataSource
> {
  /** The extension of Zarr stores, of which any segment of a source may be */
  private static readonly _storeExtension = ".zarr";

  readonly name = "Zarr";

  override normalize(
    dataSource: ZarrTableDataSource,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
  ): NormalizedZarrTableDataSource {
    return {
      ...zarrTableDataSourceDefaults,
      ...dataSource,
      source: SourceUtils.normalizeSource(
        dataSource.source,
        workspace,
        projectSource,
      ),
    };
  }

  /**
   * Returns whether a source lies within a Zarr store, judged by whether any
   * of its segments has the store extension (see
   * {@link ZarrTableDataProvider._storeExtension})
   *
   * A source may point at a group below the store root, such as the
   * `tables/<name>` of a SpatialData store.
   *
   * @param normalizedSource - The normalized source to check
   * @returns A promise that resolves to whether the source is supported
   */
  supports(normalizedSource: string): Promise<boolean> {
    return Promise.resolve(
      SourceUtils.getPathSegments(normalizedSource).some((segment) =>
        segment.toLowerCase().endsWith(ZarrTableDataProvider._storeExtension),
      ),
    );
  }

  protected override async openHierarchicalTable(
    normalizedSource: string,
    options: {
      signal?: AbortSignal;
      workspace: FileSystemDirectoryHandle | null;
    },
  ): Promise<HierarchicalTable> {
    const { signal, workspace } = options;
    signal?.throwIfAborted();
    const store = await openZarr(normalizedSource, { signal, workspace });
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
