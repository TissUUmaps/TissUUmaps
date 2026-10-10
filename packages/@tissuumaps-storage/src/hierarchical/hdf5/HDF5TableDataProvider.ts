import { type IDArray, SourceUtils } from "@tissuumaps/core";

import type { HierarchicalTable } from "../HierarchicalTable";
import { HierarchicalTableDataProviderBase } from "../HierarchicalTableDataProviderBase";
import { HierarchicalTableWorkerClient } from "../workers/HierarchicalTableWorkerClient";
import { HDF5TableData } from "./HDF5TableData";
import {
  type HDF5TableDataSource,
  type NormalizedHDF5TableDataSource,
  hdf5TableDataSourceDefaults,
} from "./HDF5TableDataSource";
import HDF5WorkerScript from "./hdf5.worker?worker&inline";

/**
 * Reads tables from HDF5 files, including AnnData `.h5ad` files
 *
 * h5wasm reads synchronously, so the file is opened in a Web Worker that
 * lives as long as the table.
 */
export class HDF5TableDataProvider extends HierarchicalTableDataProviderBase<
  HDF5TableDataSource,
  HDF5TableData,
  NormalizedHDF5TableDataSource
> {
  /** The file extensions of the sources this data provider supports */
  private static readonly _extensions = new Set([".h5", ".hdf5", ".h5ad"]);

  readonly name = "HDF5";

  override normalize(
    dataSource: HDF5TableDataSource,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
  ): NormalizedHDF5TableDataSource {
    return {
      ...hdf5TableDataSourceDefaults,
      ...dataSource,
      source: SourceUtils.normalizeSource(
        dataSource.source,
        workspace,
        projectSource,
      ),
    };
  }

  /**
   * Returns whether a source has an HDF5 file extension (see
   * {@link HDF5TableDataProvider._extensions})
   *
   * @param normalizedSource - The normalized source to check
   * @returns A promise that resolves to whether the source is supported
   */
  supports(normalizedSource: string): Promise<boolean> {
    return Promise.resolve(
      HDF5TableDataProvider._extensions.has(
        SourceUtils.getExtension(normalizedSource),
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
    const source = await SourceUtils.openSourceFile(
      normalizedSource,
      workspace,
      { signal },
    );
    return await HierarchicalTableWorkerClient.open(
      new HDF5WorkerScript(),
      source.url !== undefined ? source.url : source.file,
      { signal },
    );
  }

  protected override createTableData(
    table: HierarchicalTable,
    numRows: number,
    ids: IDArray | undefined,
    names: string[] | undefined,
  ): HDF5TableData {
    return new HDF5TableData(table, numRows, ids, names);
  }
}
