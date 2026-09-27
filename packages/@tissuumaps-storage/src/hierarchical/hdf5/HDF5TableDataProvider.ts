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
