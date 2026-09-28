import {
  ArrayUtils,
  type DataProviderLoadOptions,
  type IDArray,
  type TableDataProvider,
} from "@tissuumaps/core";

import type { HierarchicalTable } from "./HierarchicalTable";
import type { HierarchicalTableDataBase } from "./HierarchicalTableDataBase";
import type { HierarchicalTableDataSource } from "./HierarchicalTableDataSource";

/**
 * Base class of table data providers reading hierarchical containers
 *
 * Handles the form and the ID and name columns. A container format
 * normalizes its data source as every provider does, opens a
 * {@link HierarchicalTable} for the normalized source, see
 * {@link HierarchicalTableDataProviderBase.openHierarchicalTable}, and wraps
 * it in its data class, see
 * {@link HierarchicalTableDataProviderBase.createTableData}.
 *
 * @typeParam TDataSource - The data source type of the container format
 * @typeParam TData - The data class of the container format
 * @typeParam TNormalizedDataSource - The normalized data source type of the
 * container format
 */
export abstract class HierarchicalTableDataProviderBase<
  TDataSource extends HierarchicalTableDataSource,
  TData extends HierarchicalTableDataBase,
  TNormalizedDataSource extends TDataSource = TDataSource,
> implements TableDataProvider<TDataSource, TData, TNormalizedDataSource> {
  abstract readonly name: string;

  readonly schema = {
    type: "object",
    properties: {
      source: {
        type: "string",
      },
      idColumn: {
        type: "string",
      },
      nameColumn: {
        type: "string",
      },
    },
    required: ["source"],
  };

  readonly uischema = HierarchicalTableDataProviderBase.createUISchema(false);

  /**
   * Creates the UI schema of the data source form
   *
   * @param sourceIsDirectory - Whether the source is a directory, such as a
   * Zarr store, rather than a file
   * @returns The UI schema
   */
  protected static createUISchema(sourceIsDirectory: boolean) {
    return {
      type: "VerticalLayout",
      elements: [
        {
          type: "Control",
          scope: "#/properties/source",
          label: "Source",
          ...(sourceIsDirectory && { options: { directory: true } }),
        },
        {
          type: "Control",
          scope: "#/properties/idColumn",
          label: "ID Column",
        },
        {
          type: "Control",
          scope: "#/properties/nameColumn",
          label: "Name Column",
        },
      ],
    };
  }

  abstract normalize(
    dataSource: TDataSource,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
  ): TNormalizedDataSource;

  async load(
    normalizedDataSource: TNormalizedDataSource,
    options?: DataProviderLoadOptions,
  ): Promise<TData> {
    const { signal, workspace = null } = options ?? {};
    signal?.throwIfAborted();

    const table = await this.openHierarchicalTable(
      normalizedDataSource.source,
      {
        signal,
        workspace,
      },
    );
    try {
      const { idColumn, nameColumn } = normalizedDataSource;
      const [idData, nameData] = await Promise.all([
        idColumn !== undefined
          ? table.readColumn(idColumn, { signal })
          : undefined,
        nameColumn !== undefined
          ? table.readColumn(nameColumn, { signal })
          : undefined,
      ]);
      const ids =
        idData !== undefined ? ArrayUtils.toIDArray(idData) : undefined;
      const names =
        nameData !== undefined ? Array.from(nameData, String) : undefined;
      const numRows = ids?.length ?? names?.length ?? table.numRows;
      if (
        ids !== undefined &&
        names !== undefined &&
        names.length !== numRows
      ) {
        throw new Error(
          `ID column "${idColumn}" and name column "${nameColumn}" have different lengths.`,
        );
      }
      return this.createTableData(table, numRows, ids, names);
    } catch (error) {
      table.close();
      throw error;
    }
  }

  /**
   * Opens the hierarchical table a normalized source points to
   *
   * @param normalizedSource - The normalized source of the data source
   * @param options - `signal` aborts the load; `workspace` is the directory
   * handle of the open workspace, required for workspace-relative sources
   * @returns The open table; closed by the returned
   * {@link HierarchicalTableDataBase}
   */
  protected abstract openHierarchicalTable(
    normalizedSource: string,
    options: {
      signal?: AbortSignal;
      workspace: FileSystemDirectoryHandle | null;
    },
  ): Promise<HierarchicalTable>;

  /**
   * Wraps an open table in the data class of the container format
   *
   * @param table - The open table, which the data owns and closes
   * @param numRows - The number of rows
   * @param ids - The row IDs, `undefined` for sequential IDs
   * @param names - The row names, if any
   * @returns The data, which owns the table
   */
  protected abstract createTableData(
    table: HierarchicalTable,
    numRows: number,
    ids: IDArray | undefined,
    names: string[] | undefined,
  ): TData;
}
