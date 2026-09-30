import { useCallback, useMemo } from "react";

import type { IDArray, ItemsData } from "@tissuumaps/core";

import {
  VirtualTable,
  type VirtualTableColumnDef,
} from "@/components/common/virtual-table";
import { useTableData } from "@/hooks/useData";

export type ItemAnnotationsTableRowData = {
  id: number | string;
  name?: string;
  annotated?: boolean;
};

type ItemAnnotationsTableColumnDef =
  VirtualTableColumnDef<ItemAnnotationsTableRowData>;

export type ItemAnnotationsTableProps = {
  data?: ItemsData;
  height: number;
  rowHeight: number;
  tableId: string | null;
};

export function ItemAnnotationsTable({
  data,
  height,
  rowHeight,
  tableId,
}: ItemAnnotationsTableProps) {
  const tableData = useTableData(tableId);

  // the ids and the per-index accessors the rows are built from, so that only
  // the rows within the visible range have to be materialized
  const { ids, getName, isAnnotated } = useMemo(() => {
    let ids: IDArray = [];
    let getName: ((index: number) => string | undefined) | undefined;
    let isAnnotated: ((id: number | string) => boolean) | undefined;
    if (data !== undefined) {
      ids = data.getIds();
      if (tableId !== null) {
        // the selected table governs the names; while it is still loading there
        // are none yet, rather than the object's own names, which would show a
        // different column for a moment and then be replaced
        if (tableData !== null) {
          const tableIds = tableData.getIds();
          const tableNames = tableData.getNames?.();
          // table-backed items hand out the table's own ids array, so every
          // item has a row in the table, aligned by index, and no lookup
          // structure is needed
          if (tableIds === ids) {
            isAnnotated = () => true;
            if (tableNames !== undefined) {
              getName = (index) => tableNames[index];
            }
          } else {
            const tableRowsById = new Map<number | string, number>();
            for (let i = 0; i < tableIds.length; i++) {
              tableRowsById.set(tableIds[i]!, i);
            }
            isAnnotated = (id) => tableRowsById.has(id);
            if (tableNames !== undefined) {
              getName = (index) => {
                const row = tableRowsById.get(ids[index]!);
                return row !== undefined ? tableNames[row] : undefined;
              };
            }
          }
        }
      } else {
        const names = data.getNames?.();
        if (names !== undefined) {
          getName = (index) => names[index];
        }
      }
    } else if (tableData !== null) {
      ids = tableData.getIds();
      const names = tableData.getNames?.();
      if (names !== undefined) {
        getName = (index) => names[index];
      }
    }
    return { ids, getName, isAnnotated };
  }, [data, tableId, tableData]);

  const getRows = useCallback(
    (startIndex: number, endIndex: number) => {
      const rows: ItemAnnotationsTableRowData[] = [];
      for (let index = startIndex; index < endIndex; index++) {
        const id = ids[index]!;
        rows.push({
          id,
          name: getName?.(index),
          annotated: isAnnotated?.(id),
        });
      }
      return rows;
    },
    [ids, getName, isAnnotated],
  );

  const columnDefs = useMemo(() => {
    const columnDefs: ItemAnnotationsTableColumnDef[] = [
      { id: "id", header: "ID", accessorKey: "id" },
    ];
    if (getName !== undefined) {
      columnDefs.push({ id: "name", header: "Name", accessorKey: "name" });
    }
    return columnDefs;
  }, [getName]);

  return (
    <VirtualTable
      rowCount={ids.length}
      getRows={getRows}
      getRowId={(row) => String(row.id)}
      columnDefs={columnDefs}
      rowHeight={rowHeight}
      height={height}
      rowClassName={(row) =>
        row.annotated === false ? "text-muted-foreground" : undefined
      }
    />
  );
}
