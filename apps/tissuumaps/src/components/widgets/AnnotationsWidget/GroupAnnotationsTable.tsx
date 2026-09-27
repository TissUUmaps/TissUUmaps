import type {
  ColumnVisibilityState,
  SortingState,
} from "@tanstack/react-table";
import { useCallback, useMemo, useState } from "react";

import type { HighlightedItemGroup, TableColumnRef } from "@tissuumaps/core";

import {
  VirtualTable,
  type VirtualTableColumnDef,
} from "@/components/common/virtual-table";
import { Checkbox } from "@/components/ui/checkbox";

import {
  GroupColumnPicker,
  type GroupColumnPickerProps,
} from "./GroupColumnPicker";
import { GroupVisibilityCell } from "./cells/GroupVisibilityCell";
import { MutedCell } from "./cells/MutedCell";
import type { GroupVisibility } from "./useGroupVisibility";

export type GroupAnnotationsTableRowData = {
  group: string;
  count: number;
};

/**
 * A column of the group table
 *
 * The rows sort by the column's `accessorFn`; a column without one is not
 * sortable.
 */
export type GroupAnnotationsTableColumnDef =
  VirtualTableColumnDef<GroupAnnotationsTableRowData>;

/** Compares text with its numbers by value, so that `2` sorts before `10` */
const textCollator = new Intl.Collator(undefined, { numeric: true });

/**
 * Compares two sort values, numbers numerically and anything else as text
 *
 * @param a - The first sort value
 * @param b - The second sort value
 * @returns A negative number if `a` sorts first, a positive one if `b` does
 */
function compareSortValues(a: unknown, b: unknown): number {
  if (typeof a === "number" && typeof b === "number") {
    return a - b;
  }
  return textCollator.compare(String(a), String(b));
}

export type GroupAnnotationsTableProps = {
  height: number;
  rowHeight: number;

  /** The object whose items are grouped, which the eye buttons highlight */
  annotatedObject: HighlightedItemGroup["annotatedObject"];

  groupBy: TableColumnRef;
  groupCounts: Map<string, number> | null;
  groupVisibility?: GroupVisibility;
  groupColumnDefs?: GroupAnnotationsTableColumnDef[];
};

export function GroupAnnotationsTable({
  height,
  rowHeight,
  annotatedObject,
  groupBy,
  groupCounts,
  groupVisibility,
  groupColumnDefs,
}: GroupAnnotationsTableProps) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [shownColumns, setShownColumns] = useState<ColumnVisibilityState>({});

  // the columns the rows can sort by, which do not depend on the rows
  const sortableColumnDefs = useMemo(
    (): GroupAnnotationsTableColumnDef[] => [
      {
        id: "group",
        header: groupBy.column,
        accessorFn: (row) => row.group,
        enableHiding: false,
        size: 110,
        cell: ({ row }) => (
          <span className="truncate">{row.original.group}</span>
        ),
      },
      {
        id: "count",
        header: "Count",
        accessorFn: (row) => row.count,
        size: 70,
        cell: ({ row }) => row.original.count.toLocaleString(),
      },
      ...(groupColumnDefs ?? []),
    ],
    [groupBy, groupColumnDefs],
  );

  const columnVisibility = useMemo(() => {
    const columnVisibility: ColumnVisibilityState = {};
    for (const columnDef of sortableColumnDefs) {
      columnVisibility[columnDef.id!] =
        columnDef.meta?.isShownByDefault ?? true;
    }
    return { ...columnVisibility, ...shownColumns };
  }, [sortableColumnDefs, shownColumns]);

  const [sortedColumn] = sorting;
  const activeSorting =
    sortedColumn !== undefined && columnVisibility[sortedColumn.id] === true
      ? sortedColumn
      : undefined;

  const pickableColumns: GroupColumnPickerProps["columns"] = [
    ...(groupVisibility !== undefined
      ? [{ id: "visible", header: "Visibility" }]
      : []),
    ...sortableColumnDefs
      .filter((columnDef) => columnDef.enableHiding !== false)
      .map((columnDef) => ({
        id: columnDef.id!,
        header:
          typeof columnDef.header === "string"
            ? columnDef.header
            : columnDef.id!,
      })),
  ].map((column) => ({
    ...column,
    isShown: columnVisibility[column.id] ?? true,
  }));

  // one row is materialized per group, not per item: a column whose values are
  // all distinct belongs in the item table, which windows its rows
  const groupRows = useMemo(() => {
    if (groupCounts === null) {
      return [];
    }
    const groupRows = Array.from(groupCounts, ([group, count]) => ({
      group,
      count,
    }));
    const sortedColumnDef = sortableColumnDefs.find(
      (columnDef) => columnDef.id === activeSorting?.id,
    );
    if (
      activeSorting === undefined ||
      sortedColumnDef === undefined ||
      !("accessorFn" in sortedColumnDef)
    ) {
      return groupRows;
    }
    const getSortValue = sortedColumnDef.accessorFn;
    const order = activeSorting.desc ? -1 : 1;
    // the sort is stable, so that rows with the same value keep the table order
    groupRows.sort(
      (a, b) =>
        order * compareSortValues(getSortValue(a, 0), getSortValue(b, 0)),
    );
    return groupRows;
  }, [groupCounts, activeSorting, sortableColumnDefs]);

  const getRows = useCallback(
    (startIndex: number, endIndex: number): GroupAnnotationsTableRowData[] =>
      groupRows.slice(startIndex, endIndex),
    [groupRows],
  );

  const columnDefs = useMemo(() => {
    if (groupVisibility === undefined) {
      return sortableColumnDefs;
    }
    const { isVisible, isInactive, onVisibleChange } = groupVisibility;
    const groups = groupRows.map((groupRow) => groupRow.group);
    const numVisibleGroups = groups.filter(isVisible).length;
    const visibleColumnDef: GroupAnnotationsTableColumnDef = {
      id: "visible",
      size: 36,
      enableResizing: false,
      header: () => (
        <MutedCell isMuted={isInactive}>
          <span className="flex h-6 w-full items-center px-1">
            <Checkbox
              checked={groups.length > 0 && numVisibleGroups === groups.length}
              indeterminate={
                numVisibleGroups > 0 && numVisibleGroups < groups.length
              }
              onCheckedChange={(checked) => {
                onVisibleChange(groups, checked);
              }}
              title="Show listed groups"
              aria-label="Show listed groups"
            />
          </span>
        </MutedCell>
      ),
      cell: ({ row }) => (
        <MutedCell isMuted={isInactive}>
          <GroupVisibilityCell
            visible={isVisible(row.original.group)}
            onVisibleChange={(visible) => {
              onVisibleChange([row.original.group], visible);
            }}
            itemGroup={{
              annotatedObject,
              groupBy,
              group: row.original.group,
            }}
          />
        </MutedCell>
      ),
    };
    return [visibleColumnDef, ...sortableColumnDefs];
  }, [
    groupVisibility,
    groupRows,
    annotatedObject,
    groupBy,
    sortableColumnDefs,
  ]);

  return (
    <VirtualTable
      rowCount={groupRows.length}
      getRows={getRows}
      getRowId={(row) => row.group}
      columnDefs={columnDefs}
      rowHeight={rowHeight}
      height={height}
      sorting={activeSorting !== undefined ? [activeSorting] : []}
      onSortingChange={setSorting}
      columnVisibility={columnVisibility}
      headerAction={
        <GroupColumnPicker
          columns={pickableColumns}
          onShownChange={(id, isShown) => {
            setShownColumns((shownColumns) => ({
              ...shownColumns,
              [id]: isShown,
            }));
          }}
        />
      }
    />
  );
}
