import type {
  ColumnSort,
  ColumnVisibilityState,
  SortingState,
} from "@tanstack/react-table";
import { useCallback, useMemo, useState } from "react";

import type { HighlightedItemGroup } from "@tissuumaps/core";

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
import { InactiveCell } from "./cells/InactiveCell";

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

/** How the group table shows and toggles the visibility of a group */
export type GroupVisibility = {
  isVisible: (group: string) => boolean;

  /** Whether the eye buttons are grayed out, as toggling them changes the property source */
  isInactive: boolean;

  onVisibleChange: (groups: string[], visible: boolean) => void;
};

/** The group rows are listed by name until a column is sorted by */
const defaultSorting: ColumnSort = { id: "group", desc: false };

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

  groupByColumn: string;
  groupCounts: Map<string, number> | null;
  groupVisibility?: GroupVisibility;
  groupColumnDefs?: GroupAnnotationsTableColumnDef[];
};

export function GroupAnnotationsTable({
  height,
  rowHeight,
  annotatedObject,
  groupByColumn,
  groupCounts,
  groupVisibility,
  groupColumnDefs,
}: GroupAnnotationsTableProps) {
  const [sorting, setSorting] = useState<SortingState>([defaultSorting]);
  const [shownColumns, setShownColumns] = useState<ColumnVisibilityState>({});

  // the columns the rows can sort by, which do not depend on the rows
  const sortableColumnDefs = useMemo(
    (): GroupAnnotationsTableColumnDef[] => [
      {
        id: "group",
        header: groupByColumn,
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
    [groupByColumn, groupColumnDefs],
  );

  const columnVisibility = useMemo(() => {
    const columnVisibility: ColumnVisibilityState = {};
    for (const columnDef of sortableColumnDefs) {
      columnVisibility[columnDef.id!] =
        columnDef.meta?.isShownByDefault ?? true;
    }
    return { ...columnVisibility, ...shownColumns };
  }, [sortableColumnDefs, shownColumns]);

  const [sortedColumn = defaultSorting] = sorting;
  const activeSorting =
    columnVisibility[sortedColumn.id] === true ? sortedColumn : defaultSorting;

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
      (columnDef) => columnDef.id === activeSorting.id,
    );
    const getSortValue =
      sortedColumnDef !== undefined && "accessorFn" in sortedColumnDef
        ? sortedColumnDef.accessorFn
        : undefined;
    const order = activeSorting.desc ? -1 : 1;
    groupRows.sort(
      (a, b) =>
        order *
        ((getSortValue !== undefined
          ? compareSortValues(getSortValue(a, 0), getSortValue(b, 0))
          : 0) || textCollator.compare(a.group, b.group)),
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
        <InactiveCell isInactive={isInactive}>
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
        </InactiveCell>
      ),
      cell: ({ row }) => (
        <InactiveCell isInactive={isInactive}>
          <GroupVisibilityCell
            visible={isVisible(row.original.group)}
            onVisibleChange={(visible) => {
              onVisibleChange([row.original.group], visible);
            }}
            itemGroup={{
              annotatedObject,
              column: groupByColumn,
              group: row.original.group,
            }}
          />
        </InactiveCell>
      ),
    };
    return [visibleColumnDef, ...sortableColumnDefs];
  }, [
    groupVisibility,
    groupRows,
    annotatedObject,
    groupByColumn,
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
      sorting={[activeSorting]}
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
