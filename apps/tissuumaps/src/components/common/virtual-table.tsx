import {
  type ColumnDef,
  type ColumnVisibilityState,
  type RowData,
  type SortingState,
  columnResizingFeature,
  columnSizingFeature,
  columnVisibilityFeature,
  functionalUpdate,
  metaHelper,
  rowSortingFeature,
  tableFeatures,
  useTable,
} from "@tanstack/react-table";
import { ArrowDownIcon, ArrowUpIcon } from "lucide-react";
import { type ReactNode, useMemo } from "react";

import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useCompressedRowVirtualizer } from "@/hooks/useCompressedRowVirtualizer";
import { cn } from "@/lib/utils";

/** Column metadata of a virtual table */
export type VirtualTableColumnMeta = {
  /** Whether the caller shows the column until the user picks the columns */
  isShownByDefault?: boolean;
};

/**
 * The table features of a virtual table
 *
 * A table only has the APIs of the features registered here.
 */
const features = tableFeatures({
  // every cell takes the width of its column
  columnSizingFeature,
  // the user resizes a column by dragging the edge of its header
  columnResizingFeature,
  // columns can be hidden
  columnVisibilityFeature,
  // the headers toggle the sort state; the caller sorts the rows, as the table
  // only holds the visible ones
  rowSortingFeature,
  // the type of `columnDef.meta`
  columnMeta: metaHelper<VirtualTableColumnMeta>(),
});

/**
 * Renders a header or cell template without turning it into a component
 *
 * `flexRender` would do the latter, so that a column definition rebuilt on a
 * state change remounts every header and cell, losing the focus and the open
 * popovers of the interactive ones.
 */
function renderTemplate<TContext>(
  template: string | ((context: TContext) => unknown) | undefined,
  context: TContext,
): ReactNode {
  if (typeof template === "function") {
    // the column definition types a template's return value as `any`
    return template(context) as ReactNode;
  }
  return template !== undefined ? (
    <span className="truncate px-1">{template}</span>
  ) : undefined;
}

/**
 * A column definition of a virtual table
 *
 * Header and cell templates are called as plain functions while the table
 * renders, so they must not call hooks; they may return elements of
 * components that do.
 */
export type VirtualTableColumnDef<TRowData extends RowData> = ColumnDef<
  typeof features,
  TRowData
>;

export type VirtualTableProps<TRowData extends RowData> = {
  rowCount: number;
  /** Returns the rows within `[startIndex, endIndex)` */
  getRows: (startIndex: number, endIndex: number) => TRowData[];
  getRowId: (row: TRowData) => string;
  columnDefs: VirtualTableColumnDef<TRowData>[];
  rowHeight: number;
  height: number;
  /**
   * How many rows are rendered beyond each end of the visible range
   *
   * Rows outside of it are not rendered, so scrolling reveals blank space until
   * the next render; the overscan covers a scroll of up to this many rows.
   */
  overscan?: number;
  rowClassName?: (row: TRowData) => string | undefined;

  /** Shown in the top right corner of the header, whatever the scroll */
  headerAction?: ReactNode;

  /**
   * The column the rows are sorted by, if any, which the caller sorts them by;
   * the columns are not sortable without
   */
  sorting?: SortingState;

  onSortingChange?: (sorting: SortingState) => void;

  /** Which columns are shown, by column ID; columns are shown by default */
  columnVisibility?: ColumnVisibilityState;

  className?: string;
};

export function VirtualTable<TRowData extends RowData>({
  rowCount,
  getRows,
  getRowId,
  columnDefs,
  rowHeight,
  height,
  overscan = 2,
  rowClassName,
  headerAction,
  sorting,
  onSortingChange,
  columnVisibility,
  className,
}: VirtualTableProps<TRowData>) {
  const {
    containerRef,
    headerRef,
    firstIndex,
    lastIndex,
    layoutRowsHeight,
    rowShift,
  } = useCompressedRowVirtualizer<HTMLTableSectionElement>(
    rowCount,
    rowHeight,
    height,
    overscan,
  );

  // only the rows within the visible range are materialized, so that the cost
  // of the table does not depend on the number of rows
  const rows = useMemo(
    () => getRows(firstIndex, lastIndex),
    [getRows, firstIndex, lastIndex],
  );

  const table = useTable<typeof features, TRowData>({
    features,
    data: rows,
    columns: columnDefs,
    getRowId,
    columnResizeMode: "onChange",
    state: {
      ...(sorting !== undefined && { sorting }),
      ...(columnVisibility !== undefined && { columnVisibility }),
    },
    onSortingChange: (updater) => {
      if (sorting !== undefined) {
        onSortingChange?.(functionalUpdate(updater, sorting));
      }
    },
    manualSorting: true,
    enableSorting: sorting !== undefined,
    enableMultiSort: false,
    sortDescFirst: false,
  });

  const tableRows = table.getRowModel().rows;

  return (
    <div
      ref={containerRef}
      className={cn("overflow-auto relative", className)}
      style={{ height: `${height}px` }}
    >
      <Table className="grid w-max min-w-full text-xs">
        <TableHeader
          ref={headerRef}
          className="grid sticky top-0 z-10 bg-muted text-xs text-muted-foreground"
        >
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow
              key={headerGroup.id}
              className="flex w-full hover:bg-transparent"
            >
              {headerGroup.headers.map((header) => (
                <TableHead
                  key={header.id}
                  className="relative flex h-8 items-center p-0 text-inherit"
                  style={{ width: `${header.getSize()}px` }}
                  colSpan={header.colSpan}
                >
                  {!header.isPlaceholder &&
                    (header.column.getCanSort() ? (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-full w-full justify-start rounded-none px-0 pr-2 text-xs font-medium text-inherit hover:bg-transparent hover:text-foreground"
                        title={`Sort by ${header.column.id}`}
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        {renderTemplate(
                          header.column.columnDef.header,
                          header.getContext(),
                        )}
                        {header.column.getIsSorted() === "desc" && (
                          <ArrowDownIcon className="size-3.5" />
                        )}
                        {header.column.getIsSorted() === "asc" && (
                          <ArrowUpIcon className="size-3.5" />
                        )}
                      </Button>
                    ) : (
                      renderTemplate(
                        header.column.columnDef.header,
                        header.getContext(),
                      )
                    ))}
                  {header.column.getCanResize() && (
                    <div
                      className={cn(
                        "absolute top-0 right-0 h-full w-1.5 cursor-col-resize touch-none select-none border-r hover:bg-primary/30",
                        header.column.getIsResizing() && "bg-primary/50",
                      )}
                      onMouseDown={header.getResizeHandler()}
                      onTouchStart={header.getResizeHandler()}
                      onDoubleClick={() => header.column.resetSize()}
                    />
                  )}
                </TableHead>
              ))}
              {headerAction !== undefined && (
                <TableHead className="sticky right-0 ml-auto flex h-8 w-9 shrink-0 items-center justify-center bg-muted p-0">
                  {headerAction}
                </TableHead>
              )}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody
          className="grid relative"
          style={{ height: `${layoutRowsHeight}px` }}
        >
          {tableRows.map((row, index) => (
            <TableRow
              key={row.id}
              className={cn(
                "flex absolute w-full border-0 items-center overflow-hidden",
                rowClassName?.(row.original),
              )}
              style={{
                height: `${rowHeight}px`,
                transform: `translateY(${(firstIndex + index) * rowHeight - rowShift}px)`,
              }}
            >
              {row.getVisibleCells().map((cell) => (
                <TableCell
                  key={cell.id}
                  className="flex items-center px-1"
                  style={{ width: `${cell.column.getSize()}px` }}
                >
                  {renderTemplate(
                    cell.column.columnDef.cell,
                    cell.getContext(),
                  )}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
