import type { Popover as PopoverPrimitive } from "@base-ui/react/popover";
import { SearchIcon } from "lucide-react";
import { useMemo, useRef, useState } from "react";

import type {
  HighlightedItemGroup,
  ItemsData,
  TableColumnRef,
} from "@tissuumaps/core";

import { Fieldset, FieldsetLegend } from "@/components/common/fieldset";
import { TableColumnField } from "@/components/controls/TableColumnField";
import type { TableColumnInputStatus } from "@/components/controls/TableColumnField/TableColumnInput";
import { formatTableColumn } from "@/components/controls/TableColumnField/formatTableColumn";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

import {
  GroupAnnotationsTable,
  type GroupAnnotationsTableColumnDef,
} from "./GroupAnnotationsTable";
import { ItemAnnotationsTable } from "./ItemAnnotationsTable";
import type { GroupVisibility } from "./useGroupVisibility";

// rows have a fixed height, so that the visible range follows from the scroll
// offset alone; cells must fit within it
const tableRowHeight = 28;

/** Groups beyond which the group table shows a message instead of its rows */
const maxGroupCount = 200_000;

/**
 * Keeps the groups whose name contains a query, ignoring case
 *
 * @param groupCounts - The row count of every group
 * @param nameQuery - The text to look for, or `""` to keep every group
 * @returns The row count of every group whose name contains the query
 */
function filterGroupsByName(
  groupCounts: Map<string, number>,
  nameQuery: string,
): Map<string, number> {
  const lowerCaseNameQuery = nameQuery.toLowerCase();
  if (lowerCaseNameQuery === "") {
    return groupCounts;
  }
  return new Map(
    Array.from(groupCounts).filter(([group]) =>
      group.toLowerCase().includes(lowerCaseNameQuery),
    ),
  );
}

export type AnnotationsWidgetProps = {
  data?: ItemsData;
  tableHeight: number;
  tableId: string | null;

  /**
   * The object whose items are annotated, which the eye buttons highlight; its
   * identity must be stable, as the highlight follows it
   */
  annotatedObject: HighlightedItemGroup["annotatedObject"];

  selectedGroupBy: TableColumnRef | null;
  onSelectedGroupByChange: (column: TableColumnRef | null) => void;
  groupCounts: Map<string, number> | null;
  groupVisibility?: GroupVisibility;
  groupColumnDefs?: GroupAnnotationsTableColumnDef[];
  className?: string;
};

export function AnnotationsWidget({
  data,
  tableHeight,
  tableId,
  annotatedObject,
  selectedGroupBy,
  onSelectedGroupByChange,
  groupCounts,
  groupVisibility,
  groupColumnDefs,
  className,
}: AnnotationsWidgetProps) {
  const tables = useProjectStore((state) => state.tables);
  const [nameQuery, setNameQuery] = useState("");

  // the popover stays open while the group-by column is being resolved or
  // failed to resolve, so that its outcome is shown; Escape always closes it
  const groupByActionsRef = useRef<PopoverPrimitive.Root.Actions>(null);
  const groupByStatusRef = useRef<TableColumnInputStatus>("idle");
  const isGroupByCloseRequestedRef = useRef(false);

  // the cap counts every group, as a new map is filled with every group
  const hasTooManyGroups =
    groupCounts !== null && groupCounts.size > maxGroupCount;

  const filteredGroupCounts = useMemo(
    () =>
      groupCounts !== null && !hasTooManyGroups
        ? filterGroupsByName(groupCounts, nameQuery)
        : groupCounts,
    [groupCounts, hasTooManyGroups, nameQuery],
  );

  const isGroupVisible = groupVisibility?.isVisible;

  // the shown items are those of the groups that pass the filter and are
  // visible
  const itemCounts = useMemo(() => {
    if (groupCounts === null || filteredGroupCounts === null) {
      return null;
    }
    let total = 0;
    for (const count of groupCounts.values()) {
      total += count;
    }
    let shown = 0;
    for (const [group, count] of filteredGroupCounts) {
      if (isGroupVisible === undefined || isGroupVisible(group)) {
        shown += count;
      }
    }
    return { shown, total };
  }, [groupCounts, filteredGroupCounts, isGroupVisible]);

  const isFilterDisabled = selectedGroupBy === null || hasTooManyGroups;

  return (
    <Fieldset
      className={cn("flex flex-col gap-y-2 border rounded-md p-2", className)}
    >
      <FieldsetLegend className="flex flex-row items-center gap-x-1 font-medium text-foreground">
        <span className="shrink-0">
          {selectedGroupBy !== null ? "Items by" : "Items"}
        </span>
        <Popover
          actionsRef={groupByActionsRef}
          onOpenChange={(open, eventDetails) => {
            isGroupByCloseRequestedRef.current = false;
            if (
              !open &&
              eventDetails.reason !== "escape-key" &&
              groupByStatusRef.current !== "idle"
            ) {
              eventDetails.cancel();
              isGroupByCloseRequestedRef.current =
                groupByStatusRef.current === "pending";
            }
          }}
        >
          <PopoverTrigger
            disabled={tableId === null}
            title="Click to group by another column"
            className={cn(
              "min-w-0 truncate underline decoration-muted-foreground decoration-dotted underline-offset-4 data-disabled:opacity-50",
              selectedGroupBy === null && "font-normal text-muted-foreground",
            )}
          >
            {selectedGroupBy !== null
              ? formatTableColumn(selectedGroupBy, tables)
              : "not grouped"}
          </PopoverTrigger>
          <PopoverContent align="start" className="w-96 p-3">
            <TableColumnField
              label="Group by column"
              tableId={tableId}
              value={selectedGroupBy}
              onValueChange={onSelectedGroupByChange}
              onStatusChange={(status) => {
                groupByStatusRef.current = status;
                if (
                  status !== "pending" &&
                  isGroupByCloseRequestedRef.current
                ) {
                  isGroupByCloseRequestedRef.current = false;
                  if (status === "idle") {
                    groupByActionsRef.current?.close();
                  }
                }
              }}
            />
          </PopoverContent>
        </Popover>
        {itemCounts !== null && (
          <span
            className="ml-auto shrink-0 text-xs font-normal text-muted-foreground"
            title="Items in the shown groups / items in the table"
          >
            {itemCounts.shown.toLocaleString()} /{" "}
            {itemCounts.total.toLocaleString()}
          </span>
        )}
      </FieldsetLegend>
      <InputGroup className="h-7" data-disabled={isFilterDisabled}>
        <InputGroupAddon>
          <SearchIcon />
        </InputGroupAddon>
        <InputGroupInput
          className="h-full text-xs md:text-xs"
          aria-label="Filter groups"
          placeholder="Filter groups"
          disabled={isFilterDisabled}
          value={nameQuery}
          onChange={(event) => setNameQuery(event.target.value)}
        />
      </InputGroup>
      {tableId !== null && selectedGroupBy !== null ? (
        hasTooManyGroups ? (
          <span className="text-xs text-muted-foreground">
            {groupCounts.size.toLocaleString()} groups: too many to list. Group
            by a column with at most {maxGroupCount.toLocaleString()} values.
          </span>
        ) : (
          <GroupAnnotationsTable
            height={tableHeight}
            rowHeight={tableRowHeight}
            annotatedObject={annotatedObject}
            groupBy={selectedGroupBy}
            groupCounts={filteredGroupCounts}
            groupVisibility={groupVisibility}
            groupColumnDefs={groupColumnDefs}
          />
        )
      ) : (
        <ItemAnnotationsTable
          data={data}
          height={tableHeight}
          rowHeight={tableRowHeight}
          tableId={tableId}
        />
      )}
    </Fieldset>
  );
}
