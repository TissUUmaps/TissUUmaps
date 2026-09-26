import { useMemo, useState } from "react";

import type { HighlightedItemGroup, ItemsData } from "@tissuumaps/core";

import { Field, FieldLabel } from "@/components/common/field";
import { Fieldset, FieldsetLegend } from "@/components/common/fieldset";
import { Input } from "@/components/ui/input";
import { TableColumnInput } from "@/components/widgets/TableColumnInput";
import { cn } from "@/lib/utils";

import {
  GroupAnnotationsTable,
  type GroupAnnotationsTableColumnDef,
} from "./GroupAnnotationsTable";
import { ItemAnnotationsTable } from "./ItemAnnotationsTable";
import type { GroupVisibility } from "./useGroupVisibility";

// rows have a fixed height, so that the visible range follows from the scroll
// offset alone; cells must fit within it
const tableRowHeight = 28;

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

  selectedGroupByColumn: string | null;
  onSelectedGroupByColumnChange: (column: string | null) => void;
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
  selectedGroupByColumn,
  onSelectedGroupByColumnChange,
  groupCounts,
  groupVisibility,
  groupColumnDefs,
  className,
}: AnnotationsWidgetProps) {
  const [nameQuery, setNameQuery] = useState("");

  const filteredGroupCounts = useMemo(
    () =>
      groupCounts !== null ? filterGroupsByName(groupCounts, nameQuery) : null,
    [groupCounts, nameQuery],
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

  return (
    <Fieldset
      className={cn("flex flex-col gap-y-2 border rounded-md p-2", className)}
    >
      <FieldsetLegend className="font-medium text-foreground">
        Annotations
        {itemCounts !== null && (
          <span
            className="ml-1 text-xs font-normal text-muted-foreground"
            title="Items in the shown groups / items in the table"
          >
            ({itemCounts.shown.toLocaleString()} /{" "}
            {itemCounts.total.toLocaleString()})
          </span>
        )}
      </FieldsetLegend>
      <Field disabled={tableId === null}>
        <FieldLabel>Group by</FieldLabel>
        <TableColumnInput
          tableId={tableId}
          value={selectedGroupByColumn}
          onValueChange={onSelectedGroupByColumnChange}
        />
      </Field>
      <Field disabled={selectedGroupByColumn === null}>
        <FieldLabel>Filter groups</FieldLabel>
        <Input
          value={nameQuery}
          onChange={(event) => setNameQuery(event.target.value)}
        />
      </Field>
      {tableId !== null && selectedGroupByColumn !== null ? (
        <GroupAnnotationsTable
          height={tableHeight}
          rowHeight={tableRowHeight}
          annotatedObject={annotatedObject}
          groupByColumn={selectedGroupByColumn}
          groupCounts={filteredGroupCounts}
          groupVisibility={groupVisibility}
          groupColumnDefs={groupColumnDefs}
        />
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
