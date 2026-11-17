import { type ReactNode, useEffect, useRef, useState } from "react";

import type { TableColumnQueryGroup } from "@tissuumaps/core";

import { SimpleSelect } from "@/components/common/simple-select";
import { TableColumnInput } from "@/components/controls/TableColumnField/TableColumnInput";
import { formatUnavailability } from "@/components/controls/TableColumnField/formatUnavailability";
import { Button } from "@/components/ui/button";
import { useTableData } from "@/data/hooks/useData";

import { StyleRow } from "./StyleRow";
import { findColumnByText } from "./findColumnByText";

/** A row of the picker: a top-level query group and its leaves */
type Row = {
  id: string;
  name: string;
  leaves: Pick<TableColumnQueryGroup, "id" | "name" | "unavailable">[];
  hasFrom: boolean;
};

/**
 * Lays out query groups as rows: each top-level query group is a row titled
 * by its name; a leaf is the row's only set of columns, any other query group
 * offers its leaves under "from", those of deeper levels named by their path
 *
 * @param groups - The top-level query groups
 * @returns The rows
 */
function toRows(groups: TableColumnQueryGroup[]): Row[] {
  const getLeaves = (
    group: TableColumnQueryGroup,
    prefix: string,
  ): Row["leaves"] =>
    group.subgroups === undefined
      ? [
          {
            id: group.id,
            name: `${prefix}${group.name}`,
            ...(group.unavailable !== undefined && {
              unavailable: group.unavailable,
            }),
          },
        ]
      : group.subgroups.flatMap((subgroup) =>
          getLeaves(
            subgroup,
            subgroup.subgroups === undefined
              ? prefix
              : `${prefix}${subgroup.name} › `,
          ),
        );
  return groups.map((group) => ({
    id: group.id,
    name: group.name,
    leaves: getLeaves(group, ""),
    hasFrom: group.subgroups !== undefined,
  }));
}

export type StyleByPickerProps = {
  /** The ID of the object's own table, if it has one */
  tableId: string | null;

  /** The column styled by, if any */
  column: string | null;

  /** Styles by a column, or by none */
  onColumnChange: (column: string | null) => void;

  /**
   * The text shown while no column is styled by, in the first row for tables
   * with query groups
   */
  placeholder: string;

  /**
   * Controls shown before the input of the column styled by, e.g. its type
   */
  columnControls?: ReactNode;

  /** Classes of the control cells of the rows */
  className?: string;
};

/**
 * Picks the column to style by, by its query group rather than by its query,
 * as rows of the {@link StyleRow} grid
 *
 * Tables with query groups (see `TableData.getQueryGroups`) get one row per
 * top-level query group, titled by its name (e.g. "Annotations",
 * "Features"); the table names its query groups, the app only lays them out.
 * A top-level leaf is the row's set of columns; any other top-level query
 * group gets a "from" menu of its leaves (e.g. `X`, a layer or `raw`), whose
 * first available leaf is the default. The row of the column styled by is
 * filled, the others empty; picking in any applies right away. Switching
 * "from" keeps the entity styled by if the table maps it to the new leaf (see
 * `TableData.mapColumn`). A text matching no column of a leaf is looked up in
 * the other readable leaves of its row and offered there. Tables without query
 * groups get one "Column" row.
 */
export function StyleByPicker({
  tableId,
  column,
  onColumnChange,
  placeholder,
  columnControls,
  className,
}: StyleByPickerProps) {
  const tableData = useTableData(tableId);
  const rows = toRows(tableData?.getQueryGroups?.() ?? []);
  // the description of the column styled by, kept until the next one resolves,
  // so that the row does not empty in between
  const [description, setDescription] = useState<{
    column: string;
    queryGroupId: string;
    name: string;
  } | null>(null);
  // the leaf chosen under "from", by row ID, for this session only
  const [selectedLeafIds, setSelectedLeafIds] = useState<
    Partial<Record<string, string>>
  >({});
  const [notice, setNotice] = useState<string | null>(null);
  const [elsewhere, setElsewhere] = useState<{
    rowId: string;
    leaf: Row["leaves"][number];
    column: string;
    name: string;
  } | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  useEffect(() => () => abortControllerRef.current?.abort(), []);

  // nothing to describe
  if (
    (tableData?.describeColumn === undefined || column === null) &&
    description !== null
  ) {
    setDescription(null);
  }

  useEffect(() => {
    if (tableData?.describeColumn === undefined || column === null) {
      return;
    }
    const abortController = new AbortController();
    const { signal } = abortController;
    tableData
      .describeColumn(column, { signal })
      .then((newDescription) => {
        if (signal.aborted) {
          return;
        }
        setDescription(
          newDescription !== null ? { column, ...newDescription } : null,
        );
        // the row of the column styled by shows its leaf
        const row = toRows(tableData.getQueryGroups?.() ?? []).find((row) =>
          row.leaves.some((leaf) => leaf.id === newDescription?.queryGroupId),
        );
        if (row !== undefined) {
          setSelectedLeafIds((selected) => ({
            ...selected,
            [row.id]: undefined,
          }));
        }
      })
      .catch((error: unknown) => {
        if (!signal.aborted) {
          console.error("Failed to describe the column", error);
        }
      });
    return () => abortController.abort();
  }, [tableData, column]);

  // a new column to style by takes back the messages
  // https://react.dev/reference/react/useState#storing-information-from-previous-renders
  const [prevColumn, setPrevColumn] = useState(column);
  if (column !== prevColumn) {
    setPrevColumn(column);
    setNotice(null);
    setElsewhere(null);
  }

  if (rows.length === 0) {
    return (
      <StyleRow label="Column" className={className}>
        <div className="flex min-w-0 flex-row items-center gap-1">
          {columnControls}
          <TableColumnInput
            tableId={tableId}
            value={column}
            onValueChange={onColumnChange}
            placeholder={placeholder}
            className="min-w-0 flex-1"
          />
        </div>
      </StyleRow>
    );
  }

  const styledRow = rows.find((row) =>
    row.leaves.some((leaf) => leaf.id === description?.queryGroupId),
  );

  const startSwitch = () => {
    abortControllerRef.current?.abort();
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    setNotice(null);
    setElsewhere(null);
    return abortController.signal;
  };

  const switchLeaf = async (row: Row, queryGroupId: string) => {
    const signal = startSwitch();
    const newLeaf = row.leaves.find((leaf) => leaf.id === queryGroupId);
    if (
      newLeaf === undefined ||
      column === null ||
      styledRow?.id !== row.id ||
      newLeaf.id === description?.queryGroupId ||
      tableData?.mapColumn === undefined
    ) {
      setSelectedLeafIds((selected) => ({
        ...selected,
        [row.id]: queryGroupId,
      }));
      return;
    }
    // the same entity, if the table maps it to the new leaf; otherwise the
    // row keeps showing the column styled by and its leaf
    try {
      const mappedColumn = await tableData.mapColumn(column, {
        queryGroupId,
        signal,
      });
      if (signal.aborted) {
        return;
      }
      if (mappedColumn !== null) {
        setSelectedLeafIds((selected) => ({
          ...selected,
          [row.id]: queryGroupId,
        }));
        onColumnChange(mappedColumn);
      } else {
        setNotice(`No match in ${newLeaf.name}`);
      }
    } catch (error) {
      if (!signal.aborted) {
        console.error("Failed to find the column in the query group", error);
      }
    }
  };

  // a text matching no column of a leaf is searched for in the other leaves of
  // its row, by exact name or query only, and offered rather than applied
  const searchElsewhere = async (
    text: string,
    row: Row,
    queryGroupId: string,
  ) => {
    const signal = startSwitch();
    if (tableData === null) {
      return;
    }
    try {
      for (const leaf of row.leaves) {
        if (leaf.id === queryGroupId || leaf.unavailable !== undefined) {
          continue;
        }
        const found = await findColumnByText(tableData, leaf.id, text, {
          signal,
        });
        if (signal.aborted) {
          return;
        }
        if (found !== null) {
          setElsewhere({ rowId: row.id, leaf, ...found });
          return;
        }
      }
    } catch (error) {
      if (!signal.aborted) {
        console.error("Failed to search the other query groups", error);
      }
    }
  };

  return (
    <>
      {rows.map((row) => {
        const isActive = styledRow?.id === row.id;
        // the chosen leaf, else the styled one, else the first available one
        const preferredLeafId =
          selectedLeafIds[row.id] ??
          (isActive ? description?.queryGroupId : undefined);
        const leaf =
          row.leaves.find((leaf) => leaf.id === preferredLeafId) ??
          row.leaves.find((leaf) => leaf.unavailable === undefined) ??
          row.leaves[0];
        if (leaf === undefined) {
          return null;
        }
        const isDescriptionStale = description?.column !== column;
        const showsStyledColumn =
          isActive &&
          (leaf.id === description?.queryGroupId || isDescriptionStale);
        return (
          <StyleRow
            key={row.id}
            label={row.name}
            isActive={isActive}
            className={className}
          >
            <div className="flex min-w-0 flex-row items-center gap-1">
              {showsStyledColumn && columnControls}
              <TableColumnInput
                tableId={tableId}
                queryGroupId={leaf.id}
                value={showsStyledColumn ? column : null}
                valueName={showsStyledColumn ? description?.name : undefined}
                onValueChange={onColumnChange}
                onNoMatch={(text) => void searchElsewhere(text, row, leaf.id)}
                // the first row tells what the object looks like without a
                // column
                placeholder={
                  leaf.unavailable !== undefined
                    ? formatUnavailability(leaf.unavailable)
                    : column === null && row === rows[0]
                      ? placeholder
                      : "Search…"
                }
                className="min-w-0 flex-1"
              />
              {row.hasFrom && (
                <>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    from
                  </span>
                  <div className="w-24 shrink-0">
                    <SimpleSelect
                      items={row.leaves}
                      itemLabel={(item) =>
                        item.unavailable !== undefined ? (
                          <span className="flex flex-col">
                            {item.name}
                            <span className="text-xs text-muted-foreground">
                              {formatUnavailability(item.unavailable)}
                            </span>
                          </span>
                        ) : (
                          item.name
                        )
                      }
                      itemValue={(item) => item.id}
                      isItemDisabled={(item) => item.unavailable !== undefined}
                      value={leaf.id}
                      onValueChange={(queryGroupId) => {
                        if (queryGroupId !== null) {
                          void switchLeaf(row, queryGroupId);
                        }
                      }}
                      variant="compact"
                      triggerLabel="Search in"
                    />
                  </div>
                </>
              )}
            </div>
          </StyleRow>
        );
      })}
      {notice !== null && (
        <StyleRow className={className}>
          <span className="text-xs text-muted-foreground">{notice}</span>
        </StyleRow>
      )}
      {elsewhere !== null && (
        <StyleRow className={className}>
          <span className="text-xs text-muted-foreground">
            {elsewhere.name} is in {elsewhere.leaf.name} —{" "}
            <Button
              variant="link"
              size="xs"
              className="h-auto p-0 text-xs"
              onClick={() => {
                const { rowId, leaf, column } = elsewhere;
                setElsewhere(null);
                setSelectedLeafIds((selected) => ({
                  ...selected,
                  [rowId]: leaf.id,
                }));
                onColumnChange(column);
              }}
            >
              switch?
            </Button>
          </span>
        </StyleRow>
      )}
    </>
  );
}
