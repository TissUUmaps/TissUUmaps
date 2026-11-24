import { useEffect, useRef, useState } from "react";

import type { TableColumnQueryGroup } from "@tissuumaps/core";

import { SimpleSelect } from "@/components/common/simple-select";
import { TableColumnInput } from "@/components/controls/TableColumnField/TableColumnInput";
import { useTableData } from "@/data/hooks/useData";

import { StyleRow } from "./StyleRow";

/** A leaf query group, i.e. a set of columns that is searched as one */
type Leaf = { name: string; id: string };

/** A row of the picker: a top-level query group and its leaves */
type Row = { name: string; leaves: Leaf[]; hasFrom: boolean };

/**
 * Lays out query groups as rows: each top-level query group is a row titled
 * by its name; a leaf is the row's only set of columns, a node offers its
 * leaves under "from", those of deeper nodes named by their path
 *
 * @param groups - The top-level query groups
 * @returns The rows
 */
function toRows(groups: TableColumnQueryGroup[]): Row[] {
  const getLeaves = (group: TableColumnQueryGroup, prefix: string): Leaf[] =>
    "id" in group
      ? [{ ...group, name: `${prefix}${group.name}` }]
      : group.subgroups.flatMap((child) =>
          getLeaves(
            child,
            "id" in child ? prefix : `${prefix}${child.name} › `,
          ),
        );
  return groups.map((group) =>
    "id" in group
      ? { name: group.name, leaves: [group], hasFrom: false }
      : {
          name: group.name,
          leaves: getLeaves(group, ""),
          hasFrom: true,
        },
  );
}

export type ColorByPickerProps = {
  /** The ID of the object's own table, if it has one */
  tableId: string | null;

  /** The colored column, if any */
  column: string | null;

  /** Colors by a column, or by none */
  onColumnChange: (column: string | null) => void;

  /** The text shown while no column is colored, for tables without groups */
  placeholder: string;

  /** Classes of the control cells of the rows */
  className?: string;
};

/**
 * Picks the column to color by, by its group rather than by its query, as
 * rows of the {@link StyleRow} grid
 *
 * Tables with query groups (see `TableData.loadQueryGroups`) get one row per
 * top-level query group, titled by its name (e.g. "Annotations",
 * "Features"); the table names its query groups, the app only lays them out.
 * A top-level leaf is the row's set of columns; a top-level node gets a "from"
 * menu of its leaves (e.g. `X`, a layer or `raw`), whose first leaf is the
 * default. The row of the colored column is filled, the others empty;
 * picking in any applies right away. Switching "from" keeps the colored
 * entity if the table maps it to the new leaf (see `TableData.mapColumn`);
 * names never decide that. Tables without query groups get one "Column"
 * row.
 */
export function ColorByPicker({
  tableId,
  column,
  onColumnChange,
  placeholder,
  className,
}: ColorByPickerProps) {
  const tableData = useTableData(tableId);
  const [groups, setGroups] = useState<Row[] | null>(null);
  // the description of the colored column, kept until the next one resolves,
  // so that the row does not empty in between
  const [description, setDescription] = useState<{
    column: string;
    groupId: string;
    name: string;
  } | null>(null);
  // the query group chosen under "from" per group (by index), for this session
  // only
  const [selectedGroupIds, setSelectedGroupIds] = useState<
    Partial<Record<number, string>>
  >({});
  const [notice, setNotice] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  useEffect(() => () => abortControllerRef.current?.abort(), []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- no stale groups
    setGroups(null);
    if (tableData === null) {
      return;
    }
    const abortController = new AbortController();
    const { signal } = abortController;
    (tableData.loadQueryGroups?.({ signal }) ?? Promise.resolve([]))
      .then((newGroups) => {
        if (!signal.aborted) {
          setGroups(toRows(newGroups));
        }
      })
      .catch((error: unknown) => {
        if (!signal.aborted) {
          console.error("Failed to load the query groups", error);
          setGroups([]);
        }
      });
    return () => abortController.abort();
  }, [tableData]);

  const hasGroups = groups !== null && groups.length > 0;

  /**
   * @param groupId - The `id` of a leaf query group
   * @returns The index of the row having the leaf, or -1
   */
  const findGroupIndex = (groupId: string | undefined | null) =>
    groups?.findIndex((group) =>
      group.leaves.some((leaf) => leaf.id === groupId),
    ) ?? -1;

  useEffect(() => {
    if (
      tableData?.describeColumn === undefined ||
      column === null ||
      !hasGroups
    ) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- nothing to describe
      setDescription(null);
      return;
    }
    const abortController = new AbortController();
    const { signal } = abortController;
    tableData
      .describeColumn(column, { signal })
      .then((newDescription) => {
        if (!signal.aborted) {
          setDescription(
            newDescription !== null ? { column, ...newDescription } : null,
          );
          // the row of the colored column shows its query group
          const groupIndex =
            groups?.findIndex((group) =>
              group.leaves.some((leaf) => leaf.id === newDescription?.groupId),
            ) ?? -1;
          if (groupIndex >= 0) {
            setSelectedGroupIds((selected) => ({
              ...selected,
              [groupIndex]: undefined,
            }));
          }
        }
      })
      .catch((error: unknown) => {
        if (!signal.aborted) {
          console.error("Failed to describe the column", error);
        }
      });
    return () => abortController.abort();
  }, [tableData, column, hasGroups, groups]);

  // a newly colored column takes back the notice
  // https://react.dev/reference/react/useState#storing-information-from-previous-renders
  const [prevColumn, setPrevColumn] = useState(column);
  if (column !== prevColumn) {
    setPrevColumn(column);
    setNotice(null);
  }

  if (!hasGroups) {
    return (
      <>
        <StyleRow label="Column" className={className}>
          <TableColumnInput
            tableId={tableId}
            value={column}
            onValueChange={onColumnChange}
            placeholder={placeholder}
          />
        </StyleRow>
      </>
    );
  }

  const coloredGroupIndex = findGroupIndex(description?.groupId);

  const startSwitch = () => {
    abortControllerRef.current?.abort();
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    setNotice(null);
    return abortController.signal;
  };

  const switchLeaf = async (groupIndex: number, groupId: string) => {
    const signal = startSwitch();
    const newLeaf = groups[groupIndex]?.leaves.find(
      (leaf) => leaf.id === groupId,
    );
    if (
      newLeaf === undefined ||
      column === null ||
      coloredGroupIndex !== groupIndex ||
      newLeaf.id === description?.groupId ||
      tableData === null
    ) {
      setSelectedGroupIds((selected) => ({
        ...selected,
        [groupIndex]: groupId,
      }));
      return;
    }
    // the same entity, if the table maps it to the new query group; otherwise the
    // row keeps showing the colored column and its query group
    try {
      const mappedColumn =
        (await tableData.mapColumn?.(column, {
          groupId,
          signal,
        })) ?? null;
      if (signal.aborted) {
        return;
      }
      if (mappedColumn !== null) {
        setSelectedGroupIds((selected) => ({
          ...selected,
          [groupIndex]: groupId,
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

  return (
    <>
      {groups.map((group, groupIndex) => {
        const isActive = coloredGroupIndex === groupIndex;
        // the chosen query group, else the colored one, else the first one
        const preferredGroupId =
          selectedGroupIds[groupIndex] ??
          (isActive ? description?.groupId : undefined);
        const leaf =
          group.leaves.find((leaf) => leaf.id === preferredGroupId) ??
          group.leaves[0];
        if (leaf === undefined) {
          return null;
        }
        const isDescriptionStale = description?.column !== column;
        const showsColoredColumn =
          isActive && (leaf.id === description?.groupId || isDescriptionStale);
        return (
          <StyleRow
            key={groupIndex}
            label={group.name}
            isActive={isActive}
            className={className}
          >
            <div className="flex min-w-0 flex-row items-center gap-1">
              <TableColumnInput
                tableId={tableId}
                groupId={leaf.id}
                value={showsColoredColumn ? column : null}
                valueName={showsColoredColumn ? description?.name : undefined}
                onValueChange={onColumnChange}
                placeholder="Search…"
                className="min-w-0 flex-1"
              />
              {group.hasFrom && (
                <>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    from
                  </span>
                  <div className="w-24 shrink-0">
                    <SimpleSelect
                      items={group.leaves}
                      itemLabel={(item) => item.name}
                      itemValue={(item) => item.id}
                      value={leaf.id}
                      onValueChange={(groupId) => {
                        if (groupId !== null) {
                          void switchLeaf(groupIndex, groupId);
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
    </>
  );
}
