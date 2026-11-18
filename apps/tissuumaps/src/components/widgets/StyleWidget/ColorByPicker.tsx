import { useEffect, useRef, useState } from "react";

import type { TableColumnQueryGroup } from "@tissuumaps/core";

import { SimpleSelect } from "@/components/common/simple-select";
import { TableColumnInput } from "@/components/controls/TableColumnField/TableColumnInput";
import { useTableData } from "@/data/hooks/useData";

import { StyleRow } from "./StyleRow";

/** A row of the picker: a top-level query group and its leaves */
type Row = {
  id: string;
  name: string;
  leaves: Pick<TableColumnQueryGroup, "id" | "name">[];
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
      ? [{ id: group.id, name: `${prefix}${group.name}` }]
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
 * Picks the column to color by, by its query group rather than by its query,
 * as rows of the {@link StyleRow} grid
 *
 * Tables with query groups (see `TableData.getQueryGroups`) get one row per
 * top-level query group, titled by its name (e.g. "Annotations",
 * "Features"); the table names its query groups, the app only lays them out.
 * A top-level leaf is the row's set of columns; any other top-level query
 * group gets a "from" menu of its leaves (e.g. `X`, a layer or `raw`), whose
 * first leaf is the default. The row of the colored column is filled, the
 * others empty; picking in any applies right away. Switching "from" keeps the
 * colored entity if the table maps it to the new leaf (see
 * `TableData.mapColumn`). Tables without query groups get one "Column" row.
 */
export function ColorByPicker({
  tableId,
  column,
  onColumnChange,
  placeholder,
  className,
}: ColorByPickerProps) {
  const tableData = useTableData(tableId);
  const rows = toRows(tableData?.getQueryGroups?.() ?? []);
  // the description of the colored column, kept until the next one resolves,
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
  const abortControllerRef = useRef<AbortController | null>(null);
  useEffect(() => () => abortControllerRef.current?.abort(), []);

  useEffect(() => {
    if (tableData?.describeColumn === undefined || column === null) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- nothing to describe
      setDescription(null);
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
        // the row of the colored column shows its leaf
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

  // a newly colored column takes back the notice
  // https://react.dev/reference/react/useState#storing-information-from-previous-renders
  const [prevColumn, setPrevColumn] = useState(column);
  if (column !== prevColumn) {
    setPrevColumn(column);
    setNotice(null);
  }

  if (rows.length === 0) {
    return (
      <StyleRow label="Column" className={className}>
        <TableColumnInput
          tableId={tableId}
          value={column}
          onValueChange={onColumnChange}
          placeholder={placeholder}
        />
      </StyleRow>
    );
  }

  const coloredRow = rows.find((row) =>
    row.leaves.some((leaf) => leaf.id === description?.queryGroupId),
  );

  const switchLeaf = async (row: Row, queryGroupId: string) => {
    abortControllerRef.current?.abort();
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    const { signal } = abortController;
    setNotice(null);
    const newLeaf = row.leaves.find((leaf) => leaf.id === queryGroupId);
    if (
      newLeaf === undefined ||
      column === null ||
      coloredRow?.id !== row.id ||
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
    // row keeps showing the colored column and its leaf
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

  return (
    <>
      {rows.map((row) => {
        const isActive = coloredRow?.id === row.id;
        // the chosen leaf, else the colored one, else the first one
        const preferredLeafId =
          selectedLeafIds[row.id] ??
          (isActive ? description?.queryGroupId : undefined);
        const leaf =
          row.leaves.find((leaf) => leaf.id === preferredLeafId) ??
          row.leaves[0];
        if (leaf === undefined) {
          return null;
        }
        const isDescriptionStale = description?.column !== column;
        const showsColoredColumn =
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
              <TableColumnInput
                tableId={tableId}
                queryGroupId={leaf.id}
                value={showsColoredColumn ? column : null}
                valueName={showsColoredColumn ? description?.name : undefined}
                onValueChange={onColumnChange}
                placeholder="Search…"
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
                      itemLabel={(item) => item.name}
                      itemValue={(item) => item.id}
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
    </>
  );
}
