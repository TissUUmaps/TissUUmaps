import { type ReactNode, useEffect, useRef, useState } from "react";

import type { TableColumnSection } from "@tissuumaps/core";

import { SimpleSelect } from "@/components/common/simple-select";
import { TableColumnInput } from "@/components/controls/TableColumnField/TableColumnInput";
import { findColumnByName } from "@/components/controls/TableColumnField/findColumnByName";
import { Button } from "@/components/ui/button";
import { useTableData } from "@/data/hooks/useData";

import { SettingsRow } from "./SettingsRow";
import type { ColumnLocation } from "./useColumnLocation";

/** A row: a top-level section and its leaves */
type Row = {
  id: string;
  name: string;
  leaves: Pick<TableColumnSection, "id" | "name">[];
  hasFrom: boolean;
};

/**
 * Lays out sections as rows: each top-level section is a row titled
 * by its name; a leaf is the row's only set of columns, any other section
 * offers its leaves under "from", those of deeper levels named by their path
 *
 * @param sections - The top-level sections
 * @returns The rows
 */
function toRows(sections: TableColumnSection[]): Row[] {
  const getLeaves = (
    section: TableColumnSection,
    prefix: string,
  ): Row["leaves"] =>
    section.subsections === undefined
      ? [{ id: section.id, name: `${prefix}${section.name}` }]
      : section.subsections.flatMap((subsection) =>
          getLeaves(
            subsection,
            subsection.subsections === undefined
              ? prefix
              : `${prefix}${subsection.name} › `,
          ),
        );
  return sections.map((section) => ({
    id: section.id,
    name: section.name,
    leaves: getLeaves(section, ""),
    hasFrom: section.subsections !== undefined,
  }));
}

export type ColumnRowsProps = {
  /** The ID of the object's own table, if it has one */
  tableId: string | null;

  /** The followed column, if any */
  column: string | null;

  /** Where the followed column is (see `useColumnLocation`) */
  location: ColumnLocation | null;

  /** Picks the column to follow, or none; resolves to whether it applied */
  onColumnChange: (column: string | null) => Promise<boolean>;

  /**
   * Picks the column that `loadColumn` resolves to, if any; resolves to
   * whether it applied
   */
  onLoadedColumnChange: (
    loadColumn: (options: { signal: AbortSignal }) => Promise<string | null>,
  ) => Promise<boolean>;

  /**
   * The text shown while no column is followed, in the first row for tables
   * with sections
   */
  placeholder: string;

  /**
   * Controls shown before the input of the followed column, e.g. its type
   */
  columnControls?: ReactNode;

  /** Classes of the control cells of the rows */
  className?: string;
};

/**
 * Picks the column to follow, by its section rather than by its query, as
 * rows of the {@link SettingsRow} grid
 *
 * Tables with sections (see `TableData.getColumnSections`) get one row per
 * top-level section, titled by its name (e.g. "Observations",
 * "Features"); the table names its sections, the app only lays them out.
 * A top-level leaf is the row's set of columns; any other top-level
 * section gets a "from" menu of its leaves (e.g. `X`, a layer or `raw`), whose
 * first leaf is the default. The row of the followed column is filled, the
 * others empty; picking in any applies right away. Switching "from" keeps the
 * followed entity if the table maps it to the new leaf (see
 * `TableData.mapColumn`). A text matching no column of a leaf is looked up in
 * the other leaves of its row and offered there. A message shows only while
 * the column it was made for is followed. A column in no section fills no
 * row. Tables without sections get one "Column" row.
 */
export function ColumnRows({
  tableId,
  column,
  location,
  onColumnChange,
  onLoadedColumnChange,
  placeholder,
  columnControls,
  className,
}: ColumnRowsProps) {
  const tableData = useTableData(tableId);
  const rows = toRows(tableData?.getColumnSections?.() ?? []);
  // the leaf chosen under "from", by row ID, for this session only
  const [selectedLeafIds, setSelectedLeafIds] = useState<
    Partial<Record<string, string>>
  >({});
  // the messages, each with the followed column it was made for
  const [notice, setNotice] = useState<{
    text: string;
    forColumn: string | null;
  } | null>(null);
  const [elsewhere, setElsewhere] = useState<{
    leaf: Row["leaves"][number];
    column: string;
    name: string;
    forColumn: string | null;
  } | null>(null);
  const searchAbortControllerRef = useRef<AbortController | null>(null);
  useEffect(() => () => searchAbortControllerRef.current?.abort(), []);

  // the leaf of the followed column, while in a section
  const sectionLocation = location?.kind === "section" ? location : null;

  // the row of the followed column shows its leaf
  // https://react.dev/reference/react/useState#storing-information-from-previous-renders
  const [prevLocation, setPrevLocation] = useState(location);
  if (location !== prevLocation) {
    setPrevLocation(location);
    const row = rows.find((row) =>
      row.leaves.some((leaf) => leaf.id === sectionLocation?.sectionId),
    );
    if (row !== undefined) {
      setSelectedLeafIds((selected) => ({ ...selected, [row.id]: undefined }));
    }
  }

  // a new column to follow takes back the messages, so that none shows again
  // when its column is followed again
  // https://react.dev/reference/react/useState#storing-information-from-previous-renders
  const [prevColumn, setPrevColumn] = useState(column);
  if (column !== prevColumn) {
    setPrevColumn(column);
    setNotice(null);
    setElsewhere(null);
  }

  if (rows.length === 0) {
    return (
      <SettingsRow label="Column" className={className}>
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
      </SettingsRow>
    );
  }

  const followedRow = rows.find((row) =>
    row.leaves.some((leaf) => leaf.id === sectionLocation?.sectionId),
  );

  // a switch or a search aborts the search before it and takes back its
  // messages
  const resetSearch = () => {
    searchAbortControllerRef.current?.abort();
    searchAbortControllerRef.current = null;
    setNotice(null);
    setElsewhere(null);
  };

  const switchLeaf = (row: Row, sectionId: string) => {
    resetSearch();
    const newLeaf = row.leaves.find((leaf) => leaf.id === sectionId);
    // bound, as the check below does not narrow inside the callback
    const mapColumn = tableData?.mapColumn?.bind(tableData);
    if (
      newLeaf === undefined ||
      column === null ||
      followedRow?.id !== row.id ||
      newLeaf.id === sectionLocation?.sectionId ||
      mapColumn === undefined
    ) {
      setSelectedLeafIds((selected) => ({
        ...selected,
        [row.id]: sectionId,
      }));
      return;
    }
    // the same entity, if the table maps it to the new leaf; otherwise the
    // row keeps showing the followed column and its leaf
    void onLoadedColumnChange(async ({ signal }) => {
      const mappedColumn = await mapColumn(column, { sectionId, signal });
      if (mappedColumn === null && !signal.aborted) {
        setNotice({ text: `No match in ${newLeaf.name}`, forColumn: column });
      }
      return mappedColumn;
    });
  };

  // a text matching no column of a leaf is searched for in the other leaves of
  // its row, by name only, and offered rather than applied
  const searchElsewhere = async (text: string, row: Row, sectionId: string) => {
    resetSearch();
    const abortController = new AbortController();
    searchAbortControllerRef.current = abortController;
    const { signal } = abortController;
    if (tableData === null) {
      return;
    }
    try {
      for (const leaf of row.leaves) {
        if (leaf.id === sectionId) {
          continue;
        }
        const found = await findColumnByName(tableData, leaf.id, text, {
          signal,
        });
        if (signal.aborted) {
          return;
        }
        if (found !== null) {
          setElsewhere({ leaf, ...found, forColumn: column });
          return;
        }
      }
    } catch (error) {
      if (!signal.aborted) {
        console.error("Failed to search the other sections", error);
      }
    }
  };

  return (
    <>
      {rows.map((row) => {
        const isActive = followedRow?.id === row.id;
        // the chosen leaf, else the followed one, else the first one
        const preferredLeafId =
          selectedLeafIds[row.id] ??
          (isActive ? sectionLocation?.sectionId : undefined);
        const leaf =
          row.leaves.find((leaf) => leaf.id === preferredLeafId) ??
          row.leaves[0];
        if (leaf === undefined) {
          return null;
        }
        const isLocationStale = location?.column !== column;
        const showsFollowedColumn =
          isActive &&
          (leaf.id === sectionLocation?.sectionId || isLocationStale);
        return (
          <SettingsRow
            key={row.id}
            label={row.name}
            isActive={isActive}
            className={className}
          >
            <div className="flex min-w-0 flex-row items-center gap-1">
              {showsFollowedColumn && columnControls}
              <TableColumnInput
                tableId={tableId}
                sectionId={leaf.id}
                value={showsFollowedColumn ? column : null}
                valueName={
                  showsFollowedColumn ? sectionLocation?.name : undefined
                }
                onValueChange={onColumnChange}
                onNoMatch={(text) => void searchElsewhere(text, row, leaf.id)}
                // the first row tells what the object looks like without a
                // column
                placeholder={
                  column === null && row === rows[0] ? placeholder : "Search…"
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
                      itemLabel={(item) => item.name}
                      itemValue={(item) => item.id}
                      value={leaf.id}
                      onValueChange={(sectionId) => {
                        if (sectionId !== null) {
                          switchLeaf(row, sectionId);
                        }
                      }}
                      variant="compact"
                      triggerLabel="Search in"
                    />
                  </div>
                </>
              )}
            </div>
          </SettingsRow>
        );
      })}
      {notice !== null && notice.forColumn === column && (
        <SettingsRow className={className}>
          <span className="text-xs text-muted-foreground">{notice.text}</span>
        </SettingsRow>
      )}
      {elsewhere !== null && elsewhere.forColumn === column && (
        <SettingsRow className={className}>
          <span className="text-xs text-muted-foreground">
            {elsewhere.name} is in {elsewhere.leaf.name} —{" "}
            <Button
              variant="link"
              size="xs"
              className="h-auto p-0 text-xs"
              onClick={() => {
                setElsewhere(null);
                void onColumnChange(elsewhere.column);
              }}
            >
              switch?
            </Button>
          </span>
        </SettingsRow>
      )}
    </>
  );
}
