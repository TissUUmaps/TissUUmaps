import { useMemo, useState } from "react";

import { type Config, ConfigUtils } from "@tissuumaps/core";

import { useItemGroupCounts } from "@/hooks/useItemGroupCounts";

import { getDominantGroupByColumn } from "./getDominantGroupByColumn";

/** A setting of an annotated object and its configuration */
export type GroupTableSetting = {
  /** The settings category of the setting */
  category: string;

  /** The configuration of the setting */
  config: Config<string>;
};

/** The state of the group table: the table column it groups by, and its groups */
export type GroupTableState = {
  /** The name of the annotated object, which new maps are named after */
  objectName: string;

  /** The table column that the group table groups by, if any */
  column: string | null;

  setColumn: (column: string | null) => void;

  /** The row count of every group, or `null` while there is none */
  groupCounts: Map<string, number> | null;
};

/**
 * Chooses the table column that the group table groups by, and counts its
 * groups
 *
 * The group table follows the column that the open settings category groups
 * by. Otherwise it starts from the column that most settings group by, and
 * keeps the column picked by the user until the table changes.
 *
 * @param objectName - The name of the annotated object
 * @param tableId - The ID of the annotated object's table
 * @param settings - The settings of the annotated object, in order of priority
 * @param activeSettingsCategory - The open settings category, if any
 * @returns The state of the group table, shared by its columns
 */
export function useGroupTable(
  objectName: string,
  tableId: string | null,
  settings: GroupTableSetting[],
  activeSettingsCategory: string | null,
): GroupTableState {
  const activeConfig = settings.find(
    (setting) => setting.category === activeSettingsCategory,
  )?.config;
  const activeGroupByColumn =
    tableId !== null && activeConfig !== undefined
      ? (ConfigUtils.getGroupByColumn(activeConfig) ?? null)
      : null;

  const defaultGroupByColumn =
    tableId !== null
      ? (activeGroupByColumn ??
        getDominantGroupByColumn(settings.map((setting) => setting.config)))
      : null;

  const [column, setColumn] = useState(defaultGroupByColumn);

  // https://react.dev/reference/react/useState#storing-information-from-previous-renders
  const [prevTableId, setPrevTableId] = useState(tableId);
  if (tableId !== prevTableId) {
    setPrevTableId(tableId);
    setColumn(defaultGroupByColumn);
  }

  const [prevActiveGroupByColumn, setPrevActiveGroupByColumn] =
    useState(activeGroupByColumn);
  if (activeGroupByColumn !== prevActiveGroupByColumn) {
    setPrevActiveGroupByColumn(activeGroupByColumn);
    if (activeGroupByColumn !== null) {
      setColumn(activeGroupByColumn);
    }
  }

  const groupCounts = useItemGroupCounts(tableId, column);

  return useMemo(
    () => ({ objectName, column, setColumn, groupCounts }),
    [objectName, column, groupCounts],
  );
}
