import { useMemo, useState } from "react";

import { type Config, ConfigUtils } from "@tissuumaps/core";

/** A setting of an annotated object and its configuration */
export type GroupTableSetting = {
  /** The settings category of the setting */
  category: string;

  /** The configuration of the setting */
  config: Config<string>;
};

/** The state of the group table: the table column it groups by */
export type GroupTableState = {
  /** The table column that the group table groups by, if any */
  column: string | null;

  setColumn: (column: string | null) => void;
};

/**
 * Chooses the table column that the group table groups by
 *
 * The group table follows the column that the open settings category groups
 * by, and keeps the column picked by the user otherwise.
 *
 * @param tableId - The ID of the annotated object's table
 * @param settings - The settings of the annotated object
 * @param activeSettingsCategory - The open settings category, if any
 * @returns The group-by column and its setter, shared by the columns of the
 * group table
 */
export function useGroupTable(
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

  const [column, setColumn] = useState(activeGroupByColumn);

  // https://react.dev/reference/react/useState#storing-information-from-previous-renders
  const [prevActiveGroupByColumn, setPrevActiveGroupByColumn] =
    useState(activeGroupByColumn);
  if (activeGroupByColumn !== prevActiveGroupByColumn) {
    setPrevActiveGroupByColumn(activeGroupByColumn);
    if (activeGroupByColumn !== null) {
      setColumn(activeGroupByColumn);
    }
  }

  return useMemo(() => ({ column, setColumn }), [column]);
}
