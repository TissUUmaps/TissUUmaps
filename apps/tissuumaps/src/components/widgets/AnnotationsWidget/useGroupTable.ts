import { useMemo, useState } from "react";

import {
  type Config,
  ConfigUtils,
  type TableColumnRef,
} from "@tissuumaps/core";

import { getDominantGroupByColumn } from "./getDominantGroupByColumn";
import { useItemGroupCounts } from "./useItemGroupCounts";

/** The state of the group table: the table column it groups by, and its groups */
export type GroupTableState = {
  /** The name of the annotated object, which new maps are named after */
  objectName: string;

  /** The ID of the annotated object's table, which columns naming none are of */
  tableId: string | null;

  /** The table column that the group table groups by, if any */
  groupBy: TableColumnRef | null;

  setGroupBy: (groupBy: TableColumnRef | null) => void;

  /** The row count of every group, or `null` while there is none */
  groupCounts: Map<string, number> | null;
};

function isSameGroupBy(
  groupBy: TableColumnRef | null,
  otherGroupBy: TableColumnRef | null,
): boolean {
  return groupBy === null || otherGroupBy === null
    ? groupBy === otherGroupBy
    : ConfigUtils.isSameTableColumn(groupBy, otherGroupBy);
}

/**
 * Chooses the table column that the group table groups by, and counts its
 * groups
 *
 * The group table starts from the column that most settings group by. When a
 * setting starts grouping by a column, the group table switches to it;
 * otherwise it keeps its column, including one picked by the user, until the
 * table changes.
 *
 * @param objectName - The name of the annotated object
 * @param tableId - The ID of the annotated object's table
 * @param configs - The configurations of the annotated object's settings, in
 * order of priority
 * @returns The state of the group table, shared by its columns
 */
export function useGroupTable(
  objectName: string,
  tableId: string | null,
  configs: Config<string>[],
): GroupTableState {
  const groupBys = configs.map((config) =>
    tableId !== null ? (ConfigUtils.getGroupByColumn(config) ?? null) : null,
  );

  const defaultGroupBy =
    tableId !== null ? getDominantGroupByColumn(configs, tableId) : null;

  const [groupBy, setGroupBy] = useState<TableColumnRef | null>(defaultGroupBy);

  // https://react.dev/reference/react/useState#storing-information-from-previous-renders
  const [prevTableId, setPrevTableId] = useState(tableId);
  const [prevGroupBys, setPrevGroupBys] = useState(groupBys);
  const isChanged = (newGroupBy: TableColumnRef | null, index: number) =>
    !isSameGroupBy(newGroupBy, prevGroupBys[index] ?? null);
  if (tableId !== prevTableId) {
    setPrevTableId(tableId);
    setPrevGroupBys(groupBys);
    setGroupBy(defaultGroupBy);
  } else if (groupBys.some(isChanged)) {
    setPrevGroupBys(groupBys);
    const startedGroupBy =
      groupBys.find(
        (newGroupBy, index) =>
          newGroupBy !== null && isChanged(newGroupBy, index),
      ) ?? null;
    if (startedGroupBy !== null) {
      setGroupBy(startedGroupBy);
    }
  }

  const groupCounts = useItemGroupCounts(
    groupBy?.table ?? tableId,
    groupBy?.column ?? null,
  );

  return useMemo(
    () => ({ objectName, tableId, groupBy, setGroupBy, groupCounts }),
    [objectName, tableId, groupBy, groupCounts],
  );
}
