import { type Config, ConfigUtils, type GroupByConfig } from "@tissuumaps/core";

import type { GroupTableState } from "./useGroupTable";

/**
 * Determines whether a configuration groups by the column of the group table
 *
 * @param config - The configuration
 * @param groupTable - The state of the group table
 * @returns Whether `groupBy` is the active source and groups by that column
 */
export function isGroupedByColumn<TConfig extends Config<string>>(
  config: TConfig,
  groupTable: GroupTableState,
): config is Extract<TConfig, GroupByConfig<false>> {
  const groupByColumn = ConfigUtils.getGroupByColumn(config);
  return (
    groupTable.column !== null &&
    groupByColumn !== undefined &&
    ConfigUtils.isSameTableColumn(
      groupByColumn,
      groupTable.column,
      groupTable.tableId ?? undefined,
    )
  );
}
