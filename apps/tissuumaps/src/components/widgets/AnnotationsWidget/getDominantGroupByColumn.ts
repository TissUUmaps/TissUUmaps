import {
  type Config,
  ConfigUtils,
  type TableColumnRef,
} from "@tissuumaps/core";

/**
 * Returns the table column that most of the given configurations group by
 *
 * @param configs - Property configurations, in order of priority
 * @param defaultTable - The table of columns that name none
 * @returns The column, or `null` if no configuration groups by one
 */
export function getDominantGroupByColumn(
  configs: Config<string>[],
  defaultTable?: string,
): TableColumnRef | null {
  const counts = new Map<string, { groupBy: TableColumnRef; count: number }>();
  for (const config of configs) {
    const groupBy = ConfigUtils.getGroupByColumn(config);
    if (groupBy !== undefined) {
      // table IDs contain no colon
      const key = `${groupBy.table ?? defaultTable ?? ""}:${groupBy.column}`;
      counts.set(key, { groupBy, count: (counts.get(key)?.count ?? 0) + 1 });
    }
  }
  let dominantGroupBy: TableColumnRef | null = null;
  let dominantCount = 0;
  // the insertion order of the counts is the priority order of the configs
  for (const { groupBy, count } of counts.values()) {
    if (count > dominantCount) {
      dominantGroupBy = groupBy;
      dominantCount = count;
    }
  }
  return dominantGroupBy;
}
