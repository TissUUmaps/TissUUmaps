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
  const counts = new Map<string, { column: TableColumnRef; count: number }>();
  for (const config of configs) {
    const column = ConfigUtils.getGroupByColumn(config);
    if (column !== undefined) {
      const key = JSON.stringify([
        column.table ?? defaultTable ?? null,
        column.column,
      ]);
      counts.set(key, { column, count: (counts.get(key)?.count ?? 0) + 1 });
    }
  }
  let dominantColumn: TableColumnRef | null = null;
  let dominantCount = 0;
  // the insertion order of the counts is the priority order of the configs
  for (const { column, count } of counts.values()) {
    if (count > dominantCount) {
      dominantColumn = column;
      dominantCount = count;
    }
  }
  return dominantColumn;
}
