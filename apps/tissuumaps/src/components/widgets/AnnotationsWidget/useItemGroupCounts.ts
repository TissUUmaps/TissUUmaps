import { useEffect, useState } from "react";

import { type TableData, TableUtils } from "@tissuumaps/core";

import { useTableData } from "@/data/hooks/useData";

type LoadedGroupCounts = {
  tableData: TableData;
  column: string;
  groupCounts: Map<string, number>;
};

/**
 * Loads how many rows of a table each group of a categorical column holds
 *
 * Groups are the cell values as strings (see `TableUtils.loadGroupCounts`).
 *
 * @param tableId - The ID of the table, if any
 * @param column - The name of the categorical table column, if any
 * @returns The row count of every group, in the order the groups first appear,
 * or `null` without a table or column, while loading, or if loading failed
 */
export function useItemGroupCounts(
  tableId: string | null,
  column: string | null,
): Map<string, number> | null {
  const tableData = useTableData(tableId);

  // the counts are kept with what they were loaded from, so that the ones of
  // a previous table or column are not returned as the current ones
  const [loadedGroupCounts, setLoadedGroupCounts] =
    useState<LoadedGroupCounts | null>(null);

  useEffect(() => {
    if (tableData === null || column === null) {
      return;
    }
    const abortController = new AbortController();
    TableUtils.loadGroupCounts(tableData, column, {
      signal: abortController.signal,
    })
      .then((groupCounts) => {
        if (!abortController.signal.aborted) {
          setLoadedGroupCounts({ tableData, column, groupCounts });
        }
      })
      .catch((error) => {
        if (!abortController.signal.aborted) {
          console.error(
            `Failed to load the group counts of column '${column}'`,
            error,
          );
        }
      });
    return () => {
      abortController.abort();
    };
  }, [tableData, column]);

  return loadedGroupCounts?.tableData === tableData &&
    loadedGroupCounts.column === column
    ? loadedGroupCounts.groupCounts
    : null;
}
