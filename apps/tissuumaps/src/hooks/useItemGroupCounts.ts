import { useEffect, useState } from "react";

import type { TableData } from "@tissuumaps/core";

import { useTableData } from "./useData";

type LoadedGroupCounts = {
  tableData: TableData;
  column: string;
  groupCounts: Map<string, number>;
};

/**
 * Loads how many rows of a table each group of a categorical column holds
 *
 * Groups are the cell values as strings, as group-to-value maps key them, so
 * values with the same string (e.g. `null` and `"null"`) form one group.
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
    tableData
      .loadUniqueValueCounts<unknown>(column, {
        signal: abortController.signal,
      })
      .then((uniqueValueCounts) => {
        if (!abortController.signal.aborted) {
          const groupCounts = new Map<string, number>();
          for (const [value, count] of uniqueValueCounts) {
            const group = String(value);
            groupCounts.set(group, (groupCounts.get(group) ?? 0) + count);
          }
          setLoadedGroupCounts({ tableData, column, groupCounts });
        }
      })
      .catch((error) => {
        if (!abortController.signal.aborted) {
          console.error("Error loading table unique value counts", error);
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
