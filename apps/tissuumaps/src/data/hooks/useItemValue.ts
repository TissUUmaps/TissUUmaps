import { useEffect, useState } from "react";

import { ArrayUtils, type TableData } from "@tissuumaps/core";

import { useTableData } from "./useData";

/**
 * Loads the value of an item in a table column
 *
 * The item's row is the first one with its ID.
 *
 * @param tableId - The ID of the table, if any
 * @param column - The name of the table column, if any
 * @param itemId - The ID of the item, if any
 * @returns The value, or `null` without a table, column or item, while
 * loading, if loading failed, or if the table has no row for the item
 */
export function useItemValue(
  tableId: string | null,
  column: string | null,
  itemId: number | string | null,
): { value: unknown } | null {
  const tableData = useTableData(tableId);

  // the value is kept with what it was loaded from, so that the one of a
  // previous table, column or item is not returned as the current one
  const [loadedItemValue, setLoadedItemValue] = useState<{
    tableData: TableData;
    column: string;
    itemId: number | string;
    itemValue: { value: unknown } | null;
  } | null>(null);

  useEffect(() => {
    if (tableData === null || column === null || itemId === null) {
      return;
    }
    const abortController = new AbortController();
    const { signal } = abortController;
    tableData
      .loadValues<unknown>(column, { signal })
      .then((values) => {
        if (!signal.aborted) {
          const rowIndex = ArrayUtils.indexOfId(tableData.getIds(), itemId);
          const itemValue =
            rowIndex !== -1 ? { value: values[rowIndex] } : null;
          setLoadedItemValue({ tableData, column, itemId, itemValue });
        }
      })
      .catch((error) => {
        if (!signal.aborted) {
          console.error(
            `Failed to load the value of item ${itemId} in column '${column}'`,
            error,
          );
        }
      });
    return () => {
      abortController.abort();
    };
  }, [tableData, column, itemId]);

  return loadedItemValue?.tableData === tableData &&
    loadedItemValue.column === column &&
    loadedItemValue.itemId === itemId
    ? loadedItemValue.itemValue
    : null;
}
