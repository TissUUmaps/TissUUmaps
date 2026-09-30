import { useEffect, useState } from "react";

import { TableUtils } from "@tissuumaps/core";

import { useTableData } from "./useData";

/**
 * Loads the range of the values of a numerical table column
 *
 * @param tableId - The ID of the table, if any
 * @param column - The name of the numerical table column, if any
 * @returns The minimum and maximum value of the column, or `null` without a
 * table or column, while loading, if loading failed, or if the column is not
 * numerical (see `TableUtils.loadValueRange`)
 */
export function useValueRange(
  tableId: string | null,
  column: string | null,
): [number, number] | null {
  const tableData = useTableData(tableId);

  // the range is kept with what it was loaded from, so that the one of a
  // previous table or column is not returned as the current one
  const [loadedValueRange, setLoadedValueRange] = useState<{
    tableId: string;
    column: string;
    valueRange: [number, number] | undefined;
  } | null>(null);

  useEffect(() => {
    if (tableId === null || tableData === null || column === null) {
      return;
    }
    const abortController = new AbortController();
    TableUtils.loadValueRange(tableData, column, {
      signal: abortController.signal,
    })
      .then((valueRange) => {
        if (!abortController.signal.aborted) {
          setLoadedValueRange({ tableId, column, valueRange });
        }
      })
      .catch((error) => {
        if (!abortController.signal.aborted) {
          console.error("Error loading table value range", error);
        }
      });
    return () => abortController.abort();
  }, [tableId, tableData, column]);

  return tableData !== null &&
    loadedValueRange?.tableId === tableId &&
    loadedValueRange.column === column
    ? (loadedValueRange.valueRange ?? null)
    : null;
}
