import { useEffect, useState } from "react";

import type { TableData } from "@tissuumaps/core";

/**
 * Where a column is in a table with sections: in a leaf section, under its
 * name there; in no section (e.g. the `obs` index); or not in the table
 */
export type ColumnLocation = { column: string } & (
  | { kind: "section"; sectionId: string; name: string }
  | { kind: "unsectioned" }
  | { kind: "missing" }
);

/**
 * Loads where the followed column is, for tables with sections (see
 * `TableData.describeColumn`)
 *
 * The location is kept until the next one resolves, so that the rows and the
 * header do not change in between; its `column` tells whether it is stale. A
 * failed load is logged and keeps the previous location.
 *
 * @param tableData - The object's own table, if loaded
 * @param column - The followed column, if any
 * @returns The location of the column, or `null` if there is no column or
 * the table has no sections
 */
export function useColumnLocation(
  tableData: TableData | null,
  column: string | null,
): ColumnLocation | null {
  const [loaded, setLoaded] = useState<{
    tableData: TableData;
    location: ColumnLocation;
  } | null>(null);

  useEffect(() => {
    if (tableData?.describeColumn === undefined || column === null) {
      return;
    }
    const abortController = new AbortController();
    const { signal } = abortController;
    const describing = tableData.describeColumn(column, { signal });
    (async () => {
      const description = await describing;
      // a column in no section still resolves; a missing one does not
      const location: ColumnLocation =
        description !== null
          ? { column, kind: "section", ...description }
          : (await tableData.resolveColumnQuery(column, { signal })) !== null
            ? { column, kind: "unsectioned" }
            : { column, kind: "missing" };
      if (!signal.aborted) {
        setLoaded({ tableData, location });
      }
    })().catch((error: unknown) => {
      if (!signal.aborted) {
        console.error("Failed to describe the column", error);
      }
    });
    return () => abortController.abort();
  }, [tableData, column]);

  if (
    column === null ||
    tableData?.describeColumn === undefined ||
    loaded?.tableData !== tableData
  ) {
    return null;
  }
  return loaded.location;
}
