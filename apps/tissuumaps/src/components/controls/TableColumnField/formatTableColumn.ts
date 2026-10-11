import type { Table, TableColumnRef } from "@tissuumaps/core";

/**
 * Formats a table column reference for display
 *
 * A column of an explicitly chosen table is prefixed by that table's name, or
 * its ID if the table is missing.
 *
 * @param tableColumnRef - The table column reference
 * @param tables - The tables of the current project
 * @returns The displayed column
 */
export function formatTableColumn(
  tableColumnRef: TableColumnRef,
  tables: Table[],
): string {
  if (tableColumnRef.table === undefined) {
    return tableColumnRef.column;
  }
  const table = tables.find((table) => table.id === tableColumnRef.table);
  return `${table?.name ?? tableColumnRef.table}:${tableColumnRef.column}`;
}
