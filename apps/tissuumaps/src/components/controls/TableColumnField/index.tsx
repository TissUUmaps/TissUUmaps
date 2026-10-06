import { useState } from "react";

import type { Table, TableColumnRef } from "@tissuumaps/core";

import { Field, FieldLabel } from "@/components/common/field";
import { SimpleSelect } from "@/components/common/simple-select";
import { useProjectStore } from "@/stores/project";

import { TableColumnInput } from "./TableColumnInput";

export type TableColumnFieldProps = {
  label: string;
  tableId: string | null;
  value: TableColumnRef | null;
  onValueChange: (value: TableColumnRef | null) => void;
  className?: string;
};

/**
 * The select value for the object's own table, which is not stored
 *
 * Not a string, so that it never matches a table ID; base-ui would also treat
 * an empty string as no value and show it as a placeholder.
 */
const sourceTableValue = -1;

/**
 * A labelled input for a table column, of the object's own table by default
 *
 * The label names the table the column is taken from, as a compact select. The
 * default "source table" stores no table, so that the column follows the
 * object's table; choosing a table stores it, even the object's own one.
 */
export function TableColumnField({
  label,
  tableId,
  value,
  onValueChange,
  className,
}: TableColumnFieldProps) {
  const tables = useProjectStore((state) => state.tables);
  const ownTable = tables.find((table) => table.id === tableId);

  const valueTable = value?.table;
  const [selectedTable, setSelectedTable] = useState(valueTable);
  // https://react.dev/reference/react/useState#storing-information-from-previous-renders
  // clearing the column keeps the chosen table
  const [prevValueTable, setPrevValueTable] = useState(valueTable);
  const [prevTableId, setPrevTableId] = useState(tableId);
  if (
    (value !== null && valueTable !== prevValueTable) ||
    tableId !== prevTableId
  ) {
    setPrevValueTable(valueTable);
    setPrevTableId(tableId);
    setSelectedTable(valueTable);
  }

  return (
    <Field disabled={tableId === null} className={className}>
      <div className="flex min-w-0 flex-row items-center gap-x-1">
        <FieldLabel className="shrink-0 whitespace-nowrap">
          {ownTable !== undefined ? `${label} from` : label}
        </FieldLabel>
        {ownTable !== undefined && (
          <Field
            disabled={tableId === null}
            className="flex min-w-0"
            title="Click to take the column from another table"
          >
            <SimpleSelect<Table | null, false, string | number>
              aria-label="Table"
              items={[null, ...tables]}
              itemLabel={(table) =>
                table !== null ? (
                  <TableName table={table} tables={tables} />
                ) : (
                  <span className="truncate">
                    source table
                    <span className="ml-1 text-muted-foreground">
                      ({ownTable.name})
                    </span>
                  </span>
                )
              }
              variant="inline"
              itemValue={(table) => table?.id ?? sourceTableValue}
              value={selectedTable ?? sourceTableValue}
              onValueChange={(newTableId) => {
                if (newTableId === null) {
                  return;
                }
                const newTable =
                  typeof newTableId === "string" ? newTableId : undefined;
                setSelectedTable(newTable);
                // the column input resolves the column again only if the
                // queried table changes
                if (
                  value !== null &&
                  (newTable ?? tableId) === (selectedTable ?? tableId)
                ) {
                  onValueChange({ table: newTable, column: value.column });
                }
              }}
            />
          </Field>
        )}
      </div>
      <TableColumnInput
        tableId={selectedTable ?? tableId}
        value={value?.column ?? null}
        onValueChange={(column) =>
          onValueChange(
            column !== null ? { table: selectedTable, column } : null,
          )
        }
      />
    </Field>
  );
}

type TableNameProps = {
  table: Table;
  tables: Table[];
};

function TableName({ table, tables }: TableNameProps) {
  const isNameShared = tables.some(
    (otherTable) => otherTable !== table && otherTable.name === table.name,
  );
  return (
    <span className="truncate italic">
      {table.name}
      {isNameShared && (
        <span className="ml-1 text-muted-foreground">({table.id})</span>
      )}
    </span>
  );
}
