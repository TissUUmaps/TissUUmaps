import type { CellProps } from "@jsonforms/core";
import { useJsonForms, withJsonFormsCellProps } from "@jsonforms/react";

import { TableColumnInput } from "@/components/controls/TableColumnField/TableColumnInput";

export const TableColumnCell = withJsonFormsCellProps((props: CellProps) => {
  const { core } = useJsonForms();
  const tableId = (core?.data as { table?: string } | undefined)?.table ?? null;
  // disabled rather than replaced by text, e.g. while the form is read-only:
  // remounting the input inside the still mounted field would make the field
  // register its controls again and again
  const options = {
    ...(props.config as { [key: string]: unknown }),
    ...props.uischema.options,
  };
  return (
    <TableColumnInput
      tableId={tableId}
      value={(props.data as string | undefined) ?? null}
      onValueChange={(column) =>
        props.handleChange(props.path, column ?? undefined)
      }
      placeholder={options.placeholder as string | undefined}
      disabled={!props.enabled}
    />
  );
});
