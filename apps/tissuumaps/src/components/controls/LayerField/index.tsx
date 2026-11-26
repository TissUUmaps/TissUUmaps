import type { TableColumnRef } from "@tissuumaps/core";

import { Field, FieldLabel } from "@/components/common/field";
import { SimpleSelect } from "@/components/common/simple-select";
import { formatTableColumn } from "@/components/controls/TableColumnField/formatTableColumn";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

export type LayerFieldProps = {
  value: string | TableColumnRef;
  onValueChange: (value: string) => void;
  className?: string;
};

/**
 * A labelled select for the layer of an object
 *
 * A layer read from a table column is shown read-only.
 */
export function LayerField({
  value,
  onValueChange,
  className,
}: LayerFieldProps) {
  const layers = useProjectStore((state) => state.layers);
  const tables = useProjectStore((state) => state.tables);
  return (
    <Field className={cn("flex flex-row items-center gap-x-2", className)}>
      <FieldLabel>Layer</FieldLabel>
      {typeof value === "string" ? (
        <SimpleSelect
          items={layers}
          itemLabel={(l) => l.name}
          itemValue={(l) => l.id}
          value={value}
          onValueChange={(newValue) => {
            if (newValue !== null) {
              onValueChange(newValue);
            }
          }}
        />
      ) : (
        <Input
          disabled
          value={`column: ${formatTableColumn(value, tables)}`}
          readOnly
        />
      )}
    </Field>
  );
}
