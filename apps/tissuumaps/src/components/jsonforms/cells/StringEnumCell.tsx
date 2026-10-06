import type { EnumCellProps } from "@jsonforms/core";
import { withJsonFormsEnumCellProps } from "@jsonforms/react";

import { SimpleSelect } from "@/components/common/simple-select";

export const StringEnumCell = withJsonFormsEnumCellProps(
  (props: EnumCellProps) => {
    const value = props.data as string;
    if (!props.enabled) {
      const option = props.options?.find((option) => option.value === value);
      return option?.label ?? value;
    }
    return (
      <SimpleSelect
        id={props.id}
        value={value}
        onValueChange={(newValue) => props.handleChange(props.path, newValue)}
        items={props.options || []}
        itemLabel={(option) => option.label}
        itemValue={(option) => option.value as string}
      />
    );
  },
);
