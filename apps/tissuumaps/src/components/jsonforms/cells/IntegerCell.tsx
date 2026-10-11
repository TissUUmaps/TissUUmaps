import type { CellProps } from "@jsonforms/core";
import { withJsonFormsCellProps } from "@jsonforms/react";

import { Input } from "@/components/ui/input";

export const IntegerCell = withJsonFormsCellProps((props: CellProps) => {
  const value = (props.data as string | number | undefined | null) ?? "";
  if (!props.enabled) {
    return value;
  }
  const options = {
    ...(props.config as { [key: string]: unknown }),
    ...props.uischema.options,
  };
  return (
    <Input
      type="number"
      step="1"
      id={props.id}
      value={value}
      onChange={(event) =>
        props.handleChange(
          props.path,
          event.target.value !== "" ? parseInt(event.target.value) : undefined,
        )
      }
      autoFocus={options.focus as boolean | undefined}
    />
  );
});
