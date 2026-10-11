import type { CellProps } from "@jsonforms/core";
import { withJsonFormsCellProps } from "@jsonforms/react";

import { Input } from "@/components/ui/input";

export const TextCell = withJsonFormsCellProps((props: CellProps) => {
  const value = (props.data as string | undefined | null) ?? "";
  if (!props.enabled) {
    return value;
  }
  const options = {
    ...(props.config as { [key: string]: unknown }),
    ...props.uischema.options,
  };
  return (
    <Input
      type="text"
      id={props.id}
      value={value}
      onChange={(event) =>
        props.handleChange(
          props.path,
          event.target.value !== "" ? event.target.value : undefined,
        )
      }
      autoFocus={options.focus as boolean | undefined}
      placeholder={options.placeholder as string | undefined}
      maxLength={props.schema.maxLength}
    />
  );
});
