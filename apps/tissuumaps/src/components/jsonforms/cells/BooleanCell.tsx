import type { CellProps } from "@jsonforms/core";
import { withJsonFormsCellProps } from "@jsonforms/react";

import { Switch } from "@/components/ui/switch";

export const BooleanCell = withJsonFormsCellProps((props: CellProps) => {
  if (!props.enabled) {
    return props.data ? "Yes" : "No";
  }
  const options = {
    ...(props.config as { [key: string]: unknown }),
    ...props.uischema.options,
  };
  return (
    <Switch
      id={props.id}
      checked={!!props.data}
      onCheckedChange={(checked) => props.handleChange(props.path, checked)}
      autoFocus={options.focus as boolean | undefined}
    />
  );
});
