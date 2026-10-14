import { type ControlProps, computeLabel } from "@jsonforms/core";
import { DispatchCell, withJsonFormsControlProps } from "@jsonforms/react";

import { cn } from "@/lib/utils";

export const ListControl = withJsonFormsControlProps((props: ControlProps) => {
  const isUnset =
    props.data === undefined || props.data === null || props.data === "";
  // an unset field shows its default, if any, as that is what is used
  if (!props.visible || (isUnset && props.schema.default === undefined)) {
    return null;
  }
  return (
    <div className="contents">
      <dt className="text-muted-foreground">
        {computeLabel(props.label, props.required ?? false, true)}
      </dt>
      <dd className={cn("wrap-anywhere", isUnset && "text-muted-foreground")}>
        {isUnset ? (
          // not formatted by a cell, e.g. as an enum label
          String(props.schema.default)
        ) : (
          <DispatchCell
            uischema={props.uischema}
            schema={props.schema}
            path={props.path}
            enabled={false}
          />
        )}
      </dd>
    </div>
  );
});
