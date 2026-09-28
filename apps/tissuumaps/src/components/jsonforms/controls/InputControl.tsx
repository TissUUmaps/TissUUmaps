import {
  type ControlProps,
  computeLabel,
  isDescriptionHidden,
} from "@jsonforms/core";
import { DispatchCell, withJsonFormsControlProps } from "@jsonforms/react";
import { useState } from "react";

import {
  Field,
  FieldDescription,
  FieldError,
  FieldLabel,
} from "@/components/common/field";

export const InputControl = withJsonFormsControlProps((props: ControlProps) => {
  const [isFocused, setFocused] = useState<boolean>(false);

  // readonly mode
  if (!props.enabled) {
    if (props.data === undefined || props.data === null || props.data === "") {
      return null;
    }
    return (
      <div className="contents">
        <dt className="text-muted-foreground">
          {computeLabel(props.label, props.required ?? false, true)}
        </dt>
        <dd className="wrap-anywhere">
          <DispatchCell
            uischema={props.uischema}
            schema={props.schema}
            path={props.path}
            enabled={props.enabled}
          />
        </dd>
      </div>
    );
  }

  const options = {
    ...(props.config as { [key: string]: unknown }),
    ...props.uischema.options,
  };
  const showDescription = !isDescriptionHidden(
    props.visible,
    props.description,
    isFocused,
    (options.showUnfocusedDescription as boolean | undefined) ?? false,
  );

  return (
    <Field onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}>
      <FieldLabel>
        {computeLabel(
          props.label,
          props.required ?? false,
          (options.hideRequiredAsterisk as boolean | undefined) ?? false,
        )}
      </FieldLabel>
      {showDescription && (
        <FieldDescription>{props.description}</FieldDescription>
      )}
      <DispatchCell
        uischema={props.uischema}
        schema={props.schema}
        path={props.path}
        id={props.id + "-input"}
        enabled={props.enabled}
      />
      {props.errors && <FieldError>{props.errors}</FieldError>}
    </Field>
  );
});
