import type {
  GroupLayout as GroupLayoutSchema,
  LayoutProps,
} from "@jsonforms/core";
import {
  JsonFormsDispatch,
  useJsonForms,
  withJsonFormsLayoutProps,
} from "@jsonforms/react";
import { memo } from "react";

import { Fieldset, FieldsetLegend } from "@/components/common/fieldset";

// eslint-disable-next-line react-refresh/only-export-components
const MemoizedGroupLayout = memo((props: Omit<LayoutProps, "data">) => {
  const layout = props.uischema as GroupLayoutSchema;
  const { renderers, cells } = useJsonForms();
  return (
    <Fieldset hidden={!props.visible} className="flex flex-col">
      {props.label && <FieldsetLegend>{props.label}</FieldsetLegend>}
      {layout.elements.map((element, i) => (
        <div key={`${props.path}-${i}`}>
          <JsonFormsDispatch
            renderers={renderers}
            cells={cells}
            uischema={element}
            schema={props.schema}
            path={props.path}
            enabled={props.enabled}
          />
        </div>
      ))}
    </Fieldset>
  );
});

export const GroupLayout = withJsonFormsLayoutProps(
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  ({ data, ...otherProps }: LayoutProps) => {
    return <MemoizedGroupLayout {...otherProps} />;
  },
  false,
);
