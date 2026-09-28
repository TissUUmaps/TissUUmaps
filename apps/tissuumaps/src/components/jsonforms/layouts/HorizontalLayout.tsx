import type {
  HorizontalLayout as HorizontalLayoutSchema,
  LayoutProps,
} from "@jsonforms/core";
import {
  JsonFormsDispatch,
  useJsonForms,
  withJsonFormsLayoutProps,
} from "@jsonforms/react";
import { memo } from "react";

import { ReadOnlyLayoutElements } from "./ReadOnlyLayoutElements";

// eslint-disable-next-line react-refresh/only-export-components
const MemoizedHorizontalLayout = memo((props: Omit<LayoutProps, "data">) => {
  const layout = props.uischema as HorizontalLayoutSchema;
  const { renderers, cells } = useJsonForms();

  if (!props.enabled) {
    return <ReadOnlyLayoutElements {...props} />;
  }

  return (
    <div hidden={!props.visible} className="grid grid-flow-col">
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
    </div>
  );
});

export const HorizontalLayout = withJsonFormsLayoutProps(
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  ({ data, ...otherProps }: LayoutProps) => {
    return <MemoizedHorizontalLayout {...otherProps} />;
  },
  false,
);
