import type { Layout, LayoutProps } from "@jsonforms/core";
import {
  JsonFormsDispatch,
  useJsonForms,
  withJsonFormsLayoutProps,
} from "@jsonforms/react";

// The fields go straight into the caller's `<dl>`, without a layout of their
// own, so that every field is one row of the same grid
export const ListLayout = withJsonFormsLayoutProps(
  ({ uischema, schema, path, visible }: LayoutProps) => {
    const { renderers, cells } = useJsonForms();
    return visible
      ? (uischema as Layout).elements.map((element, i) => (
          <JsonFormsDispatch
            key={`${path}-${i}`}
            renderers={renderers}
            cells={cells}
            uischema={element}
            schema={schema}
            path={path}
            enabled={false}
          />
        ))
      : null;
  },
  false,
);
