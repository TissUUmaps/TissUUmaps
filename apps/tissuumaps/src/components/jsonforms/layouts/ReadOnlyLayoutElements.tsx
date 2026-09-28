import type { Layout, LayoutProps } from "@jsonforms/core";
import { JsonFormsDispatch, useJsonForms } from "@jsonforms/react";

// The fields go straight into the widget's `<dl>`, without a layout of their
// own, so that every read-only field is one row of the same grid
export function ReadOnlyLayoutElements({
  uischema,
  schema,
  path,
  visible,
}: Omit<LayoutProps, "data">) {
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
}
