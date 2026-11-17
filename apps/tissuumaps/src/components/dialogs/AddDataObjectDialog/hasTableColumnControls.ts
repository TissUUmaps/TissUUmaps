import type { UISchemaElement } from "@jsonforms/core";

/**
 * Returns whether a data provider's form picks columns of a table, i.e. has
 * a control declaring `options: { tableColumn: true }` (see
 * `DataProvider.uischema`)
 *
 * @param uischema - The UI schema of the data provider
 * @returns Whether the UI schema has a table column control
 */
export function hasTableColumnControls(uischema: UISchemaElement): boolean {
  const options = uischema.options as { tableColumn?: unknown } | undefined;
  if (options?.tableColumn === true) {
    return true;
  }
  const { elements } = uischema as { elements?: UISchemaElement[] };
  return elements?.some(hasTableColumnControls) ?? false;
}
