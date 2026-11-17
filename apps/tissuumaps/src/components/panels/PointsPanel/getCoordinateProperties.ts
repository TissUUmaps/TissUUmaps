import type { UISchemaElement } from "@jsonforms/core";

/**
 * Finds the data source properties holding the X and Y coordinate columns,
 * as declared by the controls of a data provider's UI schema (see
 * `DataProvider.uischema`)
 *
 * @param uischema - The UI schema of the data provider
 * @returns The names of the X and Y properties, or `null` unless the UI schema
 * declares both
 */
export function getCoordinateProperties(
  uischema: UISchemaElement,
): { x: string; y: string } | null {
  const properties: { x?: string; y?: string } = {};
  const visit = (element: UISchemaElement) => {
    const coordinate = (element.options as { coordinate?: unknown } | undefined)
      ?.coordinate;
    const scope = (element as { scope?: unknown }).scope;
    if (
      (coordinate === "x" || coordinate === "y") &&
      typeof scope === "string" &&
      scope.startsWith("#/properties/")
    ) {
      properties[coordinate] = scope.slice("#/properties/".length);
    }
    const { elements } = element as { elements?: UISchemaElement[] };
    elements?.forEach(visit);
  };
  visit(uischema);
  const { x, y } = properties;
  return x !== undefined && y !== undefined ? { x, y } : null;
}
