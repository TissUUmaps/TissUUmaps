import {
  type ColorConfig,
  type Labels,
  type Layer,
  type Points,
  type Shapes,
  type TableColumnRef,
  type VisibilityConfig,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
} from "@tissuumaps/core";

/**
 * Finds the table column that a color configuration takes its colors from
 *
 * The color settings keep the inactive sources as drafts, so a "from" source
 * only counts while it is the active one.
 *
 * @param colorConfig - The color configuration
 * @returns The active "from" source, or `null` if another source is active
 */
export function getActiveFrom(colorConfig: ColorConfig) {
  return getActiveConfigSource(colorConfig) === "from" &&
    isFromConfig(colorConfig)
    ? colorConfig.from
    : null;
}

// Only a constant visibility is checked, as a colorbar is still useful for the
// items that a table column leaves visible.
function isConstantlyHidden(visibilityConfig: VisibilityConfig): boolean {
  return (
    getActiveConfigSource(visibilityConfig) === "constant" &&
    isConstantConfig(visibilityConfig) &&
    !visibilityConfig.constant.value
  );
}

// Only a layer ID is checked: with a layer per item, a colorbar is still useful
// for the items on the visible layers.
function isOnHiddenLayer(
  layer: string | TableColumnRef,
  layers: Layer[],
): boolean {
  return (
    typeof layer === "string" &&
    layers.find(({ id }) => id === layer)?.visibility === false
  );
}

/**
 * Lists the colorbars of the shown objects whose colors come from a table
 * column
 *
 * Shapes get a colorbar for their fill and one for their stroke.
 *
 * @param state - The layers and the objects of the project
 * @returns One colorbar per color, with a unique key, the name to show, the
 * color and visibility configurations, and the table of the object's data
 * source
 */
export function getColorbars(state: {
  layers: Layer[];
  labels: Labels[];
  points: Points[];
  shapes: Shapes[];
}) {
  const { layers, labels, points, shapes } = state;
  return [
    ...labels
      // without a table of their own, the renderer draws labels in default
      // colors
      .filter(
        (labels) =>
          labels.visibility &&
          !isOnHiddenLayer(labels.layer, layers) &&
          labels.dataSource.table !== undefined,
      )
      .map((labels) => ({
        key: `labels-${labels.id}`,
        name: labels.name,
        colorConfig: labels.labelColor,
        visibilityConfig: labels.labelVisibility,
        tableId: labels.dataSource.table ?? null,
      })),
    ...points
      .filter(
        (points) => points.visibility && !isOnHiddenLayer(points.layer, layers),
      )
      .map((points) => ({
        key: `points-${points.id}`,
        name: points.name,
        colorConfig: points.pointColor,
        visibilityConfig: points.pointVisibility,
        tableId: points.dataSource.table ?? null,
      })),
    ...shapes
      .filter(
        (shapes) =>
          shapes.visibility &&
          !isOnHiddenLayer(shapes.layer, layers) &&
          !isConstantlyHidden(shapes.shapeVisibility),
      )
      .flatMap((shapes) => [
        {
          key: `shapes-${shapes.id}-fill`,
          name: `${shapes.name} (fill)`,
          colorConfig: shapes.shapeFillColor,
          visibilityConfig: shapes.shapeFillVisibility,
          tableId: shapes.dataSource.table ?? null,
        },
        {
          key: `shapes-${shapes.id}-stroke`,
          name: `${shapes.name} (stroke)`,
          colorConfig: shapes.shapeStrokeColor,
          visibilityConfig: shapes.shapeStrokeVisibility,
          tableId: shapes.dataSource.table ?? null,
        },
      ]),
  ].filter(
    ({ colorConfig, visibilityConfig }) =>
      getActiveFrom(colorConfig) !== null &&
      !isConstantlyHidden(visibilityConfig),
  );
}
