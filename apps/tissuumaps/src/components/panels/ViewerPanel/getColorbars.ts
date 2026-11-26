import {
  type ColorConfig,
  type FromConfig,
  type Labels,
  type Layer,
  type Points,
  type Shapes,
  type Table,
  type TableColumnRef,
  type VisibilityConfig,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
} from "@tissuumaps/core";

type ColorFrom = Extract<ColorConfig, FromConfig>["from"];

/**
 * Finds the table column that a color configuration takes its colors from
 *
 * The color settings keep the inactive sources as drafts, so a "from" source
 * only counts while it is the active one.
 *
 * @param colorConfig - The color configuration
 * @returns The active "from" source, or `null` if another source is active
 */
function getActiveFrom(colorConfig: ColorConfig): ColorFrom | null {
  // mirrors ColorResolver.resolveColors
  return getActiveConfigSource(colorConfig) === "from" &&
    isFromConfig(colorConfig)
    ? colorConfig.from
    : null;
}

/**
 * Checks whether a visibility configuration hides all items
 *
 * Only a constant visibility is checked, as a colorbar is still useful for the
 * items that a table column leaves visible.
 *
 * @param visibilityConfig - The visibility configuration
 * @returns Whether the active source is a constant `false`
 */
function isConstantlyHidden(visibilityConfig: VisibilityConfig): boolean {
  return (
    getActiveConfigSource(visibilityConfig) === "constant" &&
    isConstantConfig(visibilityConfig) &&
    !visibilityConfig.constant.value
  );
}

/**
 * Checks whether an object is on a hidden layer
 *
 * Only a layer ID is checked: with a layer per item, a colorbar is still useful
 * for the items on the visible layers.
 *
 * @param layer - The object's layer ID, or the table column holding the layers
 * @param layers - The layers of the project
 * @returns Whether the object's layer is hidden
 */
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
 * Colors with the same table, column, palette and range share one colorbar.
 * A colorbar is named after the objects that use it. The fill or stroke of
 * shapes is named unless both use the colorbar.
 *
 * @param state - The layers, the tables and the objects of the project
 * @returns One colorbar per color scale, with a unique key, the name to show,
 * the active "from" source and the ID of the table to load the column's value
 * range from; the source names a table only if all colors name the same one
 */
export function getColorbars(state: {
  layers: Layer[];
  tables: Table[];
  labels: Labels[];
  points: Points[];
  shapes: Shapes[];
}): { key: string; name: string; from: ColorFrom; tableId: string }[] {
  const { layers, tables, labels, points, shapes } = state;
  const objectColors = [
    ...labels
      // without a table of their own in the project, the renderer draws labels
      // in default colors, even if the color column is in another table;
      // mirrors OpenSeadragonLabelsRenderer._resolveDataTransfer
      .filter(
        (labels) =>
          labels.visibility &&
          !isOnHiddenLayer(labels.layer, layers) &&
          tables.some(({ id }) => id === labels.dataSource.table),
      )
      .map((labels) => ({
        objectKey: `labels:${labels.id}`,
        objectName: labels.name,
        part: null,
        colorConfig: labels.labelColor,
        visibilityConfig: labels.labelVisibility,
        objectTableId: labels.dataSource.table,
      })),
    ...points
      .filter(
        (points) => points.visibility && !isOnHiddenLayer(points.layer, layers),
      )
      .map((points) => ({
        objectKey: `points:${points.id}`,
        objectName: points.name,
        part: null,
        colorConfig: points.pointColor,
        visibilityConfig: points.pointVisibility,
        objectTableId: points.dataSource.table,
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
          objectKey: `shapes:${shapes.id}`,
          objectName: shapes.name,
          part: "fill" as const,
          colorConfig: shapes.shapeFillColor,
          visibilityConfig: shapes.shapeFillVisibility,
          objectTableId: shapes.dataSource.table,
        },
        {
          objectKey: `shapes:${shapes.id}`,
          objectName: shapes.name,
          part: "stroke" as const,
          colorConfig: shapes.shapeStrokeColor,
          visibilityConfig: shapes.shapeStrokeVisibility,
          objectTableId: shapes.dataSource.table,
        },
      ]),
  ];

  const colorbars = new Map<
    string,
    { from: ColorFrom; tableId: string; colors: typeof objectColors }
  >();
  for (const objectColor of objectColors) {
    const from = getActiveFrom(objectColor.colorConfig);
    // the column is in the configured table, or else in the object's own; a
    // table that is not in the project colors nothing; mirrors
    // WebGLRendererBase.createTableLoader
    const tableId = from?.table ?? objectColor.objectTableId;
    if (
      from === null ||
      tableId === undefined ||
      !tables.some(({ id }) => id === tableId) ||
      isConstantlyHidden(objectColor.visibilityConfig)
    ) {
      continue;
    }
    const [rangeMin, rangeMax] = from.range ?? [null, null];
    const key = JSON.stringify([
      tableId,
      rangeMin,
      rangeMax,
      from.palette,
      from.column,
    ]);
    const colorbar = colorbars.get(key);
    if (colorbar === undefined) {
      colorbars.set(key, { from, tableId, colors: [objectColor] });
    } else {
      colorbar.colors.push(objectColor);
      // a colorbar names a table only if all its colors name the same one
      if (from.table !== colorbar.from.table) {
        colorbar.from = { ...colorbar.from, table: undefined };
      }
    }
  }

  return [...colorbars].map(([key, { from, tableId, colors }]) => {
    const objectNames = new Map<string, string>();
    for (const { objectKey, objectName, part } of colors) {
      // a shape whose fill and stroke both use the colorbar gets its plain name
      objectNames.set(
        objectKey,
        part === null || objectNames.has(objectKey)
          ? objectName
          : `${objectName} (${part})`,
      );
    }
    return { key, name: [...objectNames.values()].join(", "), from, tableId };
  });
}
