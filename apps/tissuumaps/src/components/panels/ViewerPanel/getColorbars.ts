import {
  type ColorConfig,
  type FromConfig,
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

type ObjectColor = {
  objectKey: string;
  objectName: string;
  part: "fill" | "stroke" | null;
  colorConfig: ColorConfig;
  visibilityConfig: VisibilityConfig;
  objectTableId: string | undefined;
};

/**
 * Lists the colorbars of the shown objects whose colors come from a table
 * column
 *
 * Colors with the same column, palette and range share one colorbar. Without
 * a range, they also need the same table, whose column sets the value range.
 * A colorbar is named after the objects that use it. The fill or stroke of
 * shapes is named unless both use the colorbar.
 *
 * @param state - The layers and the objects of the project
 * @returns One colorbar per color scale, with a unique key, the name to show,
 * the active "from" source and the ID of the table that holds the column
 */
export function getColorbars(state: {
  layers: Layer[];
  labels: Labels[];
  points: Points[];
  shapes: Shapes[];
}): { key: string; name: string; from: ColorFrom; tableId: string }[] {
  const { layers, labels, points, shapes } = state;
  const objectColors: ObjectColor[] = [
    ...labels
      // without a table of their own, the renderer draws labels in default
      // colors, even if the color column is in another table
      .filter(
        (labels) =>
          labels.visibility &&
          !isOnHiddenLayer(labels.layer, layers) &&
          labels.dataSource.table !== undefined,
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
    { from: ColorFrom; tableId: string; colors: ObjectColor[] }
  >();
  for (const objectColor of objectColors) {
    const from = getActiveFrom(objectColor.colorConfig);
    const tableId = from?.table ?? objectColor.objectTableId;
    // without a table, no renderer colors the items from a column
    if (
      from === null ||
      tableId === undefined ||
      isConstantlyHidden(objectColor.visibilityConfig)
    ) {
      continue;
    }
    // with a range, the colorbar does not depend on the table
    const key = `${from.range === undefined ? tableId : from.range.join(",")}:${from.palette}:${from.column}`;
    const colorbar = colorbars.get(key);
    if (colorbar === undefined) {
      colorbars.set(key, { from, tableId, colors: [objectColor] });
    } else {
      colorbar.colors.push(objectColor);
      // a colorbar shared by several tables names none of them
      if (tableId !== colorbar.tableId) {
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
