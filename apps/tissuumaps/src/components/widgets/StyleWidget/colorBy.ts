import {
  type ColorConfig,
  type TableColumnRef,
  defaultCategoricalColorPalette,
  defaultContinuousColorPalette,
  getActiveConfigSource,
  isFromConfig,
  isGroupByConfig,
} from "@tissuumaps/core";

/** How the values of a column are mapped to colors */
export type ColorByEncoding = "categories" | "gradient";

/**
 * What a color configuration colors by, as the style widget shows it
 *
 * `none` is the property's default coloring without a column. `custom`
 * stands for configurations the widget does not edit, such as a single color
 * where random colors are the default, or a column of another table; they are
 * edited in the advanced settings, and `label` describes them.
 */
export type ColorBy =
  | { kind: "none" }
  | { kind: "column"; column: string; encoding: ColorByEncoding }
  | { kind: "custom"; label: string };

/**
 * Returns whether a table column reference names a column of the object's
 * own table, the only table the style widget colors by
 *
 * @param ref - The table column reference
 * @param tableId - The ID of the object's own table, if it has one
 * @returns Whether the column is one of the own table
 */
export function isOwnTableColumn(
  ref: TableColumnRef,
  tableId: string | null,
): boolean {
  return tableId !== null && (ref.table === undefined || ref.table === tableId);
}

/**
 * Reads what a color configuration colors by
 *
 * Coloring by a column of the object's own table reads as that column,
 * grouped into categories or mapped to a gradient; the property's default
 * coloring without a column reads as none; anything else is custom.
 *
 * @param config - The color configuration
 * @param tableId - The ID of the object's own table, if it has one
 * @param options - `noneSource` is the source of the property's default
 * coloring without a column (`"constant"` by default, `"random"` for labels)
 * @returns What the configuration colors by
 */
export function readColorBy(
  config: ColorConfig,
  tableId: string | null,
  options?: { noneSource?: "constant" | "random" },
): ColorBy {
  const { noneSource = "constant" } = options ?? {};
  const source = getActiveConfigSource(config) ?? "constant";
  switch (source) {
    case "groupBy":
      return isGroupByConfig(config) &&
        isOwnTableColumn(config.groupBy, tableId)
        ? {
            kind: "column",
            column: config.groupBy.column,
            encoding: "categories",
          }
        : { kind: "custom", label: "Custom (see advanced settings)" };
    case "from":
      return isFromConfig(config) && isOwnTableColumn(config.from, tableId)
        ? { kind: "column", column: config.from.column, encoding: "gradient" }
        : { kind: "custom", label: "Custom (see advanced settings)" };
    case "constant":
    case "random":
      return source === noneSource
        ? { kind: "none" }
        : {
            kind: "custom",
            label: source === "constant" ? "Single color" : "Random colors",
          };
  }
}

/**
 * Colors by a column of the object's own table
 *
 * The branch of the encoding is kept if it already names the column, so that
 * switching between categories and a gradient keeps the palette, color map
 * and range of each. For another column, the branch keeps only its palette
 * (the default palette if there is no branch yet): the value range and the
 * color map belong to the values of the previous column.
 *
 * @param config - The color configuration to update
 * @param column - The column to color by
 * @param encoding - How the column's values are mapped to colors
 * @param tableId - The ID of the object's own table
 * @returns The updated color configuration
 */
export function colorByColumn(
  config: ColorConfig,
  column: string,
  encoding: ColorByEncoding,
  tableId: string | null,
): ColorConfig {
  if (encoding === "categories") {
    const groupBy = isGroupByConfig<false, { palette?: string }>(config)
      ? config.groupBy
      : undefined;
    return {
      ...config,
      source: "groupBy",
      groupBy:
        groupBy !== undefined &&
        isOwnTableColumn(groupBy, tableId) &&
        groupBy.column === column
          ? groupBy
          : {
              column,
              palette: groupBy?.palette ?? defaultCategoricalColorPalette,
              map: undefined,
            },
    };
  }
  const from = isFromConfig<{ palette: string; range?: [number, number] }>(
    config,
  )
    ? config.from
    : undefined;
  return {
    ...config,
    source: "from",
    from:
      from !== undefined &&
      isOwnTableColumn(from, tableId) &&
      from.column === column
        ? from
        : {
            column,
            palette: from?.palette ?? defaultContinuousColorPalette,
          },
  };
}

/**
 * Stops coloring by a column, returning to the default source of the property
 *
 * The branch of the default source is kept if the configuration has one.
 *
 * @param config - The color configuration to update
 * @param defaultConfig - The default color configuration of the property
 * @returns The updated color configuration
 */
export function colorByNone(
  config: ColorConfig,
  defaultConfig: ColorConfig,
): ColorConfig {
  return {
    ...structuredClone(defaultConfig),
    ...config,
    source: getActiveConfigSource(defaultConfig),
  } as ColorConfig;
}
