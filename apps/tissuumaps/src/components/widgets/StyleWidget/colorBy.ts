import {
  type ColorConfig,
  ConfigUtils,
  type TableColumnRef,
  type VisibilityConfig,
  defaultLabelColorPalette,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
  isRandomConfig,
} from "@tissuumaps/core";

/** ID of the color palette of a newly colored categorical column */
const defaultCategoricalColorPalette = defaultLabelColorPalette;

/** ID of the color palette of a newly colored numeric column */
const defaultContinuousColorPalette = "viridis";

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
 * coloring without a column (or a hidden fill) reads as none, so that the
 * widget and the map agree; anything else is custom.
 *
 * @param config - The color configuration
 * @param tableId - The ID of the object's own table, if it has one
 * @param options - `noneSource` is the source of the property's default
 * coloring without a column (`"constant"` by default, `"random"` for labels);
 * `isHidden` tells whether what the configuration colors is hidden, e.g. the
 * fill of shapes
 * @returns What the configuration colors by
 */
export function readColorBy(
  config: ColorConfig,
  tableId: string | null,
  options?: { noneSource?: "constant" | "random"; isHidden?: boolean },
): ColorBy {
  const { noneSource = "constant", isHidden = false } = options ?? {};
  if (isHidden) {
    return { kind: "none" };
  }
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
 * Returns the color configuration that colors by a column of the object's own
 * table
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
export function withColorColumn(
  config: ColorConfig,
  column: string,
  encoding: ColorByEncoding,
  tableId: string | null,
): ColorConfig {
  const isSameColumn = (ref: TableColumnRef) =>
    ConfigUtils.isSameTableColumn(ref, { column }, tableId ?? undefined);
  if (encoding === "categories") {
    const groupBy = isGroupByConfig(config) ? config.groupBy : undefined;
    return {
      ...config,
      source: "groupBy",
      groupBy:
        groupBy !== undefined && isSameColumn(groupBy)
          ? groupBy
          : {
              column,
              palette: groupBy?.palette ?? defaultCategoricalColorPalette,
              map: undefined,
            },
    };
  }
  const from = isFromConfig(config) ? config.from : undefined;
  return {
    ...config,
    source: "from",
    from:
      from !== undefined && isSameColumn(from)
        ? from
        : {
            column,
            palette: from?.palette ?? defaultContinuousColorPalette,
          },
  };
}

/**
 * Returns the color configuration that colors by no column, i.e. by the
 * default source of the property
 *
 * The branch of the default source is kept if the configuration has one, and
 * taken from the default configuration otherwise.
 *
 * @param config - The color configuration to update
 * @param defaultConfig - The default color configuration of the property
 * @returns The updated color configuration
 * @throws Error if the default configuration is not a constant or random one
 */
export function withoutColorColumn(
  config: ColorConfig,
  defaultConfig: ColorConfig,
): ColorConfig {
  switch (getActiveConfigSource(defaultConfig)) {
    case "constant":
      if (isConstantConfig(config)) {
        return { ...config, source: "constant" };
      }
      if (isConstantConfig(defaultConfig)) {
        return {
          ...config,
          source: "constant",
          constant: structuredClone(defaultConfig.constant),
        };
      }
      break;
    case "random":
      if (isRandomConfig(config)) {
        return { ...config, source: "random" };
      }
      if (isRandomConfig(defaultConfig)) {
        return {
          ...config,
          source: "random",
          random: structuredClone(defaultConfig.random),
        };
      }
      break;
  }
  throw new Error(
    "The default color configuration is no constant or random one",
  );
}

/**
 * Shows what a color configuration colors, e.g. the fill of shapes, if it is
 * hidden by a constant visibility
 *
 * @param config - The visibility configuration
 * @returns The visibility configuration, visible if it was constantly hidden
 */
export function showIfHidden(config: VisibilityConfig): VisibilityConfig {
  return isHiddenByConstant(config)
    ? { ...config, source: "constant", constant: { value: true } }
    : config;
}

/**
 * Hides what a color configuration colors, e.g. the fill of shapes
 *
 * @param config - The visibility configuration
 * @returns The visibility configuration, constantly hidden; its other branches
 * are kept
 */
export function hide(config: VisibilityConfig): VisibilityConfig {
  return { ...config, source: "constant", constant: { value: false } };
}

/**
 * @param config - A visibility configuration
 * @returns Whether the configuration constantly hides what it applies to
 */
export function isHiddenByConstant(config: VisibilityConfig): boolean {
  return (
    getActiveConfigSource(config) === "constant" &&
    isConstantConfig<boolean>(config) &&
    !config.constant.value
  );
}
