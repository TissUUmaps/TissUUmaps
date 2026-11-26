import {
  type ColorConfig,
  type Config,
  ConfigUtils,
  type CoordinateSpace,
  type MarkerConfig,
  type TableColumnRef,
  type TableColumnType,
  defaultLabelColorPalette,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
  isRandomConfig,
} from "@tissuumaps/core";

import type {
  SettingsWidgetItem,
  SettingsWidgetPart,
  SettingsWidgetProperty,
} from ".";

/** ID of the color palette of a newly colored categorical column */
const defaultCategoricalColorPalette = defaultLabelColorPalette;

/** ID of the color palette of a newly colored numerical column */
const defaultContinuousColorPalette = "viridis";

/** The column of the object's own table that a property follows */
export type StyleBy = { column: string; columnType: TableColumnType };

/**
 * Returns whether a table column reference names a column of the object's
 * own table, the only table the settings widget styles by
 *
 * @param ref - The table column reference
 * @param tableId - The ID of the object's own table, if it has one
 * @returns Whether the column is one of the own table
 */
function isOwnTableColumn(
  ref: TableColumnRef,
  tableId: string | null,
): boolean {
  return tableId !== null && (ref.table === undefined || ref.table === tableId);
}

/**
 * Reads the column that a configuration follows: a column of the object's
 * own table, categorical or numerical
 *
 * A column the property cannot follow as that type is not followed, e.g. the
 * values per category of a size from the group table.
 *
 * @param config - The configuration
 * @param tableId - The ID of the object's own table, if it has one
 * @param columnTypes - The column types the property can follow
 * @returns The column, or `null` if the configuration follows none
 */
export function readStyleBy<TConfig extends Config<string>>(
  config: TConfig,
  tableId: string | null,
  columnTypes: TableColumnType[],
): StyleBy | null {
  switch (getActiveConfigSource(config)) {
    case "groupBy":
      return columnTypes.includes("categorical") &&
        isGroupByConfig(config) &&
        isOwnTableColumn(config.groupBy, tableId)
        ? { column: config.groupBy.column, columnType: "categorical" }
        : null;
    case "from":
      return columnTypes.includes("numerical") &&
        isFromConfig(config) &&
        isOwnTableColumn(config.from, tableId)
        ? { column: config.from.column, columnType: "numerical" }
        : null;
    default:
      return null;
  }
}

/**
 * Describes a color that follows no column of the object's own table, as the
 * settings widget shows it while no column is picked
 *
 * @param config - The color configuration
 * @param defaultConfig - The default color configuration of the object
 * @returns "None" for the default single color, else what the color is, e.g.
 * "Random colors"
 */
export function describeColorWithoutColumn(
  config: ColorConfig,
  defaultConfig: ColorConfig,
): string {
  switch (getActiveConfigSource(config) ?? "constant") {
    case "groupBy":
    case "from":
      return "Custom (see Advanced)";
    case "random":
      return "Random colors";
    default:
      return getActiveConfigSource(defaultConfig) === "constant"
        ? "None"
        : "Single color";
  }
}

/**
 * Returns the column that the settings widget follows: the column of the
 * first property that follows one
 *
 * @param styleBys - The columns the properties follow, in the order of the
 * widget
 * @returns The column, or `null` if no property follows one
 */
export function findStyleBy(styleBys: (StyleBy | null)[]): StyleBy | null {
  return styleBys.find((styleBy) => styleBy !== null) ?? null;
}

/**
 * @param styleBy - The column a property follows, if any
 * @param other - The column the settings widget follows, if any
 * @returns Whether both follow the same column as the same column type
 */
export function isSameColumnStyleBy(
  styleBy: StyleBy | null,
  other: StyleBy | null,
): boolean {
  return (
    styleBy !== null &&
    other !== null &&
    styleBy.column === other.column &&
    styleBy.columnType === other.columnType
  );
}

/**
 * Returns how the properties of the settings widget follow its change: a
 * named property follows the new column, a property that stops following the
 * widget's column returns to its default, and any other is left unchanged
 *
 * @param previousStyleBys - The column each property followed until now, if
 * any, by name, in the order of the widget
 * @param styleBy - The column the widget follows from now on, if any
 * @param propertyNames - The names of the properties that follow it
 * @returns A function that returns the new configuration of a property, given
 * its name, configuration, default configuration, and `withColumn`, which
 * returns the configuration that follows a column
 */
export function getFollower<TName extends string>(
  previousStyleBys: Record<TName, StyleBy | null>,
  styleBy: StyleBy | null,
  propertyNames: string[],
) {
  const previousStyleBy = findStyleBy(
    Object.values<StyleBy | null>(previousStyleBys),
  );
  return <TConfig extends Config<string>>(
    name: TName,
    config: TConfig,
    defaultConfig: TConfig,
    withColumn: (styleBy: StyleBy) => TConfig,
  ): TConfig => {
    if (styleBy !== null && propertyNames.includes(name)) {
      return withColumn(styleBy);
    }
    return isSameColumnStyleBy(previousStyleBys[name], previousStyleBy)
      ? withoutColumn(config, defaultConfig)
      : config;
  };
}

/**
 * @param item - A property or a part
 * @returns The properties of the item: the property, or those of the part
 */
function getStyleProperties(
  item: SettingsWidgetItem,
): SettingsWidgetProperty[] {
  return "properties" in item ? item.properties : [item];
}

/**
 * @param item - A property or a part
 * @returns Whether the item follows the column
 */
export function isStyleItemOn(item: SettingsWidgetItem): boolean {
  return getStyleProperties(item).some(({ isOn }) => isOn);
}

/**
 * @param item - A property or a part
 * @param columnType - A column type
 * @returns Whether the item can follow a column of the type
 */
export function isStyleItemSupported(
  item: SettingsWidgetItem,
  columnType: TableColumnType,
): boolean {
  return getStyleProperties(item).some(({ columnTypes }) =>
    columnTypes.includes(columnType),
  );
}

/**
 * Returns the properties that follow a newly picked column or column type:
 * those that follow the column now and can follow the type; if none can, the
 * first property that can
 *
 * @param items - The properties and parts, in the order of their toggles
 * @param columnType - The new column type
 * @returns The names of the properties that follow
 */
export function getFollowingNames(
  items: SettingsWidgetItem[],
  columnType: TableColumnType,
): string[] {
  const supported = items
    .flatMap(getStyleProperties)
    .filter(({ columnTypes }) => columnTypes.includes(columnType));
  const following = supported.filter(({ isOn }) => isOn);
  return (following.length > 0 ? following : supported.slice(0, 1)).map(
    ({ name }) => name,
  );
}

/**
 * Returns the properties that follow the column once the toggles of the
 * properties and parts are set: an item kept on keeps its properties, an
 * item newly toggled on starts with its first property that can follow the
 * column type
 *
 * @param items - The properties and parts, in the order of their toggles
 * @param toggledNames - The names of the items toggled on
 * @param columnType - The column type
 * @returns The names of the properties that follow, or `null` if no item is
 * toggled on, since the column always styles something
 */
export function getToggledNames(
  items: SettingsWidgetItem[],
  toggledNames: string[],
  columnType: TableColumnType,
): string[] | null {
  const names = items.flatMap((item) => {
    if (!toggledNames.includes(item.name)) {
      return [];
    }
    const properties = getStyleProperties(item);
    return (
      isStyleItemOn(item)
        ? properties.filter(({ isOn }) => isOn)
        : properties
            .filter(({ columnTypes }) => columnTypes.includes(columnType))
            .slice(0, 1)
    ).map(({ name }) => name);
  });
  return names.length > 0 ? names : null;
}

/**
 * Returns the properties that follow the column once the toggles of a part's
 * properties are set
 *
 * @param items - The properties and parts, in the order of their toggles
 * @param part - The part
 * @param toggledNames - The names of the part's properties toggled on
 * @returns The names of the properties that follow, or `null` if none would,
 * since the column always styles something
 */
export function getPartToggledNames(
  items: SettingsWidgetItem[],
  part: SettingsWidgetPart,
  toggledNames: string[],
): string[] | null {
  const names = items.flatMap((item) =>
    (item === part
      ? part.properties.filter(({ name }) => toggledNames.includes(name))
      : getStyleProperties(item).filter(({ isOn }) => isOn)
    ).map(({ name }) => name),
  );
  return names.length > 0 ? names : null;
}

/**
 * Returns the color configuration that follows a column of the object's own
 * table
 *
 * The branch of the column type is kept if it already names the column, so
 * that switching between categorical and numerical keeps the palette, color
 * map and range of each. For another column, the branch keeps only its
 * palette (the default palette if there is no branch yet): the value range
 * and the color map belong to the values of the previous column.
 *
 * @param config - The color configuration to update
 * @param column - The column to follow
 * @param columnType - How the column's values are mapped to colors
 * @param tableId - The ID of the object's own table
 * @returns The updated color configuration
 */
export function withColorColumn(
  config: ColorConfig,
  column: string,
  columnType: TableColumnType,
  tableId: string | null,
): ColorConfig {
  const isSameColumn = (ref: TableColumnRef) =>
    ConfigUtils.isSameTableColumn(ref, { column }, tableId ?? undefined);
  if (columnType === "categorical") {
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
 * Returns the marker configuration that gives each category of a column of
 * the object's own table its own marker
 *
 * The grouping is kept if it already names the column, so that turning the
 * marker on again keeps the edited markers of the categories. A new grouping
 * takes the markers from the palette.
 *
 * @param config - The marker configuration to update
 * @param column - The column to group the marker by
 * @param tableId - The ID of the object's own table
 * @returns The updated marker configuration
 */
export function withMarkerColumn(
  config: MarkerConfig,
  column: string,
  tableId: string | null,
): MarkerConfig {
  return {
    ...config,
    source: "groupBy",
    groupBy:
      isGroupByConfig<false>(config) &&
      ConfigUtils.isSameTableColumn(
        config.groupBy,
        { column },
        tableId ?? undefined,
      )
        ? config.groupBy
        : { column, map: undefined },
  };
}

/**
 * Returns the configuration that takes its values from a numerical column of
 * the object's own table
 *
 * The column branch is kept if it already names the column. A new one takes
 * the unit of the active source, so that sizes keep their unit.
 *
 * @param config - The configuration to update
 * @param column - The column to take the values from
 * @param tableId - The ID of the object's own table
 * @returns The updated configuration
 */
export function withFromColumn<TConfig extends Config<string>>(
  config: TConfig,
  column: string,
  tableId: string | null,
): TConfig {
  const from = isFromConfig<{ unit?: CoordinateSpace }>(config)
    ? config.from
    : undefined;
  const unit = ConfigUtils.getUnit(config);
  return {
    ...config,
    source: "from",
    from:
      from !== undefined &&
      ConfigUtils.isSameTableColumn(from, { column }, tableId ?? undefined)
        ? from
        : { column, ...(unit !== undefined && { unit }) },
  };
}

/**
 * Returns the configuration that follows no column, i.e. the source of the
 * property's default: a constant value, or random values (e.g. the colors of
 * labels)
 *
 * The branch of that source is kept if the configuration has one, and taken
 * from the default configuration otherwise; the other branches are kept
 * inactive.
 *
 * @param config - The configuration to update
 * @param defaultConfig - The default configuration of the property
 * @returns The updated configuration
 * @throws Error if the default configuration is not a constant or random one
 */
function withoutColumn<TConfig extends Config<string>>(
  config: TConfig,
  defaultConfig: TConfig,
): TConfig {
  if (getActiveConfigSource(defaultConfig) === "random") {
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
  } else {
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
  }
  throw new Error("The default configuration must be a constant or random one");
}
