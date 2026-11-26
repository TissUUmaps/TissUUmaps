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

/** ID of the color palette of a newly colored categorical column */
const defaultCategoricalColorPalette = defaultLabelColorPalette;

/** ID of the color palette of a newly colored numerical column */
const defaultContinuousColorPalette = "viridis";

/**
 * What a property follows, as the settings widget shows it
 *
 * `none` is the property's default without a column. `custom` stands for
 * configurations the widget does not edit, such as a single color where
 * random colors are the default, or a column of another table; they are
 * edited in the advanced settings, and `label` describes them.
 */
export type StyleBy =
  | { kind: "none" }
  | { kind: "column"; column: string; columnType: TableColumnType }
  | { kind: "custom"; label: string };

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
 * Reads what a configuration follows
 *
 * Following a column of the object's own table reads as that column,
 * categorical or numerical; the property's default without a column reads as
 * none, so that the widget and the map agree; anything
 * else is custom.
 *
 * @param config - The configuration
 * @param tableId - The ID of the object's own table, if it has one
 * @param options - `defaultConfig` is the property's default configuration,
 * whose source (a constant if none is given) reads as none, e.g. random
 * colors for labels
 * @returns What the configuration follows
 */
export function readStyleBy<TConfig extends Config<string>>(
  config: TConfig,
  tableId: string | null,
  options?: { defaultConfig?: TConfig },
): StyleBy {
  const { defaultConfig } = options ?? {};
  const noneSource =
    defaultConfig !== undefined
      ? getActiveConfigSource(defaultConfig)
      : "constant";
  const source = getActiveConfigSource(config) ?? "constant";
  switch (source) {
    case "groupBy":
      return isGroupByConfig(config) &&
        isOwnTableColumn(config.groupBy, tableId)
        ? {
            kind: "column",
            column: config.groupBy.column,
            columnType: "categorical",
          }
        : { kind: "custom", label: "Custom (see Advanced)" };
    case "from":
      return isFromConfig(config) && isOwnTableColumn(config.from, tableId)
        ? {
            kind: "column",
            column: config.from.column,
            columnType: "numerical",
          }
        : { kind: "custom", label: "Custom (see Advanced)" };
    default:
      return source === noneSource
        ? { kind: "none" }
        : {
            kind: "custom",
            label: source === "constant" ? "Single color" : "Random colors",
          };
  }
}

/**
 * Returns what the settings widget follows: the column of the first property
 * that follows one, or else what the first property follows
 *
 * @param styleBys - What the properties follow, in the order of the widget
 * @returns What the settings widget follows
 */
export function findStyleBy(styleBys: StyleBy[]): StyleBy {
  return (
    styleBys.find((styleBy) => styleBy.kind === "column") ??
    styleBys[0] ?? { kind: "none" }
  );
}

/**
 * @param styleBy - What a property follows
 * @param other - What the settings widget follows
 * @returns Whether both follow the same column as the same column type
 */
export function isSameColumnStyleBy(styleBy: StyleBy, other: StyleBy): boolean {
  return (
    styleBy.kind === "column" &&
    other.kind === "column" &&
    styleBy.column === other.column &&
    styleBy.columnType === other.columnType
  );
}

/** What a property of the settings widget does when the widget is changed */
export type FollowState = {
  /** The column that the property follows from now on, if any */
  column: Extract<StyleBy, { kind: "column" }> | null;

  /** Whether the property followed the column of the widget until now */
  wasFollowing: boolean;
};

/**
 * Returns what the properties of the settings widget do when the widget is
 * changed
 *
 * @param previousStyleBys - What each property followed until now, by name,
 * in the order of the widget
 * @param styleBy - What the widget follows from now on
 * @param propertyNames - The names of the properties that follow it
 * @returns What a property does, by name
 */
export function getFollowState<TName extends string>(
  previousStyleBys: Record<TName, StyleBy>,
  styleBy: StyleBy,
  propertyNames: string[],
): (name: TName) => FollowState {
  const previousStyleBy = findStyleBy(Object.values<StyleBy>(previousStyleBys));
  const column = styleBy.kind === "column" ? styleBy : null;
  return (name) => ({
    column: propertyNames.includes(name) ? column : null,
    wasFollowing: isSameColumnStyleBy(previousStyleBys[name], previousStyleBy),
  });
}

/**
 * Returns the configuration of a property after the settings widget changed: it
 * follows its new column, or returns to its default if it stops following
 * the widget's column; anything else is left unchanged
 *
 * @param config - The configuration of the property
 * @param state - What the property does (see {@link getFollowState})
 * @param options - `withColumn` and `withoutColumn` return the configuration
 * that follows a column, or none
 * @returns The configuration of the property
 */
export function followColumn<TConfig>(
  config: TConfig,
  state: FollowState,
  options: {
    withColumn: (column: Extract<StyleBy, { kind: "column" }>) => TConfig;
    withoutColumn: () => TConfig;
  },
): TConfig {
  if (state.column !== null) {
    return options.withColumn(state.column);
  }
  return state.wasFollowing ? options.withoutColumn() : config;
}

/** A property that can follow the column of the settings widget */
export type StyleProperty = {
  /** Tells the property apart from the other properties of the object */
  name: string;

  /**
   * The column types the property can follow, e.g. only categorical columns
   * for markers
   */
  columnTypes: TableColumnType[];

  /** Whether the property follows the column of the settings widget */
  isOn: boolean;
};

/**
 * The properties of a part of an object, e.g. the color, opacity and
 * visibility of the fill of shapes, toggled together
 *
 * The part follows the column while any of its properties does.
 */
export type StylePart = {
  /** Tells the part apart from the properties of the object */
  name: string;

  /** The properties of the part; the first one follows a newly toggled part */
  properties: [StyleProperty, ...StyleProperty[]];
};

/**
 * @param item - A property or a part
 * @returns The properties of the item: the property, or those of the part
 */
function getStyleProperties(item: StyleProperty | StylePart): StyleProperty[] {
  return "properties" in item ? item.properties : [item];
}

/**
 * @param item - A property or a part
 * @returns Whether the item follows the column
 */
export function isStyleItemOn(item: StyleProperty | StylePart): boolean {
  return getStyleProperties(item).some(({ isOn }) => isOn);
}

/**
 * @param item - A property or a part
 * @param columnType - A column type
 * @returns Whether the item can follow a column of the type
 */
export function isStyleItemSupported(
  item: StyleProperty | StylePart,
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
  items: (StyleProperty | StylePart)[],
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
  items: (StyleProperty | StylePart)[],
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
  items: (StyleProperty | StylePart)[],
  part: StylePart,
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
export function withoutColumn<TConfig extends Config<string>>(
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
