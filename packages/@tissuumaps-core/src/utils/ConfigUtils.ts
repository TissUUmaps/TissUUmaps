import {
  type Config,
  type GroupByConfig,
  type TableColumnRef,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
} from "../model/configs";
import type { CoordinateSpace, GroupValueMap } from "../model/primitives";

/** Utility methods for property configurations */
export class ConfigUtils {
  /**
   * Returns the group-to-value map that a configuration resolves its values
   * from, if any
   *
   * Only an active `groupBy` source (see {@link getActiveConfigSource}) with a
   * map ID resolves from a map. The map is returned as found in `maps`, rather
   * than copied: maps are never mutated, as an edit replaces the map, so
   * callers can detect an edit by comparing the returned maps by identity.
   *
   * @param config - The configuration
   * @param maps - The project-global maps to look the referenced map up in
   * @returns The map, or `undefined` if the configuration does not resolve
   * from a map, or if the map it references does not exist
   */
  static findGroupByMap<TValue>(
    config: Config<string>,
    maps: GroupValueMap<TValue>[],
  ): GroupValueMap<TValue> | undefined {
    if (
      getActiveConfigSource(config) === "groupBy" &&
      isGroupByConfig<false>(config) &&
      config.groupBy.map !== undefined
    ) {
      return maps.find((map) => map.id === config.groupBy.map);
    }
    return undefined;
  }

  /**
   * Returns the IDs of the maps that configurations refer to
   *
   * A configuration refers to its map whatever its active source, as switching
   * back to `groupBy` uses the map again.
   *
   * @param configs - The configurations
   * @returns The IDs of the referenced maps
   */
  static getGroupByMapIds(configs: Config<string>[]): Set<string> {
    const mapIds = new Set<string>();
    for (const config of configs) {
      if (isGroupByConfig<false>(config) && config.groupBy.map !== undefined) {
        mapIds.add(config.groupBy.map);
      }
    }
    return mapIds;
  }

  /**
   * Creates the function that returns the value a group-by configuration
   * assigns to a group
   *
   * With a map, a group takes its value in the map, else the map's default,
   * else `defaultValue`; a map that does not exist gives every group
   * `defaultValue`. Without a map, the groups take the palette values in
   * order, cycling through the palette if there are more groups than values;
   * a group that is not listed, or every group if there is no palette, takes
   * `defaultValue`.
   *
   * @param config - The group-by configuration
   * @param map - The map the configuration refers to (see
   * {@link findGroupByMap}), or `undefined` if it refers to none or the map
   * does not exist
   * @param defaultValue - The value of a group that nothing assigns one to
   * @param palette - The values to assign to the groups if there is no map
   * @param groups - All groups of the column the configuration groups by, in
   * the order they take the palette values (see `TableUtils.loadGroupCounts`)
   * @returns The value of a group, by group name (the cell value as a string)
   */
  static createGroupValueGetter<TValue>(
    config: GroupByConfig<false>,
    map: GroupValueMap<TValue> | undefined,
    defaultValue: TValue,
    palette?: readonly TValue[],
    groups?: Iterable<string>,
  ): (group: string) => TValue {
    if (config.groupBy.map !== undefined) {
      if (map === undefined) {
        return () => defaultValue;
      }
      const values = new Map(Object.entries(map.values));
      const mapDefault = map.default ?? defaultValue;
      return (group) => values.get(group) ?? mapDefault;
    }
    if (palette === undefined || palette.length === 0 || groups === undefined) {
      return () => defaultValue;
    }
    const paletteValues = new Map<string, TValue>();
    for (const group of groups) {
      if (!paletteValues.has(group)) {
        paletteValues.set(group, palette[paletteValues.size % palette.length]!);
      }
    }
    return (group) => paletteValues.get(group) ?? defaultValue;
  }

  /**
   * Returns the table column a configuration groups by
   *
   * @param config - The configuration
   * @returns The column, or `undefined` if `groupBy` is not the active source
   */
  static getGroupByColumn(config: Config<string>): TableColumnRef | undefined {
    return getActiveConfigSource(config) === "groupBy" &&
      isGroupByConfig(config)
      ? config.groupBy
      : undefined;
  }

  /**
   * Points a configuration at a map, grouping by a column
   *
   * The configuration keeps its other sources and the extra fields of its
   * `groupBy` specification (e.g. a palette), so that switching back to them
   * restores them. The unit of the active source, which only sizes have, is
   * carried over, so that the values keep their scale. A configuration that
   * already groups by the column with the map is returned as is, so that
   * updating an object with it changes nothing.
   *
   * @param config - The configuration
   * @param groupBy - The categorical table column to group by
   * @param mapId - ID of the project-global map to take the group values from
   * @returns The configuration, with `groupBy` as its active source
   */
  static withGroupByMap(
    config: Config<string>,
    groupBy: TableColumnRef,
    mapId: string,
  ): GroupByConfig<true> {
    const currentGroupBy = ConfigUtils.getGroupByColumn(config);
    if (
      currentGroupBy !== undefined &&
      ConfigUtils.isSameTableColumn(currentGroupBy, groupBy) &&
      isGroupByConfig<true>(config) &&
      config.groupBy.map === mapId
    ) {
      return config;
    }
    const prevGroupBy = isGroupByConfig<false, { unit?: CoordinateSpace }>(
      config,
    )
      ? config.groupBy
      : undefined;
    const unit = ConfigUtils.getUnit(config);
    return {
      ...config,
      source: "groupBy",
      groupBy: {
        ...prevGroupBy,
        ...((unit !== undefined || prevGroupBy?.unit !== undefined) && {
          unit,
        }),
        table: groupBy.table,
        column: groupBy.column,
        map: mapId,
      },
    };
  }

  /**
   * Returns the unit of the active source of a configuration
   *
   * @param config - The configuration, of which only sizes have a unit
   * @returns The unit, or `undefined` if the active source has none
   */
  static getUnit(config: Config<string>): CoordinateSpace | undefined {
    type Unit = { unit?: CoordinateSpace };
    switch (getActiveConfigSource(config)) {
      case "constant":
        return isConstantConfig<unknown, Unit>(config)
          ? config.constant.unit
          : undefined;
      case "from":
        return isFromConfig<Unit>(config) ? config.from.unit : undefined;
      case "groupBy":
        return isGroupByConfig<false, Unit>(config)
          ? config.groupBy.unit
          : undefined;
      default:
        return undefined;
    }
  }

  /**
   * Determines whether two table column references point to the same column
   *
   * References without a table point to the column of `defaultTable`.
   *
   * @param tableColumnRef - The table column reference
   * @param otherTableColumnRef - The table column reference to compare with
   * @param defaultTable - ID of the table of references without a table
   * @returns Whether both references point to the same column
   */
  static isSameTableColumn(
    tableColumnRef: TableColumnRef,
    otherTableColumnRef: TableColumnRef,
    defaultTable?: string,
  ): boolean {
    return (
      (tableColumnRef.table ?? defaultTable) ===
        (otherTableColumnRef.table ?? defaultTable) &&
      tableColumnRef.column === otherTableColumnRef.column
    );
  }
}
