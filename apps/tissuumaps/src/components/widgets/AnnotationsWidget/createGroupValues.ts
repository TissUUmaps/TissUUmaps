import {
  type Config,
  ConfigUtils,
  getActiveConfigSource,
  isConstantConfig,
} from "@tissuumaps/core";

import { isGroupedByColumn } from "./isGroupedByColumn";
import type { GroupProperty } from "./useGroupColumn";
import type { GroupTableState } from "./useGroupTable";

/** The value of every group of a property, and how to set them */
export type GroupValues<TValue> = {
  getValue: (group: string) => TValue;
  setValues: (values: { [group: string]: TValue }) => void;

  /** Whether the values are grayed out, as setting them changes the source */
  isInactive: boolean;
};

/**
 * Returns the value of every group of a property, and how to set them
 *
 * A property grouped by the table's column shows its map or palette values,
 * and a constant one its value for every group. Any other property shows its
 * default, grayed out. Setting values updates the map that the property is
 * grouped by, or creates one, and groups the property by the table's column.
 *
 * @param groupTable - The state of the group table
 * @param property - The property
 * @returns The values, or `undefined` while the table has no column or groups
 */
export function createGroupValues<TValue, TConfig extends Config<string>>(
  groupTable: GroupTableState,
  property: GroupProperty<TValue, TConfig>,
): GroupValues<TValue> | undefined {
  const { objectName, groupBy, groupCounts } = groupTable;
  if (groupBy === null || groupCounts === null) {
    return undefined;
  }
  const { name, config, default: defaultValue, adapter } = property;
  const isGrouped = isGroupedByColumn(config, groupTable);
  const map = isGrouped
    ? ConfigUtils.findGroupByMap(config, adapter.maps)
    : undefined;

  let getValue: (group: string) => TValue;
  let isInactive = false;
  if (isGrouped) {
    getValue = ConfigUtils.createGroupValueGetter(
      config,
      map,
      defaultValue,
      adapter.getPalette?.(config),
    );
  } else if (
    getActiveConfigSource(config) === "constant" &&
    isConstantConfig<TValue>(config)
  ) {
    getValue = () => config.constant.value;
  } else {
    getValue = () => defaultValue;
    isInactive = true;
  }

  // a map replaces the palette, so a new map holds the current value of every
  // group; groups it lacks take the renderer's default
  const setValues = (newValues: { [group: string]: TValue }) => {
    let mapId: string;
    if (map !== undefined) {
      mapId = map.id;
      adapter.updateMap(mapId, { values: { ...map.values, ...newValues } });
    } else {
      mapId = crypto.randomUUID();
      adapter.addMap({
        id: mapId,
        name: `${objectName} ${groupBy.column} ${name}`,
        values: {
          ...Object.fromEntries(
            Array.from(groupCounts.keys(), (group) => [group, getValue(group)]),
          ),
          ...newValues,
        },
      });
    }
    property.onConfigChange(ConfigUtils.withGroupByMap(config, groupBy, mapId));
  };

  return { getValue, setValues, isInactive };
}
