import { useMemo } from "react";

import type { VisibilityConfig } from "@tissuumaps/core";

import { useLatestCallback } from "@/hooks/useLatestCallback";

import { createGroupValues } from "./createGroupValues";
import type { GroupProperty } from "./useGroupColumn";
import type { GroupTableState } from "./useGroupTable";

/** How the group table shows and toggles the visibility of a group */
export type GroupVisibility = {
  isVisible: (group: string) => boolean;

  /** Whether the eye buttons are grayed out, as toggling them changes the property source */
  isInactive: boolean;

  onVisibleChange: (groups: string[], visible: boolean) => void;
};

/**
 * Returns how the group table shows and toggles a visibility property for each
 * group
 *
 * @param groupTable - The state of the group table
 * @param property - The visibility property
 * @returns The group visibility, or `undefined` while the table has no column
 * or groups
 */
export function useGroupVisibility(
  groupTable: GroupTableState,
  property: GroupProperty<boolean, VisibilityConfig>,
): GroupVisibility | undefined {
  const { name, default: defaultValue, config, adapter } = property;
  const onConfigChange = useLatestCallback(property.onConfigChange);

  return useMemo(() => {
    const groupValues = createGroupValues(groupTable, {
      name,
      default: defaultValue,
      config,
      onConfigChange,
      adapter,
    });
    if (groupValues === undefined) {
      return undefined;
    }
    const { getValue, setValues, isInactive } = groupValues;
    return {
      isVisible: getValue,
      isInactive,
      onVisibleChange: (groups, visible) =>
        setValues(Object.fromEntries(groups.map((group) => [group, visible]))),
    };
  }, [groupTable, name, defaultValue, config, onConfigChange, adapter]);
}
