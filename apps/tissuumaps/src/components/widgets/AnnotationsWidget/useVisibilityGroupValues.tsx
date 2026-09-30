import { useMemo } from "react";

import type { VisibilityConfig } from "@tissuumaps/core";

import { useProjectStore } from "@/stores/project";

import { type GroupValuesAdapter, defaultGroupColumnSize } from "./adapter";
import { GroupVisibilityCell } from "./cells/GroupVisibilityCell";

/** Returns the group table adapter of the visibility maps */
export function useVisibilityGroupValues(): GroupValuesAdapter<
  boolean,
  VisibilityConfig
> {
  const maps = useProjectStore((state) => state.visibilityMaps);
  const addMap = useProjectStore((state) => state.addVisibilityMap);
  const updateMap = useProjectStore((state) => state.updateVisibilityMap);
  return useMemo(
    () => ({
      maps,
      addMap,
      updateMap,
      columnSize: defaultGroupColumnSize,
      getSortValue: (visible) => Number(visible),
      renderCell: (visible, onVisibleChange) => (
        <GroupVisibilityCell
          visible={visible}
          onVisibleChange={onVisibleChange}
        />
      ),
    }),
    [maps, addMap, updateMap],
  );
}
