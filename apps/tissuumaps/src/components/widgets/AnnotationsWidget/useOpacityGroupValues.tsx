import { useMemo } from "react";

import type { OpacityConfig } from "@tissuumaps/core";

import { useProjectStore } from "@/stores/project";

import {
  type GroupValuesAdapter,
  defaultNumericGroupColumnSize,
} from "./adapter";
import { GroupOpacityCell } from "./cells/GroupOpacityCell";

/** Returns the group table adapter of the opacity maps */
export function useOpacityGroupValues(): GroupValuesAdapter<
  number,
  OpacityConfig
> {
  const maps = useProjectStore((state) => state.opacityMaps);
  const addMap = useProjectStore((state) => state.addOpacityMap);
  const updateMap = useProjectStore((state) => state.updateOpacityMap);
  return useMemo(
    () => ({
      maps,
      addMap,
      updateMap,
      columnSize: defaultNumericGroupColumnSize,
      getSortValue: (opacity) => opacity,
      renderCell: (opacity, onOpacityChange) => (
        <GroupOpacityCell opacity={opacity} onOpacityChange={onOpacityChange} />
      ),
    }),
    [maps, addMap, updateMap],
  );
}
