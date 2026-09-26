import { useMemo } from "react";

import type { OpacityConfig } from "@tissuumaps/core";

import { useProjectStore } from "@/stores/project";

import type { GroupValuesAdapter } from "./adapter";

/** Returns the group table adapter of the opacity maps */
export function useOpacityGroupValues(): GroupValuesAdapter<
  number,
  OpacityConfig
> {
  const maps = useProjectStore((state) => state.opacityMaps);
  return useMemo(
    () => ({
      maps,
      renderValue: (opacity) => opacity,
    }),
    [maps],
  );
}
