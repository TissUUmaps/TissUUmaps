import { useMemo } from "react";

import type { SizeConfig } from "@tissuumaps/core";

import { useProjectStore } from "@/stores/project";

import type { GroupValuesAdapter } from "./adapter";

/** Returns the group table adapter of the size maps */
export function useSizeGroupValues(): GroupValuesAdapter<number, SizeConfig> {
  const maps = useProjectStore((state) => state.sizeMaps);
  return useMemo(
    () => ({
      maps,
      renderValue: (size) => size,
    }),
    [maps],
  );
}
