import { EyeIcon, EyeOffIcon } from "lucide-react";
import { useMemo } from "react";

import type { VisibilityConfig } from "@tissuumaps/core";

import { useProjectStore } from "@/stores/project";

import type { GroupValuesAdapter } from "./adapter";

/** Returns the group table adapter of the visibility maps */
export function useVisibilityGroupValues(): GroupValuesAdapter<
  boolean,
  VisibilityConfig
> {
  const maps = useProjectStore((state) => state.visibilityMaps);
  return useMemo(
    () => ({
      maps,
      renderValue: (visible) => (visible ? <EyeIcon /> : <EyeOffIcon />),
    }),
    [maps],
  );
}
