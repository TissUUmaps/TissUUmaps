import { useMemo } from "react";

import {
  type Marker,
  type MarkerConfig,
  markerPalette,
} from "@tissuumaps/core";

import { markers } from "@/components/markers";
import { useProjectStore } from "@/stores/project";

import type { GroupValuesAdapter } from "./adapter";

/** Returns the group table adapter of the marker maps */
export function useMarkerGroupValues(): GroupValuesAdapter<
  Marker,
  MarkerConfig
> {
  const maps = useProjectStore((state) => state.markerMaps);
  return useMemo(
    () => ({
      maps,
      renderValue: (marker) => markers.find((m) => m.value === marker)!.icon,
      getPalette: () => markerPalette,
    }),
    [maps],
  );
}
