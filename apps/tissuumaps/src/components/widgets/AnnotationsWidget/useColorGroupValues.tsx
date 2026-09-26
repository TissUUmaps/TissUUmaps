import { Square } from "lucide-react";
import { useMemo } from "react";

import {
  type Color,
  type ColorConfig,
  findColorPalette,
} from "@tissuumaps/core";

import { useProjectStore } from "@/stores/project";

import type { GroupValuesAdapter } from "./adapter";

/** Returns the group table adapter of the color maps */
export function useColorGroupValues(): GroupValuesAdapter<Color, ColorConfig> {
  const maps = useProjectStore((state) => state.colorMaps);
  return useMemo(
    () => ({
      maps,
      renderValue: (color) => (
        <Square
          fill={`rgb(${color.r}, ${color.g}, ${color.b})`}
          className="size-4"
        />
      ),
      getPalette: (config) => findColorPalette(config.groupBy.palette)?.colors,
    }),
    [maps],
  );
}
