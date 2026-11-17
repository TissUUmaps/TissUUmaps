import { useCallback, useRef } from "react";

import type { Dims, Rect } from "@tissuumaps/core";
import { useViewportChange } from "@tissuumaps/react";

import { lengthUnits } from "@/lib/lengthUnits";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

import { formatScaleBarLength } from "./formatScaleBarLength";
import { getScaleBarLength } from "./getScaleBarLength";

/** Maximum width of the scale bar, in CSS pixels */
const maxScaleBarWidth = 200;

export type ScaleBarViewerControlProps = {
  className?: string;
};

export function ScaleBarViewerControl({
  className,
}: ScaleBarViewerControlProps) {
  const worldUnitSize = useProjectStore((state) => state.worldUnitSize);
  const lengthUnit = useProjectStore((state) => state.lengthUnit);
  const barRef = useRef<HTMLDivElement>(null);
  const labelRef = useRef<HTMLDivElement>(null);

  // The scale bar changes in every frame of a zoom, so it is written to the
  // DOM directly, in the same frame as the viewer draws.
  const updateScaleBar = useCallback(
    (viewport: Rect, containerSize: Dims) => {
      if (
        barRef.current === null ||
        labelRef.current === null ||
        containerSize.width <= 0
      ) {
        return;
      }
      const hasWorldUnitSize = worldUnitSize !== undefined && worldUnitSize > 0;
      // the label is in meters, or in world units until the world unit size is set
      const labelUnit = hasWorldUnitSize ? "m" : "units";
      const labelUnitsPerWorldUnit = hasWorldUnitSize
        ? worldUnitSize * lengthUnits[lengthUnit].size
        : 1;
      const worldUnitsPerCssPixel = viewport.width / containerSize.width;
      const labelUnitsPerCssPixel =
        labelUnitsPerWorldUnit * worldUnitsPerCssPixel;
      const labelLength = getScaleBarLength(
        labelUnitsPerCssPixel * maxScaleBarWidth,
      );
      barRef.current.style.width = `${labelLength / labelUnitsPerCssPixel}px`;
      labelRef.current.textContent = formatScaleBarLength(
        labelLength,
        labelUnit,
      );
    },
    [worldUnitSize, lengthUnit],
  );
  useViewportChange(updateScaleBar);

  // An inline-block that hides its overflow has its baseline at its bottom
  // edge, like the navigator, so that both line up at the bottom.
  return (
    <div
      className={cn(
        "ml-2 inline-block overflow-hidden rounded-lg border border-border bg-background/75 px-2 py-1 text-center text-xs shadow-lg",
        className,
      )}
    >
      <div ref={labelRef} className="tabular-nums" />
      <div
        ref={barRef}
        className="mx-auto h-1.5 border-x-2 border-b-2 border-foreground"
      />
    </div>
  );
}
