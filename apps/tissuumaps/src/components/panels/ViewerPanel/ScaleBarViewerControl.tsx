import { useCallback, useRef } from "react";

import {
  type Dims,
  PhysicalUnit,
  type Rect,
  physicalUnits,
} from "@tissuumaps/core";
import { useViewportChange } from "@tissuumaps/react";

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
  const physicalUnitSize = useProjectStore((state) => state.physicalUnitSize);
  const physicalUnit = useProjectStore((state) => state.physicalUnit);
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
      const hasPhysicalUnitSize =
        physicalUnitSize !== undefined && physicalUnitSize > 0;
      const knownPhysicalUnit = Object.values(PhysicalUnit).find(
        (unit) => unit === physicalUnit,
      );
      // the label is in world units until the physical unit size is set, then
      // in meters for a known physical unit and in the unit as typed otherwise
      let labelUnit = "units";
      let labelUnitsPerWorldUnit = 1;
      if (hasPhysicalUnitSize && knownPhysicalUnit !== undefined) {
        labelUnit = PhysicalUnit.meter;
        labelUnitsPerWorldUnit =
          physicalUnitSize * physicalUnits[knownPhysicalUnit].size;
      } else if (hasPhysicalUnitSize) {
        labelUnit = physicalUnit;
        labelUnitsPerWorldUnit = physicalUnitSize;
      }
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
    [physicalUnitSize, physicalUnit],
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
