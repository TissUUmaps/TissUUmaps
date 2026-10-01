import { useCallback, useRef, useState } from "react";

import type { Dims, Rect } from "@tissuumaps/core";
import { useViewerViewport } from "@tissuumaps/react";

import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

import { formatScaleBarLength, getScaleBarLength } from "./scaleBarLength";

/** Maximum width of the scale bar, in CSS pixels */
const maxScaleBarWidth = 200;

type ScaleBarProps = {
  worldUnitSize: number;
  className?: string;
};

function ScaleBar({ worldUnitSize, className }: ScaleBarProps) {
  const barRef = useRef<HTMLDivElement>(null);
  const [lengthLabel, setLengthLabel] = useState("");

  // The bar width changes in every frame of a zoom, so it is written to the
  // DOM directly; the label only re-renders when its text changes.
  const updateScaleBar = useCallback(
    (viewport: Rect, containerSize: Dims) => {
      const pixelSize = (worldUnitSize * viewport.width) / containerSize.width;
      const length = getScaleBarLength(pixelSize * maxScaleBarWidth);
      if (barRef.current !== null) {
        barRef.current.style.width = `${length / pixelSize}px`;
      }
      setLengthLabel(formatScaleBarLength(length));
    },
    [worldUnitSize],
  );
  useViewerViewport(updateScaleBar);

  // An inline-block that hides its overflow has its baseline at its bottom
  // edge, like the navigator, so that both line up at the bottom.
  return (
    <div
      className={cn(
        "ml-2 inline-block overflow-hidden rounded-lg border border-border bg-background/75 px-2 py-1 text-center text-xs shadow-lg",
        className,
      )}
    >
      <div className="tabular-nums">{lengthLabel}</div>
      <div
        ref={barRef}
        className="mx-auto h-1.5 border-x-2 border-b-2 border-foreground"
      />
    </div>
  );
}

export type ScaleBarViewerControlProps = { className?: string };

export function ScaleBarViewerControl({
  className,
}: ScaleBarViewerControlProps) {
  const worldUnitSize = useProjectStore((state) => state.worldUnitSize);

  if (worldUnitSize === undefined || worldUnitSize <= 0) {
    return null;
  }
  return <ScaleBar worldUnitSize={worldUnitSize} className={className} />;
}
