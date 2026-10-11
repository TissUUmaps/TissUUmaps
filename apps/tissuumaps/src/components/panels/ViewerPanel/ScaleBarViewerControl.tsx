import {
  type Ref,
  type RefObject,
  useCallback,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

import type { Dims, Rect } from "@tissuumaps/core";

import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

import { formatScaleBarLength } from "./formatScaleBarLength";
import { getScaleBarLength } from "./getScaleBarLength";

/** Maximum width of the scale bar, in CSS pixels */
const maxScaleBarWidth = 200;

export type ScaleBarViewerControlHandle = {
  update: () => void;
};

export type ScaleBarViewerControlProps = {
  viewportRef: RefObject<{ viewport: Rect; containerSize: Dims } | null>;
  ref?: Ref<ScaleBarViewerControlHandle>;
  className?: string;
};

export function ScaleBarViewerControl({
  viewportRef,
  ref,
  className,
}: ScaleBarViewerControlProps) {
  const worldUnitSize = useProjectStore((state) => state.worldUnitSize);
  const scaleBarVisible = useProjectStore((state) => state.scaleBarVisible);
  const barRef = useRef<HTMLDivElement>(null);
  const [lengthLabel, setLengthLabel] = useState("");

  // The bar width changes in every frame of a zoom, so it is written to the
  // DOM directly; the label only re-renders when its text changes.
  const updateScaleBar = useCallback(() => {
    if (
      !scaleBarVisible ||
      worldUnitSize === undefined ||
      viewportRef.current === null ||
      barRef.current === null
    ) {
      return;
    }
    const { viewport, containerSize } = viewportRef.current;
    const pixelSize = (worldUnitSize * viewport.width) / containerSize.width;
    const length = getScaleBarLength(pixelSize * maxScaleBarWidth);
    barRef.current.style.width = `${length / pixelSize}px`;
    setLengthLabel(formatScaleBarLength(length));
  }, [viewportRef, scaleBarVisible, worldUnitSize]);
  useLayoutEffect(updateScaleBar, [updateScaleBar]);

  useImperativeHandle(ref, () => ({ update: updateScaleBar }), [
    updateScaleBar,
  ]);

  if (!scaleBarVisible || worldUnitSize === undefined || worldUnitSize <= 0) {
    return null;
  }
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
