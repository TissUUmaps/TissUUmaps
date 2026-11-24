import { useEffect } from "react";

import type { Dims, Rect } from "@tissuumaps/core";

import { useOpenSeadragonContext } from "../context/OpenSeadragonContextContext";

/**
 * Calls back once the viewer is ready, and then whenever the viewport or the
 * container size changes, before the viewer draws the new viewport
 *
 * Only has an effect inside a `Viewer`, e.g. in a `ViewerControl`. Memoize
 * the callback: a new identity resubscribes and calls it again.
 *
 * @param onViewportChange - Called with the visible viewport, in world
 * coordinates, and the container size, in CSS pixels
 */
export function useViewportChange(
  onViewportChange: (viewport: Rect, containerSize: Dims) => void,
) {
  const context = useOpenSeadragonContext();

  useEffect(() => {
    if (context === null) {
      return;
    }
    const onViewportChanged = () => {
      onViewportChange(context.getViewport(), context.getContainerSize());
    };
    onViewportChanged();
    // OpenSeadragon raises "viewport-change" after every container resize too
    context.viewer.addHandler("viewport-change", onViewportChanged);
    return () => {
      context.viewer.removeHandler("viewport-change", onViewportChanged);
    };
  }, [context, onViewportChange]);
}
