import { useEffect } from "react";

import type { Dims, Rect } from "@tissuumaps/core";

import { useOpenSeadragonContext } from "../context/OpenSeadragonContextContext";

/**
 * Subscribes to the viewport of the enclosing viewer
 *
 * The handler is called once the viewer is ready, and then whenever the
 * viewport or the viewer's container size changes, before the viewer draws the
 * new viewport. It is called outside of React state, so a change does not
 * re-render the calling component.
 *
 * @param onViewportChange - Called with the visible viewport, in world
 * coordinates, and the container size, in CSS pixels; a new identity
 * resubscribes and calls it again
 */
export function useViewerViewport(
  onViewportChange: (viewport: Rect, containerSize: Dims) => void,
): void {
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
