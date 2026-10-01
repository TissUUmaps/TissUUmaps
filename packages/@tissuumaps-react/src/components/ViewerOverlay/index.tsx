import { type ReactNode, useEffect, useState } from "react";
import { createPortal } from "react-dom";

import type { Vertex } from "@tissuumaps/core";

import { useOpenSeadragonContext } from "../../context/OpenSeadragonContextContext";

export type ViewerOverlayProps = {
  /** The world position the top-left corner of the overlay is anchored at */
  worldPosition: Vertex;
  children?: ReactNode;
};

/**
 * Renders its children on top of the viewer, anchored at a world position
 *
 * The overlay follows its position as the viewer is panned and zoomed. It does
 * not take the pointer, so the cursor and text selection stay with the viewer.
 */
export function ViewerOverlay({ worldPosition, children }: ViewerOverlayProps) {
  const [container] = useState(() => {
    const container = document.createElement("div");
    container.style.pointerEvents = "none";
    // OpenSeadragon creates its overlay container before the WebGL and SVG
    // overlays are added to the same element, which would leave it below them
    container.style.zIndex = "1";
    return container;
  });

  const context = useOpenSeadragonContext();

  const { x, y } = worldPosition;
  useEffect(() => {
    if (context === null) {
      return;
    }
    context.addOverlay(container, { x, y });
    return () => {
      context.removeOverlay(container);
    };
  }, [context, container, x, y]);

  return createPortal(children, container);
}
