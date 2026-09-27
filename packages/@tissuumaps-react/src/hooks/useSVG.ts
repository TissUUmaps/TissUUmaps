import { useCallback, useEffect, useRef, useState } from "react";

import type { Dims, Rect } from "@tissuumaps/core";
import { SVGController } from "@tissuumaps/render";

import type { ViewerAdapter } from "../adapter";

export function useSVG(adapter: ViewerAdapter) {
  const { interactionMode, addShape } = adapter;

  const svgRef = useRef<{ controller: SVGController } | null>(null);
  const [svgReady, setSVGReady] = useState(false);

  const interactionModeRef = useRef(interactionMode);
  const viewportRef = useRef<Rect | null>(null);
  const containerSizeRef = useRef<Dims | null>(null);

  const setSVGViewport = useCallback((viewport: Rect) => {
    viewportRef.current = viewport;
    if (svgRef.current !== null) {
      svgRef.current.controller.setViewport(viewport);
    }
  }, []);

  const setSVGContainerSize = useCallback((containerSize: Dims) => {
    containerSizeRef.current = containerSize;
    if (svgRef.current !== null) {
      svgRef.current.controller.resizeContainer(containerSize);
    }
  }, []);

  const initSVG = useCallback(
    (parent: HTMLElement | null) => {
      if (parent === null) {
        return () => {};
      }
      const container = SVGController.createContainer();
      parent.appendChild(container);
      const controller = new SVGController(container, {
        onShapeComplete: addShape,
      });
      if (viewportRef.current !== null) {
        controller.setViewport(viewportRef.current);
      }
      if (containerSizeRef.current !== null) {
        controller.resizeContainer(containerSizeRef.current);
      }
      if (interactionModeRef.current !== undefined) {
        controller.setInteractionMode(interactionModeRef.current);
      }
      svgRef.current = { controller };
      setSVGReady(true);
      return () => {
        setSVGReady(false);
        svgRef.current = null;
        controller.destroy();
        parent.removeChild(container);
      };
    },
    [addShape],
  );

  useEffect(() => {
    interactionModeRef.current = interactionMode;
    if (svgReady && svgRef.current !== null) {
      svgRef.current.controller.setInteractionMode(interactionMode);
    }
  }, [svgReady, interactionMode]);

  return { initSVG, setSVGViewport, setSVGContainerSize, svgRef, svgReady };
}
