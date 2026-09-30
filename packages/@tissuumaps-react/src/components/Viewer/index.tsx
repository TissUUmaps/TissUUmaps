import { type ReactNode, useEffect, useState } from "react";

import { type Color, ColorUtils, type Dims, type Rect } from "@tissuumaps/core";
import type { OpenSeadragonContext } from "@tissuumaps/render";

import type { ViewerAdapter } from "../../adapter";
import { OpenSeadragonContextProvider } from "../../context/OpenSeadragonContextProvider";
import { useOpenSeadragon } from "../../hooks/useOpenSeadragon";
import { useSVG } from "../../hooks/useSVG";
import { useWebGL } from "../../hooks/useWebGL";

export type ViewerProps = {
  adapter: ViewerAdapter;
  backgroundColor: Color;
  children?: ReactNode;
  className?: string;
};

export function Viewer({
  adapter,
  backgroundColor,
  children,
  className,
}: ViewerProps) {
  const [osContext, setOSContext] = useState<OpenSeadragonContext | null>(null);

  const { initOS, osRef, osReady, updateOSContentBounds } = useOpenSeadragon(
    adapter,
    backgroundColor,
  );
  const {
    initGL,
    setGLViewport,
    setGLContainerSize,
    glPointsBounds,
    glShapesBounds,
  } = useWebGL(adapter);
  const { initSVG, setSVGViewport, setSVGContainerSize } = useSVG(adapter);

  // the setters and init callbacks are memoized; a new identity tears down and
  // recreates the whole overlay
  useEffect(() => {
    const os = osRef.current;
    if (!osReady || os === null) {
      return;
    }
    setOSContext(os.context);
    // Push the viewport and container size straight into the overlays instead
    // of through React state. OSD raises "viewport-change" and "resize" from its
    // animation-frame update, after the springs advanced and before it draws the
    // world, and getBounds(true) returns the bounds it is about to paint.
    // Updating the overlays here therefore lands in the same frame, whereas a
    // passive useEffect runs after paint and would leave the overlay one frame
    // behind the OSD canvas.
    const updateViewport = (viewport: Rect) => {
      setGLViewport(viewport);
      setSVGViewport(viewport);
    };
    const updateContainerSize = (containerSize: Dims) => {
      setGLContainerSize(containerSize);
      setSVGContainerSize(containerSize);
    };
    updateViewport(os.context.getViewport());
    updateContainerSize(os.context.getContainerSize());
    const onViewportChanged = (event: OpenSeadragon.ViewerEvent) => {
      const { x, y, width, height } =
        event.eventSource.viewport.getBounds(true);
      updateViewport({ x, y, width, height });
    };
    const onContainerResized = (event: OpenSeadragon.ResizeEvent) => {
      const { x: width, y: height } = event.newContainerSize;
      updateContainerSize({ width, height });
    };
    os.context.viewer.addHandler("resize", onContainerResized);
    os.context.viewer.addHandler("viewport-change", onViewportChanged);
    const destroyGL = initGL(os.context.viewer.canvas);
    const destroySVG = initSVG(os.context.viewer.canvas);
    return () => {
      destroyGL();
      destroySVG();
      os.context.viewer.removeHandler("resize", onContainerResized);
      os.context.viewer.removeHandler("viewport-change", onViewportChanged);
      setOSContext(null);
    };
  }, [
    osReady,
    osRef,
    initGL,
    initSVG,
    setGLViewport,
    setSVGViewport,
    setGLContainerSize,
    setSVGContainerSize,
  ]);

  useEffect(() => {
    const osContentBounds = [];
    if (glPointsBounds !== null) {
      osContentBounds.push(glPointsBounds);
    }
    if (glShapesBounds !== null) {
      osContentBounds.push(glShapesBounds);
    }
    return updateOSContentBounds(osContentBounds);
  }, [updateOSContentBounds, glPointsBounds, glShapesBounds]);

  return (
    <div
      ref={initOS}
      className={className}
      style={{ backgroundColor: ColorUtils.toHex(backgroundColor) }}
    >
      <OpenSeadragonContextProvider context={osContext}>
        {children}
      </OpenSeadragonContextProvider>
    </div>
  );
}
