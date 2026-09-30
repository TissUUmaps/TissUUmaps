import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import {
  AsyncUtils,
  type Dims,
  GeometryUtils,
  type Rect,
} from "@tissuumaps/core";
import {
  WebGLContext,
  WebGLFrameScheduler,
  WebGLPointsRenderer,
  WebGLShapesRenderer,
} from "@tissuumaps/render";

import type { ViewerAdapter } from "../adapter";

type GL = {
  canvas: HTMLCanvasElement;
  context: WebGLContext;
  pointsRenderer: WebGLPointsRenderer;
  shapesRenderer: WebGLShapesRenderer;
  scheduler: WebGLFrameScheduler;
};

/**
 * Creates a state updater that sets new bounds, but keeps the current ones if
 * they are equal, so that equal bounds do not trigger a re-render
 *
 * @param newBounds - The new bounds, or null if nothing is drawn
 * @returns The state updater
 */
function updateBounds(newBounds: Rect | null) {
  return (currentBounds: Rect | null) =>
    newBounds !== null &&
    currentBounds !== null &&
    GeometryUtils.rectEquals(currentBounds, newBounds)
      ? currentBounds
      : newBounds;
}

/**
 * Creates the canvas the WebGL renderers draw on, covering its parent
 */
function createCanvas() {
  const canvas = document.createElement("canvas");
  canvas.style.position = "absolute";
  canvas.style.top = "0";
  canvas.style.left = "0";
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  return canvas;
}

export function useWebGL(adapter: ViewerAdapter) {
  const {
    layers,
    points,
    shapes,
    tables,
    markerMaps,
    sizeMaps,
    colorMaps,
    visibilityMaps,
    opacityMaps,
    glOptions,
    loadPoints,
    loadShapes,
    loadTable,
  } = adapter;

  const glRef = useRef<GL | null>(null);
  const glPromiseRef = useRef<Promise<GL | null>>(Promise.resolve(null));
  const [glReady, setGLReady] = useState(false);

  const viewportRef = useRef<Rect | null>(null);
  const containerSizeRef = useRef<Dims | null>(null);

  const [syncPoints, dispatchSyncPoints] = useReducer((x) => x + 1, 0);
  const [syncShapes, dispatchSyncShapes] = useReducer((x) => x + 1, 0);
  const syncPointsAbortControllerRef = useRef<AbortController | null>(null);
  const syncShapesAbortControllerRef = useRef<AbortController | null>(null);
  const requestedSyncPointsRef = useRef(0);
  const requestedSyncShapesRef = useRef(0);

  const [glPointsBounds, setGLPointsBounds] = useState<Rect | null>(null);
  const [glShapesBounds, setGLShapesBounds] = useState<Rect | null>(null);

  const setGLViewport = useCallback((viewport: Rect) => {
    viewportRef.current = viewport;
    if (glRef.current !== null) {
      glRef.current.scheduler.setViewport(viewport);
    }
  }, []);

  const setGLContainerSize = useCallback((containerSize: Dims) => {
    containerSizeRef.current = containerSize;
    if (glRef.current !== null) {
      glRef.current.scheduler.setContainerSize(containerSize);
    }
  }, []);

  const initGL = useCallback((parentOrNull: HTMLElement | null) => {
    if (parentOrNull === null) {
      return () => {};
    }
    const abortController = new AbortController();
    const canvas = parentOrNull.appendChild(createCanvas());

    async function startGL() {
      abortController.signal.throwIfAborted();
      const context = new WebGLContext(canvas);
      const {
        promise: pointsRendererInitPromise,
        resolve: resolvePointsRendererInitPromise,
        reject: rejectPointsRendererInitPromise,
      } = AsyncUtils.withResolvers<void>();
      pointsRendererInitPromise.catch(() => {}); // prevent unhandled rejections in console
      let pointsRenderer: WebGLPointsRenderer;
      try {
        pointsRenderer = new WebGLPointsRenderer(
          context,
          resolvePointsRendererInitPromise,
          rejectPointsRendererInitPromise,
          { signal: abortController.signal },
        );
      } catch (error) {
        context.destroy();
        throw new Error("Error creating points renderer", { cause: error });
      }
      try {
        await pointsRendererInitPromise;
        abortController.signal.throwIfAborted(); // points renderer does not throw on abort
      } catch (error) {
        pointsRenderer.destroy();
        context.destroy();
        throw error;
      }
      let shapesRenderer: WebGLShapesRenderer;
      try {
        shapesRenderer = new WebGLShapesRenderer(context);
      } catch (error) {
        pointsRenderer.destroy();
        context.destroy();
        throw new Error("Error creating shapes renderer", { cause: error });
      }
      // points are drawn over shapes, so that transcripts show over filled cells
      const scheduler = new WebGLFrameScheduler(context, canvas, [
        shapesRenderer,
        pointsRenderer,
      ]);
      // set only now, as both may have changed while awaiting the points
      // renderer
      if (containerSizeRef.current !== null) {
        scheduler.setContainerSize(containerSizeRef.current);
      }
      if (viewportRef.current !== null) {
        scheduler.setViewport(viewportRef.current);
      }
      const gl = { canvas, context, pointsRenderer, shapesRenderer, scheduler };
      glRef.current = gl;
      setGLReady(true);
      return gl;
    }

    function stopGL() {
      // On context loss, the sync effects are only cleaned up (and thereby
      // aborted) after the re-render that unsetting glReady schedules, so their
      // synchronizations could resume on the renderers destroyed below and fail
      // with spurious errors. Abort them first; on unmount, React has already
      // cleaned up the effects, and this does nothing.
      syncPointsAbortControllerRef.current?.abort();
      syncShapesAbortControllerRef.current?.abort();
      const gl = glRef.current;
      setGLReady(false);
      glRef.current = null;
      if (gl !== null) {
        const { context, pointsRenderer, shapesRenderer, scheduler } = gl;
        scheduler.destroy();
        pointsRenderer.destroy();
        shapesRenderer.destroy();
        context.destroy();
      }
    }

    glPromiseRef.current = glPromiseRef.current.then(startGL).catch((error) => {
      if (!abortController.signal.aborted) {
        console.error("Error starting WebGL", error);
      }
      return null;
    });
    const onContextLost = (event: Event) => {
      event.preventDefault(); // allow context to be restored
      glPromiseRef.current = glPromiseRef.current
        .then(() => {
          stopGL();
          return null;
        })
        .catch((error) => {
          console.error("Error stopping WebGL", error);
          return null;
        });
    };
    const onContextRestored = () => {
      glPromiseRef.current = glPromiseRef.current
        .then(startGL)
        .catch((error) => {
          if (!abortController.signal.aborted) {
            console.error("Error starting WebGL", error);
          }
          return null;
        });
    };
    canvas.addEventListener("webglcontextlost", onContextLost);
    canvas.addEventListener("webglcontextrestored", onContextRestored);
    return () => {
      abortController.abort();
      canvas.removeEventListener("webglcontextlost", onContextLost);
      canvas.removeEventListener("webglcontextrestored", onContextRestored);
      glPromiseRef.current = glPromiseRef.current
        .then(() => {
          stopGL();
          return null;
        })
        .catch((error) => {
          console.error("Error stopping WebGL", error);
          return null;
        });
      parentOrNull.removeChild(canvas);
    };
  }, []);

  useEffect(() => {
    if (glReady && glRef.current !== null) {
      glRef.current.pointsRenderer.renderOptions =
        glOptions.pointsRenderOptions;
      glRef.current.scheduler.invalidate();
      if (glRef.current.pointsRenderer.needsSynchronization()) {
        requestedSyncPointsRef.current++;
        dispatchSyncPoints();
      }
    }
  }, [glReady, glOptions.pointsRenderOptions]);

  useEffect(() => {
    if (glReady && glRef.current !== null) {
      glRef.current.shapesRenderer.renderOptions =
        glOptions.shapesRenderOptions;
      // the rendered bounds include the stroke width, a render option
      const newShapesBounds = glRef.current.shapesRenderer.getRenderedBounds();
      setGLShapesBounds(updateBounds(newShapesBounds));
      glRef.current.scheduler.invalidate();
      if (glRef.current.shapesRenderer.needsSynchronization()) {
        requestedSyncShapesRef.current++;
        dispatchSyncShapes();
      }
    }
  }, [glReady, glOptions.shapesRenderOptions]);

  useEffect(() => {
    if (glReady && glRef.current !== null) {
      if (glRef.current.pointsRenderer.setModel(layers, points)) {
        const newPointsBounds =
          glRef.current.pointsRenderer.getRenderedBounds();
        setGLPointsBounds(updateBounds(newPointsBounds));
        glRef.current.scheduler.invalidate();
      }
      if (glRef.current.pointsRenderer.needsSynchronization()) {
        requestedSyncPointsRef.current++;
        dispatchSyncPoints();
      }
    }
  }, [glReady, layers, points]);

  useEffect(() => {
    if (glReady && glRef.current !== null) {
      if (glRef.current.shapesRenderer.setModel(layers, shapes)) {
        const newShapesBounds =
          glRef.current.shapesRenderer.getRenderedBounds();
        setGLShapesBounds(updateBounds(newShapesBounds));
        glRef.current.scheduler.invalidate();
      }
      if (glRef.current.shapesRenderer.needsSynchronization()) {
        requestedSyncShapesRef.current++;
        dispatchSyncShapes();
      }
    }
  }, [glReady, layers, shapes]);

  useEffect(() => {
    const abortController = new AbortController();
    syncPointsAbortControllerRef.current = abortController;
    if (
      glReady &&
      glRef.current !== null &&
      syncPoints === requestedSyncPointsRef.current
    ) {
      glRef.current.pointsRenderer
        .synchronize(
          {
            tables,
            markerMaps,
            sizeMaps,
            colorMaps,
            visibilityMaps,
            opacityMaps,
            loadObject: loadPoints,
            loadTable,
          },
          { signal: abortController.signal },
        )
        .then((changed) => {
          if (
            changed &&
            glRef.current !== null &&
            !abortController.signal.aborted
          ) {
            const newPointsBounds =
              glRef.current.pointsRenderer.getRenderedBounds();
            setGLPointsBounds(updateBounds(newPointsBounds));
            glRef.current.scheduler.invalidate();
          }
        })
        .catch((error) => {
          if (!abortController.signal.aborted) {
            console.error("Error synchronizing WebGL points", error);
          }
        });
    }
    return () => {
      abortController.abort();
    };
  }, [
    glReady,
    tables,
    markerMaps,
    sizeMaps,
    colorMaps,
    visibilityMaps,
    opacityMaps,
    loadPoints,
    loadTable,
    syncPoints,
  ]);

  useEffect(() => {
    const abortController = new AbortController();
    syncShapesAbortControllerRef.current = abortController;
    if (
      glReady &&
      glRef.current !== null &&
      syncShapes === requestedSyncShapesRef.current
    ) {
      glRef.current.shapesRenderer
        .synchronize(
          {
            tables,
            colorMaps,
            visibilityMaps,
            opacityMaps,
            loadObject: loadShapes,
            loadTable,
          },
          { signal: abortController.signal },
        )
        .then((changed) => {
          if (
            changed &&
            glRef.current !== null &&
            !abortController.signal.aborted
          ) {
            const newShapesBounds =
              glRef.current.shapesRenderer.getRenderedBounds();
            setGLShapesBounds(updateBounds(newShapesBounds));
            glRef.current.scheduler.invalidate();
          }
        })
        .catch((error) => {
          if (!abortController.signal.aborted) {
            console.error("Error synchronizing WebGL shapes", error);
          }
        });
    }
    return () => {
      abortController.abort();
    };
  }, [
    glReady,
    tables,
    colorMaps,
    visibilityMaps,
    opacityMaps,
    loadShapes,
    loadTable,
    syncShapes,
  ]);

  return {
    initGL,
    setGLViewport,
    setGLContainerSize,
    glPointsBounds,
    glShapesBounds,
  };
}
