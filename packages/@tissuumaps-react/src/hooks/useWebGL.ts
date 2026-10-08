import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import {
  type AnnotatedItemRef,
  AsyncUtils,
  type Dims,
  GeometryUtils,
  type Rect,
  type Vertex,
} from "@tissuumaps/core";
import {
  WebGLContext,
  WebGLPicker,
  WebGLPointsRenderer,
  WebGLShapesRenderer,
} from "@tissuumaps/render";

import type { ViewerAdapter } from "../adapter";

type GL = {
  canvas: HTMLCanvasElement;
  context: WebGLContext;
  pointsRenderer: WebGLPointsRenderer;
  shapesRenderer: WebGLShapesRenderer;
  /** Both renderers, in the order they are drawn in */
  renderers: readonly (WebGLShapesRenderer | WebGLPointsRenderer)[];
  picker: WebGLPicker;
};

/**
 * Clears the canvas and redraws both WebGL renderers
 *
 * Module-level so it closes over nothing from the render scope: the memoized
 * setters can call it without listing it as a dependency, however it changes.
 *
 * @param gl - The GL object to draw on; nothing is drawn if null
 */
function drawGL(gl: GL | null) {
  if (gl !== null) {
    gl.context.clear();
    for (const renderer of gl.renderers) {
      renderer.draw();
    }
  }
}

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
    pickedItem,
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

  const [redraw, dispatchRedraw] = useReducer((x) => x + 1, 0);

  const [glPointsBounds, setGLPointsBounds] = useState<Rect | null>(null);
  const [glShapesBounds, setGLShapesBounds] = useState<Rect | null>(null);

  const setGLViewport = useCallback((viewport: Rect) => {
    if (
      viewportRef.current === null ||
      !GeometryUtils.rectEquals(viewport, viewportRef.current)
    ) {
      viewportRef.current = viewport;
      if (glRef.current !== null) {
        glRef.current.pointsRenderer.viewport = viewport;
        glRef.current.shapesRenderer.viewport = viewport;
        drawGL(glRef.current); // keep direct to avoid lags!
      }
    }
  }, []);

  const setGLContainerSize = useCallback((containerSize: Dims) => {
    containerSizeRef.current = containerSize;
    if (glRef.current !== null) {
      const redraw = glRef.current.context.resizeCanvas(
        glRef.current.canvas,
        containerSize,
      );
      // OSD raises "resize" before it updates the viewport bounds, so this
      // draws the old viewport and is superseded by the viewport-change draw
      // later in the same update - except on a resize that leaves the bounds
      // unchanged, where it is the only draw that refills the resized, and
      // therefore blank, canvas.
      if (redraw) {
        drawGL(glRef.current); // keep direct to avoid lags!
      }
    }
  }, []);

  const pickGL = useCallback(
    (containerPosition: Vertex): AnnotatedItemRef | null => {
      const gl = glRef.current;
      if (gl === null) {
        return null;
      }
      const { canvas, picker } = gl;
      if (canvas.clientWidth === 0 || canvas.clientHeight === 0) {
        return null;
      }
      // the canvas fills its parent, but its backing size is clamped (see
      // WebGLContext.resizeCanvas), so its own size ratio takes the place of
      // devicePixelRatio
      return picker.pick({
        x: (containerPosition.x * canvas.width) / canvas.clientWidth,
        y: (containerPosition.y * canvas.height) / canvas.clientHeight,
      });
    },
    [],
  );

  const initGL = useCallback((parentOrNull: HTMLElement | null) => {
    if (parentOrNull === null) {
      return () => {};
    }
    const abortController = new AbortController();
    const canvas = parentOrNull.appendChild(createCanvas());

    async function startGL() {
      abortController.signal.throwIfAborted();
      const context = new WebGLContext(canvas);
      if (containerSizeRef.current !== null) {
        context.resizeCanvas(canvas, containerSizeRef.current);
      }
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
        throw new Error("Failed to create the points renderer", {
          cause: error,
        });
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
        throw new Error("Failed to create the shapes renderer", {
          cause: error,
        });
      }
      // points are drawn over shapes, so that transcripts show over filled cells
      const renderers = [shapesRenderer, pointsRenderer];
      let picker: WebGLPicker;
      try {
        picker = new WebGLPicker(context, renderers);
      } catch (error) {
        pointsRenderer.destroy();
        shapesRenderer.destroy();
        context.destroy();
        throw new Error("Error creating picker", { cause: error });
      }
      // set only now, as it may have changed while awaiting the renderers
      if (viewportRef.current !== null) {
        pointsRenderer.viewport = viewportRef.current;
        shapesRenderer.viewport = viewportRef.current;
      }
      const gl = {
        canvas,
        context,
        pointsRenderer,
        shapesRenderer,
        renderers,
        picker,
      };
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
        const { context, pointsRenderer, shapesRenderer, picker } = gl;
        picker.destroy();
        pointsRenderer.destroy();
        shapesRenderer.destroy();
        context.destroy();
      }
    }

    glPromiseRef.current = glPromiseRef.current.then(startGL).catch((error) => {
      if (!abortController.signal.aborted) {
        console.error("Failed to start WebGL", error);
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
          console.error("Failed to stop WebGL", error);
          return null;
        });
    };
    const onContextRestored = () => {
      glPromiseRef.current = glPromiseRef.current
        .then(startGL)
        .catch((error) => {
          if (!abortController.signal.aborted) {
            console.error("Failed to start WebGL", error);
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
          console.error("Failed to stop WebGL", error);
          return null;
        });
      parentOrNull.removeChild(canvas);
    };
  }, []);

  useEffect(() => {
    if (glReady && glRef.current !== null) {
      glRef.current.pointsRenderer.setOutlinedItem(pickedItem);
      glRef.current.shapesRenderer.setOutlinedItem(pickedItem);
      dispatchRedraw();
    }
  }, [glReady, pickedItem]);

  useEffect(() => {
    if (glReady && glRef.current !== null) {
      glRef.current.pointsRenderer.renderOptions =
        glOptions.pointsRenderOptions;
      dispatchRedraw();
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
      dispatchRedraw();
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
        dispatchRedraw();
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
        dispatchRedraw();
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
          {
            signal: abortController.signal,
            onChange: () => {
              if (glRef.current !== null && !abortController.signal.aborted) {
                const newPointsBounds =
                  glRef.current.pointsRenderer.getRenderedBounds();
                setGLPointsBounds(updateBounds(newPointsBounds));
                dispatchRedraw();
              }
            },
          },
        )
        .catch((error) => {
          if (!abortController.signal.aborted) {
            console.error("Failed to synchronize WebGL points", error);
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
          {
            signal: abortController.signal,
            onChange: () => {
              if (glRef.current !== null && !abortController.signal.aborted) {
                const newShapesBounds =
                  glRef.current.shapesRenderer.getRenderedBounds();
                setGLShapesBounds(updateBounds(newShapesBounds));
                dispatchRedraw();
              }
            },
          },
        )
        .catch((error) => {
          if (!abortController.signal.aborted) {
            console.error("Failed to synchronize WebGL shapes", error);
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

  useEffect(() => {
    if (glReady) {
      drawGL(glRef.current);
    }
  }, [glReady, redraw]);

  return {
    initGL,
    setGLViewport,
    setGLContainerSize,
    pickGL,
    glPointsBounds,
    glShapesBounds,
  };
}
