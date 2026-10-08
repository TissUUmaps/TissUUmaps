import { useEffect, useRef } from "react";

import {
  type AnnotatedItemRef,
  InteractionMode,
  type Vertex,
} from "@tissuumaps/core";
import type { OpenSeadragonContext } from "@tissuumaps/render";

import type { ViewerAdapter } from "../adapter";

/**
 * Picks the item under a click on the viewer, and clears the pick on Escape
 *
 * Only quick clicks pick, not drags, and only in `pan` mode. A point or shape
 * is picked first, as the WebGL canvas is drawn over the labels; a label is
 * picked only where there is none. A new click aborts the label pick of the
 * previous one.
 *
 * @param adapter - The viewer adapter, whose `onPickedItemChange` is called
 * with every pick
 * @param osContext - The OpenSeadragon context to listen to, if ready
 * @param pickGL - Picks a point or shape at a position on the viewer container
 * @param pickOSLabel - Picks a label at a world position
 */
export function usePicking(
  adapter: ViewerAdapter,
  osContext: OpenSeadragonContext | null,
  pickGL: (containerPosition: Vertex) => AnnotatedItemRef | null,
  pickOSLabel: (
    worldPosition: Vertex,
    options?: { signal?: AbortSignal },
  ) => Promise<AnnotatedItemRef | null>,
): void {
  const { interactionMode, onPickedItemChange } = adapter;

  // read by the handlers, which are registered once per context
  const interactionModeRef = useRef(interactionMode);
  const onPickedItemChangeRef = useRef(onPickedItemChange);
  useEffect(() => {
    interactionModeRef.current = interactionMode;
    onPickedItemChangeRef.current = onPickedItemChange;
  }, [interactionMode, onPickedItemChange]);

  useEffect(() => {
    if (osContext === null) {
      return;
    }
    let abortController: AbortController | null = null;
    const onCanvasClick = (event: OpenSeadragon.CanvasClickEvent) => {
      if (
        !event.quick ||
        interactionModeRef.current !== InteractionMode.pan ||
        onPickedItemChangeRef.current === undefined
      ) {
        return;
      }
      abortController?.abort();
      abortController = new AbortController();
      const { signal } = abortController;
      const containerPosition = { x: event.position.x, y: event.position.y };
      const worldPosition = osContext.getWorldPosition(containerPosition);
      const glItem = pickGL(containerPosition);
      if (glItem !== null) {
        onPickedItemChangeRef.current({ ...glItem, worldPosition });
        return;
      }
      pickOSLabel(worldPosition, { signal }).then(
        (labelItem) => {
          if (!signal.aborted) {
            onPickedItemChangeRef.current?.(
              labelItem !== null ? { ...labelItem, worldPosition } : null,
            );
          }
        },
        (error: unknown) => {
          if (!signal.aborted) {
            console.error("Failed to pick a label", error);
            onPickedItemChangeRef.current?.(null);
          }
        },
      );
    };
    const onCanvasKey = (event: OpenSeadragon.CanvasKeyEvent) => {
      if (
        event.originalEvent.key === "Escape" &&
        interactionModeRef.current === InteractionMode.pan
      ) {
        abortController?.abort();
        onPickedItemChangeRef.current?.(null);
      }
    };
    osContext.viewer.addHandler("canvas-click", onCanvasClick);
    osContext.viewer.addHandler("canvas-key", onCanvasKey);
    return () => {
      abortController?.abort();
      osContext.viewer.removeHandler("canvas-click", onCanvasClick);
      osContext.viewer.removeHandler("canvas-key", onCanvasKey);
    };
  }, [osContext, pickGL, pickOSLabel]);
}
