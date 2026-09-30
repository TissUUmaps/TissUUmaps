import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { type Color, ColorUtils, type Rect } from "@tissuumaps/core";
import {
  OpenSeadragonContext,
  OpenSeadragonImageRenderer,
  OpenSeadragonLabelsRenderer,
} from "@tissuumaps/render";

import type { ViewerAdapter } from "../adapter";

type OS = {
  viewerElement: HTMLDivElement;
  context: OpenSeadragonContext;
  imageRenderer: OpenSeadragonImageRenderer;
  labelsRenderer: OpenSeadragonLabelsRenderer;
};

export function useOpenSeadragon(
  adapter: ViewerAdapter,
  backgroundColor: Color,
) {
  const {
    projectInstanceId,
    layers,
    images,
    labels,
    tables,
    colorMaps,
    visibilityMaps,
    opacityMaps,
    osOptions,
    loadImage,
    loadLabels,
    loadTable,
  } = adapter;

  const osRef = useRef<OS | null>(null);
  const osPromiseRef = useRef<Promise<OS | null>>(Promise.resolve(null));
  const [osReady, setOSReady] = useState(false);

  const osOptionsRef = useRef(osOptions);
  const backgroundColorHex = ColorUtils.toHex(backgroundColor);

  const [syncImages, dispatchSyncImages] = useReducer((x) => x + 1, 0);
  const [syncLabels, dispatchSyncLabels] = useReducer((x) => x + 1, 0);
  const requestedSyncImagesRef = useRef(0);
  const requestedSyncLabelsRef = useRef(0);

  const initOS = useCallback(
    (viewerElementOrNull: HTMLDivElement | null) => {
      if (viewerElementOrNull === null) {
        return () => {};
      }
      const abortController = new AbortController();
      const viewerElement = viewerElementOrNull;

      async function startOS() {
        abortController.signal.throwIfAborted();
        const context = new OpenSeadragonContext(
          viewerElement,
          ColorUtils.fromHex(backgroundColorHex),
          osOptionsRef.current.viewerOptions,
        );
        let imageRenderer: OpenSeadragonImageRenderer | undefined;
        try {
          imageRenderer = await new Promise<OpenSeadragonImageRenderer>(
            (resolve, reject) => {
              try {
                const imageRenderer = new OpenSeadragonImageRenderer(
                  context,
                  () => resolve(imageRenderer),
                  reject,
                  { anchorIndex: 1, signal: abortController.signal },
                );
              } catch (error) {
                reject(
                  new Error("Error creating image renderer", { cause: error }),
                );
              }
            },
          );
          abortController.signal.throwIfAborted(); // image renderer does not throw on abort
        } catch (error) {
          if (imageRenderer !== undefined) {
            await imageRenderer.destroy();
          }
          await context.destroy();
          throw error;
        }
        let labelsRenderer: OpenSeadragonLabelsRenderer | undefined;
        try {
          labelsRenderer = await new Promise<OpenSeadragonLabelsRenderer>(
            (resolve, reject) => {
              try {
                const labelsRenderer = new OpenSeadragonLabelsRenderer(
                  context,
                  () => resolve(labelsRenderer),
                  reject,
                  { anchorIndex: 2, signal: abortController.signal },
                );
              } catch (error) {
                reject(
                  new Error("Error creating labels renderer", { cause: error }),
                );
              }
            },
          );
          abortController.signal.throwIfAborted(); // labels renderer does not throw on abort
        } catch (error) {
          if (imageRenderer !== undefined) {
            await imageRenderer.destroy();
          }
          if (labelsRenderer !== undefined) {
            await labelsRenderer.destroy();
          }
          await context.destroy();
          throw error;
        }
        const os = { viewerElement, context, imageRenderer, labelsRenderer };
        osRef.current = os;
        setOSReady(true);
        return os;
      }

      async function stopOS() {
        const os = osRef.current;
        setOSReady(false);
        osRef.current = null;
        if (os !== null) {
          const { context, imageRenderer, labelsRenderer } = os;
          await imageRenderer.destroy();
          await labelsRenderer.destroy();
          await context.destroy();
        }
      }

      osPromiseRef.current = osPromiseRef.current
        .then(startOS)
        .catch((error) => {
          if (!abortController.signal.aborted) {
            console.error("Error starting OpenSeadragon", error);
          }
          return null;
        });
      return () => {
        abortController.abort();
        osPromiseRef.current = osPromiseRef.current
          .then(async () => {
            await stopOS();
            return null;
          })
          .catch((error) => {
            console.error("Error stopping OpenSeadragon", error);
            return null;
          });
      };
    },
    [backgroundColorHex],
  );

  useEffect(() => {
    osOptionsRef.current = osOptions;
    if (osReady && osRef.current !== null) {
      osRef.current.context.setViewerOptions(osOptions.viewerOptions);
      osRef.current.context.configureAnimationHandlers(
        osOptions.viewerAnimationStartOptions,
        osOptions.viewerAnimationFinishOptions,
      );
    }
  }, [osReady, osOptions]);

  useEffect(() => {
    if (osReady && osRef.current !== null) {
      osRef.current.context.resetViewport();
    }
  }, [osReady, projectInstanceId]);

  useEffect(() => {
    const abortController = new AbortController();
    if (osReady && osRef.current !== null) {
      osRef.current.imageRenderer.setModel(layers, images);
      osRef.current.imageRenderer
        .updateBounds({ signal: abortController.signal })
        .catch((error) => {
          if (!abortController.signal.aborted) {
            console.error(
              "Error updating OpenSeadragon image world bounds",
              error,
            );
          }
        });
      if (osRef.current.imageRenderer.needsSynchronization()) {
        requestedSyncImagesRef.current++;
        dispatchSyncImages();
      }
    }
    return () => {
      abortController.abort();
    };
  }, [osReady, layers, images]);

  useEffect(() => {
    const abortController = new AbortController();
    if (osReady && osRef.current !== null) {
      osRef.current.labelsRenderer.setModel(layers, labels);
      osRef.current.labelsRenderer
        .updateBounds({ signal: abortController.signal })
        .catch((error) => {
          if (!abortController.signal.aborted) {
            console.error(
              "Error updating OpenSeadragon labels world bounds",
              error,
            );
          }
        });
      if (osRef.current.labelsRenderer.needsSynchronization()) {
        requestedSyncLabelsRef.current++;
        dispatchSyncLabels();
      }
    }
    return () => {
      abortController.abort();
    };
  }, [osReady, layers, labels]);

  useEffect(() => {
    const abortController = new AbortController();
    if (
      osReady &&
      osRef.current !== null &&
      syncImages === requestedSyncImagesRef.current
    ) {
      osRef.current.imageRenderer
        .synchronize(
          { loadObject: loadImage },
          { signal: abortController.signal },
        )
        .catch((error) => {
          if (!abortController.signal.aborted) {
            console.error("Error synchronizing OpenSeadragon images", error);
          }
        });
    }
    return () => {
      abortController.abort();
    };
  }, [osReady, loadImage, syncImages]);

  useEffect(() => {
    const abortController = new AbortController();
    if (
      osReady &&
      osRef.current !== null &&
      syncLabels === requestedSyncLabelsRef.current
    ) {
      osRef.current.labelsRenderer
        .synchronize(
          {
            tables,
            colorMaps,
            visibilityMaps,
            opacityMaps,
            loadObject: loadLabels,
            loadTable,
          },
          { signal: abortController.signal },
        )
        .catch((error) => {
          if (!abortController.signal.aborted) {
            console.error("Error synchronizing OpenSeadragon labels", error);
          }
        });
    }
    return () => {
      abortController.abort();
    };
  }, [
    osReady,
    tables,
    colorMaps,
    visibilityMaps,
    opacityMaps,
    loadLabels,
    loadTable,
    syncLabels,
  ]);

  const updateOSContentBounds = useCallback(
    (contentBounds: Rect[]) => {
      const abortController = new AbortController();
      if (osReady && osRef.current !== null) {
        osRef.current.context
          .setContentBounds(osRef.current, contentBounds, {
            signal: abortController.signal,
          })
          .catch((error) => {
            if (!abortController.signal.aborted) {
              console.error(
                "Error updating OpenSeadragon content bounds",
                error,
              );
            }
          });
      }
      return () => {
        abortController.abort();
      };
    },
    [osReady],
  );

  return { initOS, osRef, osReady, updateOSContentBounds };
}
