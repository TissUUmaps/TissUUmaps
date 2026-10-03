import { useCallback } from "react";

import { useOpenSeadragonContext } from "../context/OpenSeadragonContextContext";

/**
 * Returns a callback that fits the viewport of the enclosing viewer to all of
 * its content again, as on opening a project
 *
 * The callback does nothing until the viewer is ready.
 */
export function useResetViewport(): () => void {
  const context = useOpenSeadragonContext();

  return useCallback(() => {
    context?.resetViewport();
  }, [context]);
}
