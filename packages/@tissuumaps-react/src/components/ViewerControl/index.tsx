import { type ReactNode, useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { useOpenSeadragonContext } from "../../context/OpenSeadragonContextContext";
import type { ViewerControlAnchor } from "./types";

export { ViewerControlAnchor } from "./types";

export type ViewerControlProps = {
  anchor?: ViewerControlAnchor;
  attachToViewer?: boolean;
  autoFade?: boolean;
  children?: ReactNode;
};

export function ViewerControl({
  anchor,
  attachToViewer,
  autoFade,
  children,
}: ViewerControlProps) {
  const [container] = useState(() => document.createElement("div"));
  const [isAttached, setIsAttached] = useState(false);

  const context = useOpenSeadragonContext();

  useEffect(() => {
    if (context !== null) {
      context.viewer.addControl(container, {
        anchor,
        attachToViewer,
        autoFade,
      });
      // the container can only be attached after mount, and the children
      // must not render before that (see below)
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setIsAttached(true);
    }
    return () => {
      if (context !== null) {
        // @ts-expect-error OpenSeadragon typings are wrong
        context.viewer.removeControl(container);
      }
    };
  }, [context, container, anchor, attachToViewer, autoFade]);

  // Base UI lists composite items (slider thumbs, toggle group items) in a
  // layout effect and ignores nodes that are not yet in the document, so the
  // children must not render before the container is attached to the viewer.
  return createPortal(isAttached ? children : null, container);
}
