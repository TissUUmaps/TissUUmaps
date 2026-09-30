import type { DockviewApi } from "dockview-react";
import { useEffect } from "react";

import { PanelId } from "@/components/panels/panelId";
import { appStore, useAppStore } from "@/stores/app";

/**
 * Activates the dockview panel of the collection of the object whose settings
 * are to be brought to the front, as requested through the app store's
 * `focusObject`, and clears the request
 *
 * The panel expands the object's settings itself (see `useExpandedObjectIds`).
 *
 * @param dockviewApi - The API of the ready dockview, or `null` while it is not
 * ready yet
 */
export function useFocusedObjectPanel(dockviewApi: DockviewApi | null): void {
  const focusedObject = useAppStore((state) => state.focusedObject);
  useEffect(() => {
    if (dockviewApi !== null && focusedObject !== null) {
      const panel = dockviewApi.getPanel(PanelId[focusedObject.kind]);
      // dockview re-opens a panel that is activated while already active, which
      // re-attaches its content and resets the scroll positions within
      if (panel !== undefined && !panel.api.isActive) {
        panel.api.setActive();
      }
      appStore.setState({ focusedObject: null });
    }
  }, [dockviewApi, focusedObject]);
}
