import type { DockviewApi } from "dockview-react";
import { useEffect, useState } from "react";

import type { FocusedObject } from "@tissuumaps/core";

import { appStore, useAppStore } from "@/stores/app";

/**
 * The IDs of the objects of a collection whose settings are expanded, to which
 * each object requested through the app store's `focusObject` is added
 *
 * @param kind - The collection whose requests to handle
 * @returns The expanded object IDs and a setter for them
 */
export function useExpandedObjectIds(
  kind: FocusedObject["kind"],
): [string[], (expandedIds: string[]) => void] {
  const [expandedIds, setExpandedIds] = useState<string[]>([]);
  useEffect(
    () =>
      appStore.subscribe(({ focusedObject }) => {
        if (focusedObject !== null && focusedObject.kind === kind) {
          setExpandedIds((ids) =>
            ids.includes(focusedObject.id) ? ids : [...ids, focusedObject.id],
          );
        }
      }),
    [kind],
  );
  return [expandedIds, setExpandedIds];
}

/**
 * Activates the dockview panel of the collection of the object whose settings
 * are to be brought to the front, as requested through the app store's
 * `focusObject`, and clears the request
 *
 * The panels of the collections are the `<kind>Panel` dockview panels, e.g.
 * `imagesPanel`, which expand the object's settings themselves (see
 * {@link useExpandedObjectIds}).
 *
 * @param dockviewApi - The API of the ready dockview, or `null` while it is not
 * ready yet
 */
export function useFocusedObjectPanel(dockviewApi: DockviewApi | null): void {
  const focusedObject = useAppStore((state) => state.focusedObject);
  useEffect(() => {
    if (dockviewApi !== null && focusedObject !== null) {
      const panel = dockviewApi.getPanel(`${focusedObject.kind}Panel`);
      // dockview re-opens a panel that is activated while already active, which
      // re-attaches its content and resets the scroll positions within
      if (panel !== undefined && !panel.api.isActive) {
        panel.api.setActive();
      }
      appStore.setState({ focusedObject: null });
    }
  }, [dockviewApi, focusedObject]);
}
