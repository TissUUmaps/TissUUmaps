import { useEffect, useState } from "react";

import type { FocusedObject } from "@tissuumaps/core";

import { appStore } from "@/stores/app";

/**
 * Keeps the IDs of the objects of a collection whose settings are expanded,
 * adding each object requested through the app store's `focusObject`
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
