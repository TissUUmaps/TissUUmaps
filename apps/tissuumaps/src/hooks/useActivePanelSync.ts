import type { DockviewApi } from "dockview-react";
import { useEffect, useRef } from "react";

import { PanelId } from "@/panels";
import { useAppStore } from "@/stores/app";

/**
 * Finds a panel whose tab is in front, other than the viewer
 *
 * @param dockviewApi - The API of the ready dockview
 * @returns The ID of the active panel or, while the viewer is active, of the
 * panel in front of any group, or `null` if there is none
 */
function findFrontPanelId(dockviewApi: DockviewApi): string | null {
  const panels = [
    dockviewApi.activePanel,
    ...dockviewApi.groups.map((group) => group.activePanel),
  ];
  const frontPanel = panels.find(
    (panel) => panel !== undefined && panel.id !== PanelId.viewer,
  );
  return frontPanel?.id ?? null;
}

/**
 * Keeps the app store's `activePanelId` in sync with dockview's active panel,
 * i.e. the panel in front of the active group, in both directions
 *
 * The viewer has no tab and is never the active panel ID: while it is active,
 * the ID keeps the panel that was active before. Setting the ID activates that
 * panel; if it is in front of its group already, only its group is activated,
 * as dockview would otherwise re-attach its content and reset its scroll
 * positions. An ID that identifies no panel is reverted, leaving the layout
 * unchanged, and the ID of a removed panel that dockview does not replace is
 * replaced by that of a panel in front.
 *
 * @param dockviewApi - The API of the ready dockview, or `null` until it is
 * ready
 */
export function useActivePanelSync(dockviewApi: DockviewApi | null): void {
  const activePanelId = useAppStore((state) => state.activePanelId);
  const setActivePanelId = useAppStore((state) => state.setActivePanelId);

  // the last active panel ID known to agree with the layout, which therefore
  // needs no syncing to it
  const syncedPanelIdRef = useRef<string | null>(null);

  // from the layout to the store
  useEffect(() => {
    if (dockviewApi === null) {
      return;
    }
    const disposable = dockviewApi.onDidActivePanelChange(({ panel }) => {
      if (panel !== undefined && panel.id !== PanelId.viewer) {
        syncedPanelIdRef.current = panel.id;
        setActivePanelId(panel.id);
      }
    });
    return () => disposable.dispose();
  }, [dockviewApi, setActivePanelId]);

  // from the store to the layout, including an ID set before it was ready
  useEffect(() => {
    if (dockviewApi === null || activePanelId === syncedPanelIdRef.current) {
      return;
    }
    const panel = dockviewApi.getPanel(activePanelId);
    if (panel === undefined || panel.id === PanelId.viewer) {
      const syncedPanelId = syncedPanelIdRef.current;
      const revertedPanelId =
        syncedPanelId !== null &&
        dockviewApi.getPanel(syncedPanelId) !== undefined
          ? syncedPanelId
          : findFrontPanelId(dockviewApi);
      if (revertedPanelId !== null) {
        syncedPanelIdRef.current = revertedPanelId;
        setActivePanelId(revertedPanelId);
      }
      return;
    }
    if (!panel.api.isVisible) {
      panel.api.setActive();
    } else if (!panel.api.isActive) {
      panel.group.api.setActive();
    }
    syncedPanelIdRef.current = activePanelId;
  }, [dockviewApi, activePanelId, setActivePanelId]);

  useEffect(() => {
    if (dockviewApi === null) {
      return;
    }
    const disposable = dockviewApi.onDidMutateLayout(() => {
      if (dockviewApi.getPanel(activePanelId) === undefined) {
        const frontPanelId = findFrontPanelId(dockviewApi);
        if (frontPanelId !== null) {
          syncedPanelIdRef.current = frontPanelId;
          setActivePanelId(frontPanelId);
        }
      }
    });
    return () => disposable.dispose();
  }, [dockviewApi, activePanelId, setActivePanelId]);
}
