import type { DockviewApi } from "dockview-react";
import { useEffect } from "react";

import { getPluginPanelId, isPluginPanelId } from "@/panels";
import { useAppStore } from "@/stores/app";

/**
 * Keeps the plugin panels in the dockview layout in sync with the plugins
 * mounted according to the app store
 *
 * A plugin's panel is shown for exactly as long as the plugin is mounted:
 * mounting a plugin adds a panel to the dockview layout, made active since
 * plugins are only mounted on request, and unmounting or unregistering the
 * plugin removes it again.
 *
 * The panels are shown using the `PluginPanel` component and the
 * `PluginPanelHeader` tab component, both of which the application registers
 * with dockview.
 *
 * @param dockviewApi - The API of the ready dockview, or `null` while it is not
 * ready yet
 * @param referencePanelId - The ID of the panel into whose group the plugin
 * panels are added
 */
export function usePluginPanels(
  dockviewApi: DockviewApi | null,
  referencePanelId: string,
): void {
  const plugins = useAppStore((state) => state.plugins);

  useEffect(() => {
    if (dockviewApi === null) {
      return;
    }
    const pluginPanels = new Map<string, { id: string; name: string }>();
    for (const [pluginId, plugin] of plugins) {
      if (plugin.container !== undefined) {
        pluginPanels.set(getPluginPanelId(pluginId), {
          id: pluginId,
          name: plugin.name,
        });
      }
    }
    for (const dockviewPanel of dockviewApi.panels) {
      if (
        isPluginPanelId(dockviewPanel.id) &&
        !pluginPanels.has(dockviewPanel.id)
      ) {
        dockviewApi.removePanel(dockviewPanel);
      }
    }
    const referenceGroup = dockviewApi.getPanel(referencePanelId)?.group;
    for (const [id, plugin] of pluginPanels) {
      const dockviewPanel = dockviewApi.getPanel(id);
      if (dockviewPanel === undefined) {
        dockviewApi.addPanel<{ pluginId: string }>({
          id,
          title: plugin.name,
          component: "PluginPanel",
          tabComponent: "PluginPanelHeader",
          params: { pluginId: plugin.id },
          position:
            referenceGroup !== undefined ? { referenceGroup } : undefined,
        });
      } else if (dockviewPanel.title !== plugin.name) {
        dockviewPanel.api.setTitle(plugin.name);
      }
    }
  }, [dockviewApi, plugins, referencePanelId]);
}
