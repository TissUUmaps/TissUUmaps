import type { DockviewApi } from "dockview-react";
import { useEffect } from "react";

import { useAppStore } from "@/stores/app";

/**
 * The prefix of the dockview panel ID of a plugin's panel, followed by the
 * plugin's ID
 *
 * The prefix identifies the plugin panels among the dockview panels, so that
 * {@link usePluginPanels} only ever adds and removes its own.
 */
const pluginPanelIdPrefix = "plugin:";

/**
 * Keeps the plugin panels in the dockview layout in sync with the plugins
 * registered with the app store
 *
 * A plugin's panel is shown for exactly as long as the plugin is registered:
 * registering a plugin that has a `mount` adds a panel to the dockview layout,
 * and unregistering the plugin removes it again. A panel is not made active
 * when it is added, so that a plugin registering during startup does not take
 * the group it is added to over.
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
        pluginPanels.set(pluginPanelIdPrefix + pluginId, {
          id: pluginId,
          name: plugin.name,
        });
      }
    }
    for (const dockviewPanel of dockviewApi.panels) {
      if (
        dockviewPanel.id.startsWith(pluginPanelIdPrefix) &&
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
          // do not steal the active tab from whatever the user is looking at
          inactive: true,
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
