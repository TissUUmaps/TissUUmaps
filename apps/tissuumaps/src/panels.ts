/** The IDs of the application's built-in dockview panels */
export const PanelId = {
  viewer: "viewer",
  project: "project",
  images: "images",
  labels: "labels",
  points: "points",
  shapes: "shapes",
  tables: "tables",
} as const;

export type PanelId = (typeof PanelId)[keyof typeof PanelId];

/**
 * The prefix of the dockview panel ID of a plugin's panel, followed by the
 * plugin's ID
 *
 * The prefix identifies the plugin panels among the dockview panels, see
 * {@link isPluginPanelId}.
 */
const pluginPanelIdPrefix = "plugin:";

/**
 * Gets the dockview panel ID of a plugin's panel
 *
 * @param pluginId - The ID of the plugin
 * @returns The ID of the plugin's panel, which is only in the dockview layout
 * while the plugin is mounted
 */
export function getPluginPanelId(pluginId: string): string {
  return pluginPanelIdPrefix + pluginId;
}

/**
 * Checks whether a dockview panel ID is that of a plugin's panel
 *
 * @param panelId - The ID of the panel
 * @returns Whether the panel is a plugin's panel, see {@link getPluginPanelId}
 */
export function isPluginPanelId(panelId: string): boolean {
  return panelId.startsWith(pluginPanelIdPrefix);
}
