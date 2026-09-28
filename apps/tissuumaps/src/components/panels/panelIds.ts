/** The IDs of the application's dockview panels */
export const panelIds = {
  viewer: "viewerPanel",
  project: "projectPanel",
  images: "imagesPanel",
  labels: "labelsPanel",
  points: "pointsPanel",
  shapes: "shapesPanel",
  tables: "tablesPanel",
} as const;

export type PanelId = (typeof panelIds)[keyof typeof panelIds];
