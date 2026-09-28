/** The IDs of the application's dockview panels */
export const PanelId = {
  viewer: "viewerPanel",
  project: "projectPanel",
  images: "imagesPanel",
  labels: "labelsPanel",
  points: "pointsPanel",
  shapes: "shapesPanel",
  tables: "tablesPanel",
} as const;

export type PanelId = (typeof PanelId)[keyof typeof PanelId];
