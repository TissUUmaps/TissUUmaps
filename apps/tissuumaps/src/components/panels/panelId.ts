/** The IDs of the application's dockview panels */
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
