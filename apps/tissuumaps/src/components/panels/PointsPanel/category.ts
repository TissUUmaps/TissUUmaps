export const PointsSettingsCategory = {
  layer: "layer",
  transform: "transform",
  pointMarker: "pointMarker",
  pointSize: "pointSize",
  pointColor: "pointColor",
  pointVisibility: "pointVisibility",
  pointOpacity: "pointOpacity",
} as const;

export type PointsSettingsCategory =
  (typeof PointsSettingsCategory)[keyof typeof PointsSettingsCategory];
