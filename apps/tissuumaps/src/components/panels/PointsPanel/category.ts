export const PointsSettingsCategory = {
  transform: "transform",
  pointMarker: "pointMarker",
  pointSize: "pointSize",
  pointColor: "pointColor",
  pointVisibility: "pointVisibility",
  pointOpacity: "pointOpacity",
  pointOrder: "pointOrder",
} as const;

export type PointsSettingsCategory =
  (typeof PointsSettingsCategory)[keyof typeof PointsSettingsCategory];
