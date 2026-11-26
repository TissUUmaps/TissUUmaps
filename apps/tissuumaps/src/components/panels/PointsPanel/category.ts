export const PointsSettingsCategory = {
  general: "general",
  transform: "transform",
  pointMarker: "pointMarker",
  pointSize: "pointSize",
  pointColor: "pointColor",
  pointVisibility: "pointVisibility",
  pointOpacity: "pointOpacity",
  pointOrder: "pointOrder",
};

export type PointsSettingsCategory =
  (typeof PointsSettingsCategory)[keyof typeof PointsSettingsCategory];
