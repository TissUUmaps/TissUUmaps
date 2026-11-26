export const LabelsSettingsCategory = {
  layer: "layer",
  transform: "transform",
  labelColor: "labelColor",
  labelVisibility: "labelVisibility",
  labelOpacity: "labelOpacity",
} as const;

export type LabelsSettingsCategory =
  (typeof LabelsSettingsCategory)[keyof typeof LabelsSettingsCategory];
