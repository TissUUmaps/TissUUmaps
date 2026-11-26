export const ImageSettingsCategory = {
  layer: "layer",
  transform: "transform",
} as const;

export type ImageSettingsCategory =
  (typeof ImageSettingsCategory)[keyof typeof ImageSettingsCategory];
