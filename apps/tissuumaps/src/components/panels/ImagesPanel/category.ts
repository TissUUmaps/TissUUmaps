export const ImageSettingsCategory = {
  transform: "transform",
} as const;

export type ImageSettingsCategory =
  (typeof ImageSettingsCategory)[keyof typeof ImageSettingsCategory];
