export const ImageSettingsCategory = {
  general: "general",
  transform: "transform",
};

export type ImageSettingsCategory =
  (typeof ImageSettingsCategory)[keyof typeof ImageSettingsCategory];
