export const ShapesSettingsCategory = {
  transform: "transform",
  shapeVisibility: "shapeVisibility",
  shapeOpacity: "shapeOpacity",
  shapeFillColor: "shapeFillColor",
  shapeFillVisibility: "shapeFillVisibility",
  shapeFillOpacity: "shapeFillOpacity",
  shapeStrokeColor: "shapeStrokeColor",
  shapeStrokeVisibility: "shapeStrokeVisibility",
  shapeStrokeOpacity: "shapeStrokeOpacity",
} as const;

export type ShapesSettingsCategory =
  (typeof ShapesSettingsCategory)[keyof typeof ShapesSettingsCategory];
