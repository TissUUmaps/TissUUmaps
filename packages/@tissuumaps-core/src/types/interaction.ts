/**
 * Interaction modes, determining how mouse events are interpreted
 */
export const InteractionMode = {
  pan: "pan",
  drawRectangle: "drawRectangle",
  drawPolygon: "drawPolygon",
  drawFreehand: "drawFreehand",
} as const;

/** One of the interaction modes of the {@link InteractionMode} object */
export type InteractionMode =
  (typeof InteractionMode)[keyof typeof InteractionMode];
