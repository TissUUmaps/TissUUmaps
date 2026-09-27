export const ViewerControlAnchor = {
  NONE: 0,
  TOP_LEFT: 1,
  TOP_RIGHT: 2,
  BOTTOM_LEFT: 3,
  BOTTOM_RIGHT: 4,
  ABSOLUTE: 5,
} as const;

export type ViewerControlAnchor =
  (typeof ViewerControlAnchor)[keyof typeof ViewerControlAnchor];
