/** A named mapping from string keys (groups) to typed values with an optional default value */
export type GroupValueMap<TValue> = {
  /** Map ID, referenced from data object configurations */
  id: string;

  /** Human-readable map name */
  name: string;

  /**
   * Mapping from group names to values
   *
   * Group names are the table cell values as strings, e.g. `"3"` for a numeric
   * cell.
   */
  values: { [key: string]: TValue };

  /** Default value for groups not present in {@link values} */
  default?: TValue;
};

/** Coordinate space */
export type CoordinateSpace =
  /** Data (e.g. pixel) space */
  | "data"
  /** Layer (e.g. physical) space */
  | "layer"
  /** World (i.e. global) space */
  | "world";

/** Similarity transform (uniform scale, rotation, translation, and flip) */
export type SimilarityTransform = {
  /** Horizontal reflection, applied before scaling, rotation, and translation */
  flip: boolean;

  /** Uniform scale factor (1 = no scaling) */
  scale: number;

  /** Rotation around origin, in degrees */
  rotation: number;

  /** Translation in X and Y, applied after scaling and rotation */
  translation: { x: number; y: number };
};

/** The marker shapes that points can be rendered as */
export const Marker = {
  Cross: 0,
  Diamond: 1,
  Square: 2,
  TriangleUp: 3,
  Star: 4,
  Clobber: 5,
  Disc: 6,
  HBar: 7,
  VBar: 8,
  TailedArrow: 9,
  TriangleDown: 10,
  Ring: 11,
  X: 12,
  Arrow: 13,
  Gaussian: 14,
} as const;

/** One of the marker shapes of the {@link Marker} object, as its index */
export type Marker = (typeof Marker)[keyof typeof Marker];

/** A color in RGB format */
export type Color = {
  /** Red component, between 0 and 255 */
  r: number;

  /** Green component, between 0 and 255 */
  g: number;

  /** Blue component, between 0 and 255 */
  b: number;
};
