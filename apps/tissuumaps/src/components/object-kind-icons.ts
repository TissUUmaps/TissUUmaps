import {
  ChartScatterIcon,
  ImageIcon,
  type LucideIcon,
  ShapesIcon,
  TableIcon,
  TagsIcon,
} from "lucide-react";

/** The kinds of project objects */
export const ObjectKind = {
  image: "image",
  labels: "labels",
  points: "points",
  shapes: "shapes",
  table: "table",
} as const;

/** One of the kinds of the {@link ObjectKind} object */
export type ObjectKind = (typeof ObjectKind)[keyof typeof ObjectKind];

/** The icon of each kind of project object, for every list that shows one */
export const objectKindIcons: Record<ObjectKind, LucideIcon> = {
  [ObjectKind.image]: ImageIcon,
  [ObjectKind.labels]: TagsIcon,
  [ObjectKind.points]: ChartScatterIcon,
  [ObjectKind.shapes]: ShapesIcon,
  [ObjectKind.table]: TableIcon,
};
