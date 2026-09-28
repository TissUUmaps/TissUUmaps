import {
  ChartScatterIcon,
  ImageIcon,
  type LucideIcon,
  ShapesIcon,
  TableIcon,
  TagsIcon,
} from "lucide-react";

/** The icon of each kind of project object, for every list that shows one */
export const objectKindIcons = {
  image: ImageIcon,
  labels: TagsIcon,
  points: ChartScatterIcon,
  shapes: ShapesIcon,
  table: TableIcon,
} satisfies Record<string, LucideIcon>;

/** A kind of project object */
export type ObjectKind = keyof typeof objectKindIcons;
