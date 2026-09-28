import { useSortable } from "@dnd-kit/react/sortable";
import { GripVertical } from "lucide-react";

import { ObjectItem, type ObjectItemProps } from "./ObjectItem";

export type SortableObjectItemProps = Omit<
  ObjectItemProps,
  "handle" | "ref"
> & {
  index: number;
};

export function SortableObjectItem({
  index,
  ...props
}: SortableObjectItemProps) {
  const { ref, handleRef } = useSortable({ id: props.id, index });

  return (
    <ObjectItem
      ref={ref}
      handle={
        <button
          ref={handleRef}
          type="button"
          aria-label={`Reorder ${props.name || "Untitled"}`}
          className="text-muted-foreground/60 focus-visible:ring-ring/50 flex shrink-0 cursor-grab rounded-sm outline-none focus-visible:ring-[3px]"
        >
          <GripVertical className="size-4" />
        </button>
      }
      {...props}
    />
  );
}
