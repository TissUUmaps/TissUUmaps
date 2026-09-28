import { DragDropProvider } from "@dnd-kit/react";
import type { ReactNode } from "react";

import { Accordion } from "@/components/common/accordion";
import { useTopFirstSortable } from "@/hooks/useTopFirstSortable";
import { cn } from "@/lib/utils";

export { ObjectListItem, SortableObjectListItem } from "./ObjectListItem";

export type ObjectListProps = {
  children: ReactNode;
  className?: string;
};

export function ObjectList({ children, className }: ObjectListProps) {
  return (
    <Accordion multiple className={cn("gap-1", className)}>
      {children}
    </Accordion>
  );
}

export type SortableObjectListProps<TObject> = {
  objects: TObject[];
  onMove: (objectId: string, newIndex: number) => void;
  children: (object: TObject, index: number) => ReactNode;
  className?: string;
};

export function SortableObjectList<TObject>({
  objects,
  onMove,
  children,
  className,
}: SortableObjectListProps<TObject>) {
  const { topFirstItems, onDragEnd } = useTopFirstSortable(objects, onMove);

  return (
    <DragDropProvider onDragEnd={onDragEnd}>
      <ObjectList className={className}>
        {topFirstItems.map((object, index) => children(object, index))}
      </ObjectList>
    </DragDropProvider>
  );
}
