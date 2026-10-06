import type { DragEndEvent } from "@dnd-kit/react";
import { isSortable } from "@dnd-kit/react/sortable";

/**
 * Lists a project collection topmost first, and moves its items by dragging
 *
 * Collections are drawn in order, so their last item is on top. The list shows
 * them the other way around, and the drag handler converts the list index back
 * to the collection index.
 *
 * @param items - The collection, in draw order
 * @param moveItem - Moves an item of the collection to another index
 * @returns The items topmost first, and the `onDragEnd` handler of their
 * `DragDropProvider`
 */
export function useTopFirstSortable<T>(
  items: T[],
  moveItem: (itemId: string, newIndex: number) => void,
): { topFirstItems: T[]; onDragEnd: (event: DragEndEvent) => void } {
  return {
    topFirstItems: items.toReversed(),
    onDragEnd: (event) => {
      const { source, canceled } = event.operation;
      if (isSortable(source) && !canceled) {
        // dnd-kit optimistically updates the DOM
        // https://github.com/clauderic/dnd-kit/issues/1564
        moveItem(source.id as string, items.length - 1 - source.index);
      }
    },
  };
}
