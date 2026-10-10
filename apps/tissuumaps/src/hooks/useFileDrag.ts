import { useEffect, useSyncExternalStore } from "react";

/**
 * How many elements a file drag has entered and not left yet, which is
 * positive while files are dragged over the window
 */
let fileDragDepth = 0;

/** Called whenever files start or stop being dragged over the window */
const listeners = new Set<() => void>();

/** Whether a drag carries files or directories, rather than e.g. a panel */
function isFileDrag(event: DragEvent): boolean {
  return event.dataTransfer?.types.includes("Files") ?? false;
}

/**
 * Subscribes to the start and end of file drags (see `useSyncExternalStore`)
 *
 * Defined once at module level, as `useSyncExternalStore` resubscribes
 * whenever it receives a different function, i.e. on every render if the
 * function were created in the hook.
 *
 * @param listener - Called whenever files start or stop being dragged
 * @returns A function that unsubscribes the listener again
 */
function subscribeToFileDrag(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function setFileDragDepth(depth: number): void {
  const wasDragging = fileDragDepth > 0;
  fileDragDepth = depth;
  if (wasDragging !== fileDragDepth > 0) {
    listeners.forEach((listener) => listener());
  }
}

/**
 * Ends the file drag, e.g. once a drop target has handled the drop, which it
 * is no longer rendered for afterwards
 */
export function endFileDrag(): void {
  setFileDragDepth(0);
}

/**
 * Tracks whether files are dragged over the window (see
 * {@link useIsFileDragActive}), while the calling component is mounted
 *
 * Files dropped outside of a drop target are ignored, rather than opened by the
 * browser. Drop targets have to stop the propagation of `dragover`, or files
 * cannot be dropped on them. Drop targets that also stop the propagation of
 * `drop` have to end the file drag themselves. Used once, by the application's
 * root component.
 */
export function useFileDragTracking(): void {
  useEffect(() => {
    const onDragEnter = (event: DragEvent) => {
      if (isFileDrag(event)) {
        setFileDragDepth(fileDragDepth + 1);
      }
    };
    const onDragLeave = (event: DragEvent) => {
      if (isFileDrag(event)) {
        setFileDragDepth(Math.max(fileDragDepth - 1, 0));
      }
    };
    const onDragOver = (event: DragEvent) => {
      if (isFileDrag(event) && event.dataTransfer !== null) {
        event.preventDefault();
        event.dataTransfer.dropEffect = "none";
      }
    };
    const onDrop = (event: DragEvent) => {
      if (isFileDrag(event)) {
        event.preventDefault();
        endFileDrag();
      }
    };
    const onPointerMove = () => {
      if (fileDragDepth > 0) {
        endFileDrag();
      }
    };
    window.addEventListener("dragenter", onDragEnter, true);
    window.addEventListener("dragleave", onDragLeave, true);
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("drop", onDrop);
    window.addEventListener("pointermove", onPointerMove);
    return () => {
      window.removeEventListener("dragenter", onDragEnter, true);
      window.removeEventListener("dragleave", onDragLeave, true);
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("drop", onDrop);
      window.removeEventListener("pointermove", onPointerMove);
      endFileDrag();
    };
  }, []);
}

/**
 * Returns whether files or directories are being dragged over the window
 *
 * Which files they are is only known once they are dropped. Tracked while
 * {@link useFileDragTracking} is mounted.
 *
 * @returns `true` while files are dragged over the window
 */
export function useIsFileDragActive(): boolean {
  return useSyncExternalStore(subscribeToFileDrag, () => fileDragDepth > 0);
}
