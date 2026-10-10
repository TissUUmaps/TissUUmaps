import { type ReactNode, useRef, useState } from "react";

import { useAlertDialog } from "@/components/dialogs/AlertDialog/hooks";
import { endFileDrag, useIsFileDragActive } from "@/hooks/useFileDrag";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";

import { usePanelDrop } from "./hooks";

/**
 * How many milliseconds a drag has to stay on an overlay before its `onRest`
 * is called
 */
const restDelay = 500;

export type PanelDropTargetProps = {
  panelId: string;
  /** A panel's tab header is only highlighted, its content is labelled */
  variant: "tab" | "content";
  children: ReactNode;
};

/**
 * Lets files and directories be dropped on a panel's tab header or content
 * (see `usePanelDrop`)
 *
 * While files are dragged, an overlay covers the target if the panel accepts
 * them; on the content, it names the target and, while hovered, what it
 * accepts. A drop on a tab header, or a drag staying on it, brings the panel
 * to the front.
 */
export function PanelDropTarget({
  panelId,
  variant,
  children,
}: PanelDropTargetProps) {
  const isFileDragActive = useIsFileDragActive();
  const drop = usePanelDrop(panelId);
  const setActivePanelId = useAppStore((state) => state.setActivePanelId);
  const alert = useAlertDialog();

  return (
    <div className="relative size-full">
      {children}
      {drop !== null && isFileDragActive && (
        <PanelDropOverlay
          label={variant === "content" ? drop.label : undefined}
          accepts={drop.accepts}
          onRest={
            variant === "tab" ? () => setActivePanelId(panelId) : undefined
          }
          onDrop={(dataTransfer) => {
            endFileDrag();
            if (variant === "tab") {
              setActivePanelId(panelId);
            }
            drop.onDrop(dataTransfer).catch((error: unknown) => {
              console.error("Failed to handle the dropped items", error);
              void alert({
                title: "Cannot handle the dropped items",
                body: error instanceof Error ? error.message : String(error),
              });
            });
          }}
        />
      )}
    </div>
  );
}

type PanelDropOverlayProps = {
  label: string | undefined;
  accepts: string[];
  onRest?: () => void;
  onDrop: (dataTransfer: DataTransfer) => void;
};

/**
 * Covers a drop target while files are dragged, highlighted while hovered
 *
 * Rendered only during a drag, so that its hover state starts afresh with
 * every drag.
 */
function PanelDropOverlay({
  label,
  accepts,
  onRest,
  onDrop,
}: PanelDropOverlayProps) {
  const [isOver, setOver] = useState(false);
  // when the drag entered the overlay, measured with the events' timestamps,
  // as browsers keep firing `dragover` while it stays; `null` while not over
  // the overlay, or once rested
  const enteredAtRef = useRef<number | null>(null);

  return (
    <div
      className={cn(
        "absolute inset-0 z-10 flex flex-col items-center justify-center gap-1 rounded-md border-2 border-dashed border-primary/50 bg-background/80 p-2 text-center text-sm *:pointer-events-none",
        isOver && "border-primary bg-primary/10",
      )}
      onDragEnter={(event) => {
        event.preventDefault();
        setOver(true);
        enteredAtRef.current = event.timeStamp;
      }}
      onDragOver={(event) => {
        event.preventDefault();
        event.stopPropagation();
        event.dataTransfer.dropEffect = "copy";
        const enteredAt = enteredAtRef.current;
        if (enteredAt !== null && event.timeStamp - enteredAt >= restDelay) {
          enteredAtRef.current = null;
          onRest?.();
        }
      }}
      onDragLeave={() => {
        setOver(false);
        enteredAtRef.current = null;
      }}
      onDrop={(event) => {
        event.preventDefault();
        event.stopPropagation();
        onDrop(event.dataTransfer);
      }}
    >
      {label !== undefined && (
        <>
          <span className="font-medium">{label}</span>
          {isOver && (
            <span className="text-muted-foreground">{accepts.join(", ")}</span>
          )}
        </>
      )}
    </div>
  );
}
