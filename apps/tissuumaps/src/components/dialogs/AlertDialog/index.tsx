import type { ReactNode } from "react";

import {
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialog as AlertDialogRoot,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

// The caller-facing fields: everything except the open state and the
// callbacks, which are owned by the DialogProvider.
export type AlertDialogParams = Omit<AlertDialogProps, "open" | "onDismiss">;

export type AlertDialogProps = {
  title: string;
  body?: ReactNode;
  actionLabel?: string;
  /** Whether the dialog is shown. */
  open: boolean;
  /** Called when the user dismisses the dialog (action button or escape). */
  onDismiss: () => void;
};

/** Acknowledgement dialog with a single button. */
export function AlertDialog({
  title,
  body,
  actionLabel,
  open,
  onDismiss,
}: AlertDialogProps) {
  return (
    <AlertDialogRoot
      open={open}
      onOpenChange={(open) => {
        if (!open) onDismiss();
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          {body ? (
            <AlertDialogDescription>{body}</AlertDialogDescription>
          ) : null}
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogAction onClick={onDismiss}>
            {actionLabel ?? "OK"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialogRoot>
  );
}
