import type { ReactNode } from "react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

// The caller-facing fields: everything except the open state and the
// callbacks, which are owned by the DialogProvider.
export type ConfirmDialogParams = Omit<
  ConfirmDialogProps,
  "open" | "onCancel" | "onConfirm"
>;

export type ConfirmDialogProps = {
  title: string;
  body?: ReactNode;
  cancelLabel?: string;
  actionLabel?: string;
  /** Whether the dialog is shown. */
  open: boolean;
  /** Called when the user cancels the dialog (cancel button or escape). */
  onCancel: () => void;
  /** Called when the user clicks the action button. */
  onConfirm: () => void;
};

/** Confirmation dialog with a cancel and an action button. */
export function ConfirmDialog({
  title,
  body,
  cancelLabel,
  actionLabel,
  open,
  onCancel,
  onConfirm,
}: ConfirmDialogProps) {
  return (
    <AlertDialog
      open={open}
      onOpenChange={(open) => {
        if (!open) onCancel();
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
          <AlertDialogCancel>{cancelLabel ?? "Cancel"}</AlertDialogCancel>
          <AlertDialogAction onClick={onConfirm}>
            {actionLabel ?? "OK"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
