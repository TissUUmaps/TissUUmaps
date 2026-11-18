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
export type AskDialogParams = Omit<
  AskDialogProps,
  "open" | "onCancel" | "onConfirm"
>;

export type AskDialogProps = {
  title: string;
  body?: ReactNode;
  /** The actions to choose from, the last one the primary action */
  actions: { value: string; label: string }[];
  cancelLabel?: string;
  /** Whether the dialog is shown. */
  open: boolean;
  /** Called when the user cancels the dialog (cancel button or escape). */
  onCancel: () => void;
  /** Called with the value of the chosen action. */
  onConfirm: (value: string) => void;
};

/** Dialog with a cancel button and a button for each of several actions. */
export function AskDialog({
  title,
  body,
  actions,
  cancelLabel,
  open,
  onCancel,
  onConfirm,
}: AskDialogProps) {
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
          {actions.map((action, i) => (
            <AlertDialogAction
              key={action.value}
              variant={i < actions.length - 1 ? "secondary" : "default"}
              onClick={() => onConfirm(action.value)}
            >
              {action.label}
            </AlertDialogAction>
          ))}
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
