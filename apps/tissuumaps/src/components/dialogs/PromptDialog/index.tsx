import { type ComponentProps, type ReactNode, useRef } from "react";

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
import { Input } from "@/components/ui/input";

// The caller-facing fields: everything except the open state and the
// callbacks, which are owned by the DialogProvider.
export type PromptDialogParams = Omit<
  PromptDialogProps,
  "open" | "onCancel" | "onConfirm"
>;

export type PromptDialogProps = {
  title: string;
  body?: ReactNode;
  cancelLabel?: string;
  actionLabel?: string;
  defaultValue?: string;
  /** Extra props for the input, except those owned by the dialog itself. */
  inputProps?: Omit<
    ComponentProps<"input">,
    "ref" | "name" | "defaultValue" | "value" | "onChange"
  >;
  /** Whether the dialog is shown. */
  open: boolean;
  /** Called when the user cancels the dialog (cancel button or escape). */
  onCancel: () => void;
  /** Called with the input value when the form is submitted. */
  onConfirm: (value: string) => void;
};

/** Text-input dialog with a cancel and a submit button. */
export function PromptDialog({
  title,
  body,
  cancelLabel,
  actionLabel,
  defaultValue,
  inputProps,
  open,
  onCancel,
  onConfirm,
}: PromptDialogProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <AlertDialog
      open={open}
      onOpenChange={(open) => {
        if (!open) onCancel();
      }}
    >
      <AlertDialogContent>
        <form
          className="grid gap-6"
          onSubmit={(event) => {
            event.preventDefault();
            onConfirm(inputRef.current?.value ?? "");
          }}
        >
          <AlertDialogHeader>
            <AlertDialogTitle>{title}</AlertDialogTitle>
            {body ? (
              <AlertDialogDescription>{body}</AlertDialogDescription>
            ) : null}
          </AlertDialogHeader>
          <Input
            {...inputProps}
            aria-label={title}
            ref={inputRef}
            name="prompt"
            defaultValue={defaultValue}
          />
          <AlertDialogFooter>
            <AlertDialogCancel type="button">
              {cancelLabel ?? "Cancel"}
            </AlertDialogCancel>
            <AlertDialogAction type="submit">
              {actionLabel ?? "OK"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </form>
      </AlertDialogContent>
    </AlertDialog>
  );
}
