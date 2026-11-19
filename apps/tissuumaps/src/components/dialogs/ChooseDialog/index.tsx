import { type ReactNode, useState } from "react";

import { Field, FieldItem, FieldLabel } from "@/components/common/field";
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
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";

// The caller-facing fields: everything except the open state and the
// callbacks, which are owned by the DialogProvider.
export type ChooseDialogParams = Omit<
  ChooseDialogProps,
  "open" | "onCancel" | "onConfirm"
>;

export type ChooseDialogProps = {
  title: string;
  body?: ReactNode;
  options: { value: string; label: string }[];
  cancelLabel?: string;
  actionLabel?: string;
  /** Whether the dialog is shown. */
  open: boolean;
  /** Called when the user cancels the dialog (cancel button or escape). */
  onCancel: () => void;
  /** Called with the value of the chosen option. */
  onConfirm: (value: string) => void;
};

/** Dialog for choosing one of several options, with a cancel and an action button. */
export function ChooseDialog({
  title,
  body,
  options,
  cancelLabel,
  actionLabel,
  open,
  onCancel,
  onConfirm,
}: ChooseDialogProps) {
  const [value, setValue] = useState(options[0]?.value);

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
        <Field>
          <RadioGroup
            value={value}
            onValueChange={(value) => setValue(value as string)}
          >
            {options.map((option) => (
              <FieldItem
                key={option.value}
                className="flex items-center gap-x-2"
              >
                <RadioGroupItem value={option.value} />
                <FieldLabel>{option.label}</FieldLabel>
              </FieldItem>
            ))}
          </RadioGroup>
        </Field>
        <AlertDialogFooter>
          <AlertDialogCancel>{cancelLabel ?? "Cancel"}</AlertDialogCancel>
          <AlertDialogAction
            disabled={value === undefined}
            onClick={() => {
              if (value !== undefined) {
                onConfirm(value);
              }
            }}
          >
            {actionLabel ?? "OK"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
