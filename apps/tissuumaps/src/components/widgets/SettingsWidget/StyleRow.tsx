import type { ReactNode } from "react";

import { Field, FieldLabel } from "@/components/common/field";
import { cn } from "@/lib/utils";

export type StyleRowProps = {
  /** The label of the row, if any */
  label?: string;
  /** Whether the row holds the value in use, e.g. the colored column */
  isActive?: boolean;
  children: ReactNode;
  /** Classes of the control cell */
  className?: string;
};

/**
 * A row of the settings widget: a muted label and its controls, in a grid of a
 * label column and a control column
 *
 * The controls are kept compact (`h-8`) by their containers.
 */
export function StyleRow({
  label,
  isActive = false,
  children,
  className,
}: StyleRowProps) {
  return (
    <Field className="contents">
      {label !== undefined ? (
        <FieldLabel
          className={cn(
            "truncate text-xs",
            isActive ? "font-medium text-foreground" : "text-muted-foreground",
          )}
        >
          {label}
          {isActive && <span className="sr-only"> (active)</span>}
        </FieldLabel>
      ) : (
        <span />
      )}
      <div className={cn("min-w-0 [&_[data-slot=input-group]]:h-8", className)}>
        {children}
      </div>
    </Field>
  );
}
