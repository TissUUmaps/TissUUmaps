import { type JsonSchema, computeLabel } from "@jsonforms/core";

import { Field, FieldError, FieldLabel } from "@/components/common/field";
import { cn } from "@/lib/utils";

import { SourceInput } from "./SourceInput";

export type SourceFieldProps = {
  schema: JsonSchema;
  value: string | undefined;
  onValueChange: (value: string | undefined) => void;
  showErrors?: boolean;
  disabled?: boolean;
  className?: string;
};

/**
 * A labelled input for the source of a data source (see {@link SourceInput})
 *
 * Rendered only if the data source's schema declares a source; the label marks
 * a required one. With `showErrors`, a missing required source is marked as
 * invalid as well.
 */
export function SourceField({
  schema,
  value,
  onValueChange,
  showErrors = false,
  disabled = false,
  className,
}: SourceFieldProps) {
  if (schema.properties?.source === undefined) {
    return null;
  }
  const required = schema.required?.includes("source") ?? false;
  const isMissing = showErrors && required && value === undefined;
  return (
    <Field invalid={isMissing} className={cn("flex flex-col gap-2", className)}>
      <FieldLabel>{computeLabel("Source", required, false)}</FieldLabel>
      <SourceInput
        value={value}
        onValueChange={onValueChange}
        disabled={disabled}
        invalid={isMissing}
      />
      {isMissing && <FieldError match>Required</FieldError>}
    </Field>
  );
}
