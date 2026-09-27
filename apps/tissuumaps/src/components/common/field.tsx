import { Field as FieldPrimitive } from "@base-ui/react/field";

import { cn } from "@/lib/utils";

export function Field(props: FieldPrimitive.Root.Props) {
  return <FieldPrimitive.Root {...props} />;
}

export function FieldLabel({
  className,
  ...props
}: FieldPrimitive.Label.Props) {
  return (
    <FieldPrimitive.Label
      className={cn("text-foreground data-disabled:opacity-50", className)}
      {...props}
    />
  );
}

export function FieldControl(props: FieldPrimitive.Control.Props) {
  return <FieldPrimitive.Control {...props} />;
}

export function FieldDescription(props: FieldPrimitive.Description.Props) {
  return <FieldPrimitive.Description {...props} />;
}

export function FieldItem(props: FieldPrimitive.Item.Props) {
  return <FieldPrimitive.Item {...props} />;
}

export function FieldError(props: FieldPrimitive.Error.Props) {
  return <FieldPrimitive.Error {...props} />;
}

export function FieldValidity(props: FieldPrimitive.Validity.Props) {
  return <FieldPrimitive.Validity {...props} />;
}
