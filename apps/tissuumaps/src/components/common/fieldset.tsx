import { Fieldset as FielsetPrimitive } from "@base-ui/react/fieldset";

import { cn } from "@/lib/utils";

export function Fieldset({ className, ...props }: FielsetPrimitive.Root.Props) {
  return (
    // browsers give a fieldset a minimum width of its content, which would keep
    // wide content from scrolling inside it
    <FielsetPrimitive.Root className={cn("min-w-0", className)} {...props} />
  );
}

export function FieldsetLegend({
  className,
  ...props
}: FielsetPrimitive.Legend.Props) {
  return (
    <FielsetPrimitive.Legend
      className={cn("data-disabled:opacity-50", className)}
      {...props}
    />
  );
}
