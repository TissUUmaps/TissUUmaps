import { Collapsible as CollapsiblePrimitive } from "@base-ui/react/collapsible";
import { ChevronDownIcon, ChevronRightIcon } from "lucide-react";

import { cn } from "@/lib/utils";

export function Collapsible(props: CollapsiblePrimitive.Root.Props) {
  return <CollapsiblePrimitive.Root {...props} />;
}

export function CollapsibleTrigger({
  className,
  children,
  ...props
}: CollapsiblePrimitive.Trigger.Props) {
  return (
    <CollapsiblePrimitive.Trigger
      className={cn(
        "flex flex-row items-center cursor-pointer data-disabled:pointer-events-none data-disabled:opacity-50",
        className,
      )}
      {...props}
    >
      {children}
    </CollapsiblePrimitive.Trigger>
  );
}

export function CollapsibleTriggerRightDownIcon({
  className,
  ...props
}: Omit<CollapsiblePrimitive.Trigger.Props, "children">) {
  return (
    <CollapsiblePrimitive.Trigger
      className={cn(
        "group/collapsible-trigger cursor-pointer data-disabled:pointer-events-none data-disabled:opacity-50 [&_svg]:size-4",
        className,
      )}
      aria-label="Expand/collapse"
      {...props}
    >
      <ChevronRightIcon className="group-aria-expanded/collapsible-trigger:hidden" />
      <ChevronDownIcon className="hidden group-aria-expanded/collapsible-trigger:inline" />
    </CollapsiblePrimitive.Trigger>
  );
}

export function CollapsiblePanel(props: CollapsiblePrimitive.Panel.Props) {
  return <CollapsiblePrimitive.Panel {...props} />;
}
