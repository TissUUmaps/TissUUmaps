import { Accordion as AccordionPrimitive } from "@base-ui/react/accordion";
import { ChevronDownIcon, ChevronRightIcon, ChevronUpIcon } from "lucide-react";

import { cn } from "@/lib/utils";

export function Accordion({
  className,
  ...props
}: AccordionPrimitive.Root.Props) {
  return (
    <AccordionPrimitive.Root
      className={cn("flex flex-col", className)}
      {...props}
    />
  );
}

export function AccordionItem(props: AccordionPrimitive.Item.Props) {
  return <AccordionPrimitive.Item {...props} />;
}

export function AccordionHeader({
  className,
  children,
  ...props
}: AccordionPrimitive.Header.Props) {
  return (
    <AccordionPrimitive.Header
      className={cn("flex flex-row items-center text-foreground", className)}
      {...props}
    >
      {children}
    </AccordionPrimitive.Header>
  );
}

export function AccordionTrigger({
  className,
  children,
  ...props
}: AccordionPrimitive.Trigger.Props) {
  return (
    <AccordionPrimitive.Trigger
      className={cn(
        "flex flex-row items-center group/accordion-trigger data-disabled:pointer-events-none data-disabled:opacity-50",
        className,
      )}
      {...props}
    >
      {children}
    </AccordionPrimitive.Trigger>
  );
}

export function AccordionTriggerRightDownIcon({
  className,
  ...props
}: Omit<AccordionPrimitive.Trigger.Props, "children">) {
  return (
    <AccordionPrimitive.Trigger
      className={cn(
        "group/accordion-trigger data-disabled:pointer-events-none data-disabled:opacity-50",
        className,
      )}
      aria-label="Expand/collapse"
      {...props}
    >
      <ChevronRightIcon className="group-aria-expanded/accordion-trigger:hidden" />
      <ChevronDownIcon className="hidden group-aria-expanded/accordion-trigger:inline" />
    </AccordionPrimitive.Trigger>
  );
}

export function AccordionTriggerDownUpIcon({
  className,
  ...props
}: Omit<AccordionPrimitive.Trigger.Props, "children">) {
  return (
    <AccordionPrimitive.Trigger
      className={cn(
        "group/accordion-trigger data-disabled:pointer-events-none data-disabled:opacity-50",
        className,
      )}
      aria-label="Expand/collapse"
      {...props}
    >
      <ChevronDownIcon className="group-aria-expanded/accordion-trigger:hidden" />
      <ChevronUpIcon className="hidden group-aria-expanded/accordion-trigger:inline" />
    </AccordionPrimitive.Trigger>
  );
}

export function AccordionPanel(props: AccordionPrimitive.Panel.Props) {
  return <AccordionPrimitive.Panel {...props} />;
}
