import { Autocomplete as AutocompletePrimitive } from "@base-ui/react/autocomplete";
import { ChevronDownIcon, XIcon } from "lucide-react";

import { IconButton } from "@/components/common/icon-button";
import {
  InputGroup,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
import { cn } from "@/lib/utils";

export function Autocomplete<TItem>(
  props: Omit<AutocompletePrimitive.Root.Props<TItem>, "items"> & {
    items?: readonly TItem[];
  },
) {
  return <AutocompletePrimitive.Root {...props} />;
}

export function AutocompleteInputGroup(
  props: AutocompletePrimitive.InputGroup.Props,
) {
  return (
    <AutocompletePrimitive.InputGroup render={<InputGroup />} {...props} />
  );
}

export function AutocompleteInput(props: AutocompletePrimitive.Input.Props) {
  return (
    <AutocompletePrimitive.Input render={<InputGroupInput />} {...props} />
  );
}

export function AutocompleteClear({
  label = "Clear",
  ...props
}: Omit<AutocompletePrimitive.Clear.Props, "children" | "aria-label"> & {
  label?: string;
}) {
  return (
    <IconButton
      label={label}
      render={
        <AutocompletePrimitive.Clear
          render={<InputGroupButton size="icon-xs" />}
          {...props}
        />
      }
    >
      <XIcon />
    </IconButton>
  );
}

export function AutocompleteTrigger({
  label = "Show suggestions",
  ...props
}: Omit<AutocompletePrimitive.Trigger.Props, "children" | "aria-label"> & {
  label?: string;
}) {
  return (
    <IconButton
      label={label}
      render={
        <AutocompletePrimitive.Trigger
          render={<InputGroupButton size="icon-xs" />}
          {...props}
        />
      }
    >
      <ChevronDownIcon />
    </IconButton>
  );
}

export function AutocompletePopup({
  className,
  ...props
}: AutocompletePrimitive.Popup.Props) {
  return (
    <AutocompletePrimitive.Portal>
      <AutocompletePrimitive.Positioner
        className="isolate z-50 outline-none"
        sideOffset={4}
      >
        <AutocompletePrimitive.Popup
          className={cn(
            "box-border w-(--anchor-width) max-h-[min(var(--available-height),23rem)] max-w-(--available-width) origin-(--transform-origin) overflow-y-auto overscroll-contain rounded-md border bg-popover py-1 text-popover-foreground shadow-md transition-[transform,scale,opacity] data-ending-style:transition-none data-starting-style:scale-95 data-starting-style:opacity-0",
            className,
          )}
          {...props}
        />
      </AutocompletePrimitive.Positioner>
    </AutocompletePrimitive.Portal>
  );
}

export function AutocompleteStatus({
  className,
  ...props
}: AutocompletePrimitive.Status.Props) {
  return (
    <AutocompletePrimitive.Status
      className={cn(
        "px-3 py-1.5 text-xs text-muted-foreground empty:hidden",
        className,
      )}
      {...props}
    />
  );
}

export function AutocompleteList(props: AutocompletePrimitive.List.Props) {
  return <AutocompletePrimitive.List {...props} />;
}

export function AutocompleteItem({
  className,
  ...props
}: AutocompletePrimitive.Item.Props) {
  return (
    <AutocompletePrimitive.Item
      className={cn(
        "flex cursor-default select-none items-center gap-2 px-3 py-1.5 text-sm outline-none data-highlighted:bg-accent data-highlighted:text-accent-foreground",
        className,
      )}
      {...props}
    />
  );
}
