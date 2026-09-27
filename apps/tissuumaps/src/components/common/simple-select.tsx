import { Select as SelectPrimitive } from "@base-ui/react/select";
import { CheckIcon, ChevronsUpDownIcon, Trash2Icon } from "lucide-react";
import { type ReactNode, useMemo } from "react";

import { Button } from "@/components/ui/button";

export type SimpleSelectProps<
  TItem,
  TNullable extends boolean | undefined,
  TValue extends string | number | (TNullable extends true ? null : never),
  TMultiple extends boolean | undefined = false,
> = {
  items: TItem[];
  itemLabel: (item: TItem) => ReactNode;
  itemValue: (item: TItem) => TValue;
  nullable?: TNullable;
  placeholder?: string;
  onItemDelete?: (item: TItem) => void;
  isItemDeletable?: (item: TItem) => boolean;
} & Omit<
  SelectPrimitive.Root.Props<TValue, TMultiple>,
  "items" | "itemToStringLabel" | "itemToStringValue"
>;

export function SimpleSelect<
  TItem,
  TNullable extends boolean | undefined,
  TValue extends string | number | (TNullable extends true ? null : never),
  TMultiple extends boolean | undefined = false,
>({
  items,
  itemLabel,
  itemValue,
  nullable,
  placeholder,
  onItemDelete,
  isItemDeletable,
  ...props
}: SimpleSelectProps<TItem, TNullable, TValue, TMultiple>) {
  const memoizedItems = useMemo(() => {
    const result: { label: ReactNode; value: TValue; item?: TItem }[] =
      items.map((item) => ({
        label: itemLabel(item),
        value: itemValue(item),
        item,
      }));
    if (nullable) {
      result.unshift({ label: placeholder ?? "None", value: null as TValue });
    }
    return result;
  }, [items, itemLabel, itemValue, nullable, placeholder]);

  // style adapted from https://base-ui.com/react/components/select main example, with the following changes:
  // - set trigger width to w-full
  return (
    <SelectPrimitive.Root items={memoizedItems} {...props}>
      <SelectPrimitive.Trigger className="flex w-full h-10 min-w-40 items-center justify-between gap-3 rounded-md border border-input pr-3 pl-3.5 bg-[canvas] select-none hover:bg-gray-100 focus-visible:outline-2 focus-visible:-outline-offset-1 focus-visible:outline-blue-800 data-popup-open:bg-gray-100 data-disabled:pointer-events-none data-disabled:opacity-50 dark:bg-input/30 text-foreground">
        <SelectPrimitive.Value className="data-placeholder:opacity-60 flex flex-row items-center gap-2" />
        <SelectPrimitive.Icon className="flex">
          <ChevronsUpDownIcon />
        </SelectPrimitive.Icon>
      </SelectPrimitive.Trigger>
      <SelectPrimitive.Portal>
        <SelectPrimitive.Positioner
          className="outline-none select-none isolate z-50"
          sideOffset={8}
        >
          <SelectPrimitive.Popup className="group min-w-(--anchor-width) origin-(--transform-origin) bg-clip-padding rounded-md bg-[canvas] text-gray-900 shadow-lg shadow-gray-200 outline-1 outline-gray-200 transition-[transform,scale,opacity] data-ending-style:scale-90 data-ending-style:opacity-0 data-[side=none]:min-w-[calc(var(--anchor-width)+1rem)] data-[side=none]:data-ending-style:transition-none data-starting-style:scale-90 data-starting-style:opacity-0 data-[side=none]:data-starting-style:scale-100 data-[side=none]:data-starting-style:opacity-100 data-[side=none]:data-starting-style:transition-none dark:shadow-none dark:outline-gray-300 dark:bg-gray-900 dark:text-gray-100">
            <SelectPrimitive.ScrollUpArrow className="top-0 z-1 flex h-4 w-full cursor-default items-center justify-center rounded-md bg-[canvas] text-center text-xs before:absolute data-[side=none]:before:-top-full before:left-0 before:h-full before:w-full before:content-[''] dark:bg-gray-900 dark:text-gray-100" />
            <SelectPrimitive.List className="relative py-1 scroll-py-6 overflow-y-auto max-h-(--available-height) dark:bg-gray-900">
              {memoizedItems.map(({ label, value, item }) => (
                <SelectPrimitive.Item
                  key={String(value)}
                  value={value}
                  className="grid cursor-default grid-cols-[0.75rem_1fr_auto] items-center gap-2 py-2 pr-4 pl-2.5 text-sm leading-4 outline-none select-none group-data-[side=none]:pr-12 group-data-[side=none]:text-base group-data-[side=none]:leading-4 data-highlighted:relative data-highlighted:z-0 data-highlighted:text-gray-50 data-highlighted:before:absolute data-highlighted:before:inset-x-1 data-highlighted:before:inset-y-0 data-highlighted:before:z-[-1] data-highlighted:before:rounded-sm data-highlighted:before:bg-gray-900 pointer-coarse:py-2.5 pointer-coarse:text-[0.925rem]"
                >
                  <SelectPrimitive.ItemIndicator className="col-start-1">
                    <CheckIcon className="size-3" />
                  </SelectPrimitive.ItemIndicator>
                  <SelectPrimitive.ItemText className="col-start-2 flex flex-row items-center gap-2">
                    {label}
                  </SelectPrimitive.ItemText>
                  {onItemDelete !== undefined && item !== undefined && (
                    <ItemDeleteButton
                      isDeletable={isItemDeletable?.(item) ?? true}
                      onDelete={() => onItemDelete(item)}
                    />
                  )}
                </SelectPrimitive.Item>
              ))}
            </SelectPrimitive.List>
            <SelectPrimitive.ScrollDownArrow className="bottom-0 z-1 flex h-4 w-full cursor-default items-center justify-center rounded-md bg-[canvas] text-center text-xs before:absolute before:left-0 before:h-full before:w-full before:content-[''] data-[side=none]:before:-bottom-full dark:bg-gray-900 dark:text-gray-100" />
          </SelectPrimitive.Popup>
        </SelectPrimitive.Positioner>
      </SelectPrimitive.Portal>
    </SelectPrimitive.Root>
  );
}

type ItemDeleteButtonProps = {
  isDeletable: boolean;
  onDelete: () => void;
};

function ItemDeleteButton({ isDeletable, onDelete }: ItemDeleteButtonProps) {
  // the item selects itself on a click, or on a mouse up after dragging from
  // the trigger; a disabled button would pass neither event on to be stopped
  return (
    <Button
      variant="ghost"
      size="icon-xs"
      className="col-start-3 size-5 aria-disabled:opacity-30 aria-disabled:hover:bg-transparent"
      title={isDeletable ? "Delete" : "In use"}
      aria-label={isDeletable ? "Delete" : "In use"}
      aria-disabled={!isDeletable}
      tabIndex={-1}
      onPointerDown={(event) => event.stopPropagation()}
      onMouseUp={(event) => event.stopPropagation()}
      onClick={(event) => {
        event.stopPropagation();
        if (isDeletable) {
          onDelete();
        }
      }}
    >
      <Trash2Icon />
    </Button>
  );
}
