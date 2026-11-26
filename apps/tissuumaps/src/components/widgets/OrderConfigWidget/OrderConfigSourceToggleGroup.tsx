import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

import type { OrderConfigWidgetAdapter } from "./adapter";

export type OrderConfigSourceToggleGroupProps = Omit<
  Parameters<typeof ToggleGroup>[0],
  "value" | "onValueChange"
> & {
  adapter: OrderConfigWidgetAdapter;
};

export function OrderConfigSourceToggleGroup({
  adapter,
  ...props
}: OrderConfigSourceToggleGroupProps) {
  const { tableId, activeSource, currentSource, setCurrentSource } = adapter;

  return (
    <ToggleGroup
      value={[currentSource]}
      onValueChange={(value) => {
        if (value.length === 1 && value[0] !== undefined) {
          setCurrentSource(value[0] as typeof currentSource);
        }
      }}
      {...props}
    >
      <ToggleGroupItem
        value={"none" satisfies typeof currentSource}
        className={activeSource === "none" ? "font-medium" : "font-normal"}
      >
        none
      </ToggleGroupItem>
      <ToggleGroupItem
        value={"from" satisfies typeof currentSource}
        className={activeSource === "from" ? "font-medium" : "font-normal"}
        disabled={tableId === null}
      >
        from
      </ToggleGroupItem>
    </ToggleGroup>
  );
}
