import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

import type { ColorConfigWidgetAdapter } from "./adapter";

export type ColorConfigSourceToggleGroupProps = Omit<
  Parameters<typeof ToggleGroup>[0],
  "value" | "onValueChange"
> & {
  adapter: ColorConfigWidgetAdapter;
};

export function ColorConfigSourceToggleGroup({
  adapter,
  ...props
}: ColorConfigSourceToggleGroupProps) {
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
        value={"constant" satisfies typeof currentSource}
        className={activeSource === "constant" ? "font-medium" : "font-normal"}
      >
        constant
      </ToggleGroupItem>
      <ToggleGroupItem
        value={"from" satisfies typeof currentSource}
        className={activeSource === "from" ? "font-medium" : "font-normal"}
        disabled={tableId === null}
      >
        from
      </ToggleGroupItem>
      <ToggleGroupItem
        value={"groupBy" satisfies typeof currentSource}
        className={activeSource === "groupBy" ? "font-medium" : "font-normal"}
        disabled={tableId === null}
      >
        group by
      </ToggleGroupItem>
      <ToggleGroupItem
        value={"random" satisfies typeof currentSource}
        className={activeSource === "random" ? "font-medium" : "font-normal"}
      >
        random
      </ToggleGroupItem>
    </ToggleGroup>
  );
}
