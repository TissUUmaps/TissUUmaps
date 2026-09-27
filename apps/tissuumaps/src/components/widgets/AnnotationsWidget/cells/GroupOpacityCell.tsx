import { MathUtils } from "@tissuumaps/core";

import { Input } from "@/components/ui/input";

export type GroupOpacityCellProps = {
  opacity: number;
  onOpacityChange: (opacity: number) => void;
};

export function GroupOpacityCell({
  opacity,
  onOpacityChange,
}: GroupOpacityCellProps) {
  return (
    <Input
      type="number"
      inputMode="decimal"
      step={0.05}
      min={0}
      max={1}
      aria-label="Opacity"
      className="h-6 text-xs md:text-xs"
      value={opacity}
      onChange={(event) => {
        const newOpacity = event.target.valueAsNumber;
        if (!isNaN(newOpacity)) {
          onOpacityChange(MathUtils.clamp(newOpacity, 0, 1));
        }
      }}
    />
  );
}
