import type { CoordinateSpace } from "@tissuumaps/core";

import { FieldItem, FieldLabel } from "@/components/common/field";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { cn } from "@/lib/utils";

export type SizeUnitRadioGroupProps = {
  value: CoordinateSpace;
  onValueChange: (value: CoordinateSpace) => void;
  className?: string;
};

/** Picks the coordinate space in which sizes are specified */
export function SizeUnitRadioGroup({
  value,
  onValueChange,
  className,
}: SizeUnitRadioGroupProps) {
  return (
    <RadioGroup
      value={value}
      onValueChange={(newValue) => onValueChange(newValue as CoordinateSpace)}
      className={cn("flex gap-x-4", className)}
    >
      <FieldItem className="flex items-center gap-x-2">
        <RadioGroupItem value={"data" satisfies CoordinateSpace} />
        <FieldLabel>Data</FieldLabel>
      </FieldItem>
      <FieldItem className="flex items-center gap-x-2">
        <RadioGroupItem value={"layer" satisfies CoordinateSpace} />
        <FieldLabel>Layer</FieldLabel>
      </FieldItem>
      <FieldItem className="flex items-center gap-x-2">
        <RadioGroupItem value={"world" satisfies CoordinateSpace} />
        <FieldLabel>World</FieldLabel>
      </FieldItem>
    </RadioGroup>
  );
}
