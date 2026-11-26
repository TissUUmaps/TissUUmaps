import type { ComponentProps } from "react";

import { Input } from "@/components/ui/input";

export type RangeEndInputProps = Omit<
  ComponentProps<typeof Input>,
  "type" | "value" | "onChange" | "placeholder"
> & {
  value: number | null;
  onValueChange: (value: number | null) => void;
  /** The column's minimum or maximum value, shown while the end is unset */
  placeholder: number | undefined;
};

/** One end of a value range; an empty end is the column's own */
export function RangeEndInput({
  value,
  onValueChange,
  placeholder,
  ...props
}: RangeEndInputProps) {
  return (
    <Input
      type="number"
      inputMode="decimal"
      placeholder={placeholder?.toPrecision(6)}
      value={value ?? ""}
      onChange={(event) => {
        if (event.target.value === "") {
          onValueChange(null);
        } else {
          const newValue = event.target.valueAsNumber;
          if (!isNaN(newValue)) {
            onValueChange(newValue);
          }
        }
      }}
      {...props}
    />
  );
}
