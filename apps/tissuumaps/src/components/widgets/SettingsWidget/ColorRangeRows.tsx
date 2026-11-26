import { continuousColorPalettes } from "@tissuumaps/core";

import { Input } from "@/components/ui/input";
import { ColorPaletteSelect } from "@/components/widgets/ColorConfigWidget/ColorPaletteSelect";
import type { ColorConfigWidgetAdapter } from "@/components/widgets/ColorConfigWidget/adapter";

import { StyleRow } from "./StyleRow";

export type ColorRangeRowsProps = {
  /** The color configuration, whose palette and range the rows edit */
  adapter: ColorConfigWidgetAdapter;
};

/**
 * The palette and value range of a color that follows a numerical column, as
 * rows of the {@link StyleRow} grid
 */
export function ColorRangeRows({ adapter }: ColorRangeRowsProps) {
  return (
    <>
      <StyleRow label="Palette">
        <ColorPaletteSelect
          colorPalettes={continuousColorPalettes}
          value={adapter.currentFromPalette}
          onValueChange={adapter.setCurrentFromPalette}
        />
      </StyleRow>
      <StyleRow label="Range" className="grid grid-cols-2 gap-x-1">
        <RangeEndInput
          label="Min. value"
          value={adapter.currentFromRangeMin}
          onValueChange={adapter.setCurrentFromRangeMin}
          placeholder={adapter.fromColumnValueRange?.[0]}
        />
        <RangeEndInput
          label="Max. value"
          value={adapter.currentFromRangeMax}
          onValueChange={adapter.setCurrentFromRangeMax}
          placeholder={adapter.fromColumnValueRange?.[1]}
        />
      </StyleRow>
    </>
  );
}

type RangeEndInputProps = {
  label: string;
  value: number | null;
  onValueChange: (value: number | null) => void;
  /** The column's minimum or maximum value, shown while the end is unset */
  placeholder: number | undefined;
};

/** One end of the value range; an empty end is the column's own */
function RangeEndInput({
  label,
  value,
  onValueChange,
  placeholder,
}: RangeEndInputProps) {
  return (
    <Input
      type="number"
      inputMode="decimal"
      aria-label={label}
      title={label}
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
      className="h-8 min-w-0"
    />
  );
}
