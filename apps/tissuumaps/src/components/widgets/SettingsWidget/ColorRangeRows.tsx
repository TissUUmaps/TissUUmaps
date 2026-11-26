import { continuousColorPalettes } from "@tissuumaps/core";

import { ColorPaletteSelect } from "@/components/controls/ColorPaletteSelect";
import { RangeEndInput } from "@/components/controls/RangeEndInput";
import type { ColorConfigWidgetAdapter } from "@/components/widgets/ColorConfigWidget/adapter";

import { SettingsRow } from "./SettingsRow";

export type ColorRangeRowsProps = {
  /** The color configuration, whose palette and range the rows edit */
  adapter: ColorConfigWidgetAdapter;
};

/**
 * The palette and value range of a color that follows a numerical column, as
 * rows of the {@link SettingsRow} grid
 */
export function ColorRangeRows({ adapter }: ColorRangeRowsProps) {
  return (
    <>
      <SettingsRow label="Palette">
        <ColorPaletteSelect
          colorPalettes={continuousColorPalettes}
          value={adapter.currentFromPalette}
          onValueChange={adapter.setCurrentFromPalette}
        />
      </SettingsRow>
      <SettingsRow label="Range" className="grid grid-cols-2 gap-x-1">
        <RangeEndInput
          value={adapter.currentFromRangeMin}
          onValueChange={adapter.setCurrentFromRangeMin}
          placeholder={adapter.fromColumnValueRange?.[0]}
          aria-label="Min. value"
          title="Min. value"
          className="h-8 min-w-0"
        />
        <RangeEndInput
          value={adapter.currentFromRangeMax}
          onValueChange={adapter.setCurrentFromRangeMax}
          placeholder={adapter.fromColumnValueRange?.[1]}
          aria-label="Max. value"
          title="Max. value"
          className="h-8 min-w-0"
        />
      </SettingsRow>
    </>
  );
}
