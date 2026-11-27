import { categoricalColorPalettes } from "@tissuumaps/core";

import { ColorPaletteSelect } from "@/components/controls/ColorPaletteSelect";
import type { ColorConfigWidgetAdapter } from "@/components/widgets/ColorConfigWidget/adapter";

import { SettingsRow } from "./SettingsRow";

export type CategoricalColorPaletteRowProps = {
  /** The color configuration, whose palette the row edits */
  adapter: ColorConfigWidgetAdapter;
};

/**
 * The palette of a color that follows a categorical column, as a row of the
 * {@link SettingsRow} grid
 *
 * Disabled while the color has a color map, which replaces the palette (e.g.
 * once a color is edited in the group table).
 */
export function CategoricalColorPaletteRow({
  adapter,
}: CategoricalColorPaletteRowProps) {
  return (
    <SettingsRow label="Palette" disabled={adapter.currentGroupByMap !== null}>
      <ColorPaletteSelect
        colorPalettes={categoricalColorPalettes}
        value={adapter.currentGroupByPalette}
        onValueChange={adapter.setCurrentGroupByPalette}
        // the box saves only a complete color, so a palette is required
        nullable={false}
      />
    </SettingsRow>
  );
}
