import type { ColorPalette } from "@tissuumaps/core";

import { SimpleSelect } from "@/components/common/simple-select";

const colorPaletteSwatchColorCount = 32;

export type ColorPaletteSelectProps = {
  colorPalettes: ColorPalette[];
  value: string | null;
  onValueChange: (newValue: string | null) => void;
  /**
   * Shows the selected palette as a color bar filling a small trigger, named
   * by its tooltip only
   */
  asColorBar?: boolean;
};

export function ColorPaletteSelect({
  colorPalettes,
  value,
  onValueChange,
  asColorBar = false,
}: ColorPaletteSelectProps) {
  const selectedPalette = colorPalettes.find((p) => p.id === value);
  return (
    <SimpleSelect
      items={colorPalettes}
      itemLabel={(p) => <ColorPaletteLabel colorPalette={p} />}
      itemValue={(p) => p.id}
      value={value}
      onValueChange={onValueChange}
      nullable={!asColorBar}
      {...(asColorBar && {
        variant: "compact" as const,
        triggerLabel: `Palette: ${selectedPalette?.name ?? "none"}`,
        // the swatch fills the trigger, which names the palette in its tooltip
        triggerClassName:
          "[&_[data-palette-swatch]]:h-4 [&_[data-palette-swatch]]:w-full [&_[data-palette-swatch]]:shrink [&_[data-palette-name]]:sr-only",
      })}
    />
  );
}

type ColorPaletteLabelProps = {
  colorPalette: ColorPalette;
};

function ColorPaletteLabel({ colorPalette }: ColorPaletteLabelProps) {
  const { colors, name } = colorPalette;
  const count = Math.min(colors.length, colorPaletteSwatchColorCount);
  const swatchColors = Array.from(
    { length: count },
    (_, i) => colors[Math.round((i * (colors.length - 1)) / (count - 1))]!,
  );
  const stops = swatchColors.map(
    ({ r, g, b }, i) =>
      `rgb(${r}, ${g}, ${b}) ${(100 * i) / swatchColors.length}% ${(100 * (i + 1)) / swatchColors.length}%`,
  );
  return (
    <>
      <span
        data-palette-swatch
        className="h-3 w-8 shrink-0 rounded-xs border border-input"
        style={{
          backgroundImage: `linear-gradient(to right, ${stops.join(", ")})`,
        }}
      />
      <span data-palette-name>{name}</span>
    </>
  );
}
