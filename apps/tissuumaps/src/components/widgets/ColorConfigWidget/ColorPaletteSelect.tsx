import type { ColorPalette } from "@tissuumaps/core";

import { SimpleSelect } from "@/components/common/simple-select";

const colorPaletteSwatchColorCount = 32;

export type ColorPaletteSelectProps = {
  colorPalettes: ColorPalette[];
  value: string | null;
  onValueChange: (newValue: string | null) => void;
};

export function ColorPaletteSelect({
  colorPalettes,
  value,
  onValueChange,
}: ColorPaletteSelectProps) {
  return (
    <SimpleSelect
      items={colorPalettes}
      itemLabel={(p) => <ColorPaletteLabel colorPalette={p} />}
      itemValue={(p) => p.id}
      value={value}
      onValueChange={onValueChange}
      nullable
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
        className="h-3 w-8 shrink-0 rounded-xs border border-input"
        style={{
          backgroundImage: `linear-gradient(to right, ${stops.join(", ")})`,
        }}
      />
      {name}
    </>
  );
}
