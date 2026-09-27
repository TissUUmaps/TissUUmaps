import type { Color } from "@tissuumaps/core";

import { SimpleColorPicker } from "@/components/common/simple-color-picker";

export type GroupColorCellProps = {
  color: Color;
  onColorChange: (color: Color) => void;
};

export function GroupColorCell({ color, onColorChange }: GroupColorCellProps) {
  return (
    <SimpleColorPicker
      color={color}
      onColorChange={onColorChange}
      className="h-6 w-10 bg-transparent p-0.5 hover:bg-muted"
    >
      <span
        className="h-4 w-8 rounded-xs border"
        style={{ backgroundColor: `rgb(${color.r}, ${color.g}, ${color.b})` }}
      />
    </SimpleColorPicker>
  );
}
