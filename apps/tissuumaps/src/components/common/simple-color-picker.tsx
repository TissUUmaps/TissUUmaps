import { HexColorPicker } from "react-colorful";

import { type Color, ColorUtils } from "@tissuumaps/core";

import { IconButton } from "@/components/common/icon-button";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

export type SimpleColorPickerProps = {
  color: Color;
  onColorChange: (color: Color) => void;
  label?: string;
  children?: React.ReactNode;
  className?: string;
};

export function SimpleColorPicker({
  color,
  onColorChange,
  label,
  children,
  className,
}: SimpleColorPickerProps) {
  return (
    <Popover>
      <PopoverTrigger
        className={className}
        render={label === undefined ? <Button /> : <IconButton label={label} />}
      >
        {children}
      </PopoverTrigger>
      <PopoverContent>
        <HexColorPicker
          color={ColorUtils.toHex(color)}
          onChange={(hex) => onColorChange(ColorUtils.fromHex(hex))}
        />
      </PopoverContent>
    </Popover>
  );
}
