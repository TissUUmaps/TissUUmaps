import { hexToHsva, hsvaToHex } from "@uiw/color-convert";
import Chrome from "@uiw/react-color-chrome";

import { type Color, ColorUtils } from "@tissuumaps/core";

import { IconButton } from "@/components/common/icon-button";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

export type ColorPickerProps = {
  color: Color;
  onColorChange: (color: Color) => void;
  label?: string;
  children?: React.ReactNode;
  className?: string;
};

export function ColorPicker({
  color,
  onColorChange,
  label,
  children,
  className,
}: ColorPickerProps) {
  const hsva = hexToHsva(ColorUtils.toHex(color));
  return (
    <Popover>
      <PopoverTrigger
        className={className}
        render={label === undefined ? <Button /> : <IconButton label={label} />}
      >
        {children}
      </PopoverTrigger>
      <PopoverContent>
        <Chrome
          color={hsva}
          showAlpha={false}
          showTriangle={false}
          onChange={(newColor) => {
            const hex = hsvaToHex(newColor.hsva);
            onColorChange(ColorUtils.fromHex(hex));
          }}
        />
      </PopoverContent>
    </Popover>
  );
}
