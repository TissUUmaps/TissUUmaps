import { hexToHsva, hsvaToHex } from "@uiw/color-convert";
import Chrome from "@uiw/react-color-chrome";
import { GithubPlacement } from "@uiw/react-color-github";
import React from "react";

import { type Color, ColorUtils } from "@tissuumaps/core";

import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

export type ColorPickerProps = {
  color: Color;
  onColorChange: (color: Color) => void;
  children?: React.ReactNode;
  className?: string;
};

export function ColorPicker({
  color,
  onColorChange,
  children,
  className,
}: ColorPickerProps) {
  const hsva = hexToHsva(ColorUtils.toHex(color));
  return (
    <Popover>
      <PopoverTrigger className={className} render={<Button />}>
        {children}
      </PopoverTrigger>
      <PopoverContent>
        <Chrome
          color={hsva}
          placement={GithubPlacement.BottomRight}
          showAlpha={false}
          onChange={(newColor) => {
            const hex = hsvaToHex(newColor.hsva);
            onColorChange(ColorUtils.fromHex(hex));
          }}
        />
      </PopoverContent>
    </Popover>
  );
}

export default ColorPicker;
