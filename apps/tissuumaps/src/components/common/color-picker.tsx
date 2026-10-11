import { type HsvaColor, hexToHsva, hsvaToHex } from "@uiw/color-convert";
import Chrome, { ChromeInputType } from "@uiw/react-color-chrome";
import { useState } from "react";

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
  // the picked color, which keeps the hue of grays that the RGB color loses
  const [draft, setDraft] = useState<HsvaColor | null>(null);
  const hex = ColorUtils.toHex(color);
  const hsva =
    draft !== null && hsvaToHex(draft) === hex ? draft : hexToHsva(hex);

  return (
    <Popover>
      <PopoverTrigger
        className={className}
        render={label === undefined ? <Button /> : <IconButton label={label} />}
      >
        {children}
      </PopoverTrigger>
      <PopoverContent className="w-auto overflow-hidden p-0">
        <Chrome
          color={hsva}
          inputType={ChromeInputType.HEXA}
          showAlpha={false}
          showTriangle={false}
          onChange={(newColor) => {
            setDraft(newColor.hsva);
            onColorChange(ColorUtils.fromHex(newColor.hex));
          }}
        />
      </PopoverContent>
    </Popover>
  );
}
