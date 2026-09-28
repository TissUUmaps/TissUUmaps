import type { Marker } from "@tissuumaps/core";

import { IconButton } from "@/components/common/icon-button";
import { markers } from "@/components/markers";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

export type SimpleMarkerPickerProps = {
  marker: Marker;
  onMarkerChange: (marker: Marker) => void;
  children?: React.ReactNode;
  className?: string;
};

export function SimpleMarkerPicker({
  marker,
  onMarkerChange,
  children,
  className,
}: SimpleMarkerPickerProps) {
  return (
    <Popover>
      <PopoverTrigger
        className={className}
        render={<IconButton label="Marker" />}
      >
        {children}
      </PopoverTrigger>
      <PopoverContent className="grid grid-cols-4 gap-1">
        {markers.map((m) => (
          <IconButton
            key={m.value}
            label={m.label}
            variant={m.value === marker ? "secondary" : "ghost"}
            onClick={() => onMarkerChange(m.value)}
          >
            {m.icon}
          </IconButton>
        ))}
      </PopoverContent>
    </Popover>
  );
}
