import type { Marker } from "@tissuumaps/core";

import { markers } from "@/components/markers";
import { Button } from "@/components/ui/button";
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
        render={<Button variant="ghost" size="icon-sm" />}
      >
        {children}
      </PopoverTrigger>
      <PopoverContent className="grid grid-cols-4 gap-1">
        {markers.map((m) => (
          <Button
            key={m.value}
            variant={m.value === marker ? "secondary" : "ghost"}
            size="icon-sm"
            title={m.label}
            aria-label={m.label}
            onClick={() => onMarkerChange(m.value)}
          >
            {m.icon}
          </Button>
        ))}
      </PopoverContent>
    </Popover>
  );
}
