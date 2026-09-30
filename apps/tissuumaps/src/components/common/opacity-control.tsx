import { useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Slider } from "@/components/ui/slider";
import { percentFormat } from "@/lib/format";
import { cn } from "@/lib/utils";

export type OpacityControlProps = {
  opacity: number;

  /** Called on every drag tick; for values that are cheap to update */
  onOpacityChange?: (opacity: number) => void;

  /** Called once when a drag ends; for values that are costly to update */
  onOpacityCommit?: (opacity: number) => void;

  label?: string;

  /** The name of the object, which tells the rows apart for screen readers */
  name?: string;

  className?: string;
};

export function OpacityControl({
  opacity,
  onOpacityChange,
  onOpacityCommit,
  label = "Opacity",
  name,
  className,
}: OpacityControlProps) {
  // the value being dragged, shown before it is committed
  const [draft, setDraft] = useState<number | null>(null);
  const opacityLabel = percentFormat.format(draft ?? opacity);

  return (
    <Popover>
      <PopoverTrigger
        render={
          <Button
            variant="outline"
            size="xs"
            aria-label={
              name === undefined
                ? `${label} ${opacityLabel}`
                : `${label} of ${name} ${opacityLabel}`
            }
            className={cn("w-12 font-normal tabular-nums", className)}
          />
        }
      >
        {opacityLabel}
      </PopoverTrigger>
      <PopoverContent align="end" className="w-56 gap-2 p-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">{label}</span>
          <span className="text-muted-foreground text-xs tabular-nums">
            {opacityLabel}
          </span>
        </div>
        <Slider
          thumbLabels={[label]}
          min={0}
          max={1}
          step={0.01}
          value={draft ?? opacity}
          onValueChange={(value) => {
            setDraft(value);
            onOpacityChange?.(value);
          }}
          onValueCommitted={(value) => {
            setDraft(null);
            onOpacityCommit?.(value);
          }}
        />
      </PopoverContent>
    </Popover>
  );
}
