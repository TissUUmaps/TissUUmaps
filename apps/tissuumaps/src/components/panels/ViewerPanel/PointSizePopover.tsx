import { ChartScatterIcon } from "lucide-react";

import { MathUtils } from "@tissuumaps/core";

import { IconButton } from "@/components/common/icon-button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Slider } from "@/components/ui/slider";
import { percentFormat } from "@/lib/format";
import { useProjectStore } from "@/stores/project";

const minPointSizeFactor = 0.01;
const maxPointSizeFactor = 5;
const zeroZoneWidth = 0.05; // positions in [-1, -1 + zeroZoneWidth) map to 0
const snapZoneWidth = 0.1; // positions in [-snapZoneWidth, snapZoneWidth] map to 1

// Log-linear segments on either side of the snap zone, as [min, max] ranges
const lowerPositions: [number, number] = [-1 + zeroZoneWidth, -snapZoneWidth];
const lowerDecades: [number, number] = [Math.log10(minPointSizeFactor), 0];
const upperPositions: [number, number] = [snapZoneWidth, 1];
const upperDecades: [number, number] = [0, Math.log10(maxPointSizeFactor)];

// Maps a slider position in [-1, 1] to a point size factor: the bottom is 0,
// the centre snaps to 1, and each side scales logarithmically to its end.
function sliderPositionToPointSizeFactor(position: number): number {
  if (position < lowerPositions[0]) {
    return 0;
  }
  if (Math.abs(position) <= snapZoneWidth) {
    return 1;
  }
  const decades =
    position < 0
      ? MathUtils.remap(position, lowerPositions, lowerDecades)
      : MathUtils.remap(position, upperPositions, upperDecades);
  return Math.round(1000 * 10 ** decades) / 1000;
}

// Inverse of sliderPositionToPointSizeFactor, clamped to the slider range.
function pointSizeFactorToSliderPosition(pointSizeFactor: number): number {
  if (pointSizeFactor <= 0) {
    return -1;
  }
  const decades = Math.log10(
    MathUtils.clamp(pointSizeFactor, minPointSizeFactor, maxPointSizeFactor),
  );
  if (decades === 0) {
    return 0;
  }
  return decades < 0
    ? MathUtils.remap(decades, lowerDecades, lowerPositions)
    : MathUtils.remap(decades, upperDecades, upperPositions);
}

export type PointSizePopoverProps = { className?: string };

export function PointSizePopover({ className }: PointSizePopoverProps) {
  const hasPoints = useProjectStore((state) => state.points.length > 0);
  const glOptions = useProjectStore((state) => state.glOptions);
  const setGLOptions = useProjectStore((state) => state.setGLOptions);

  const { globalPointSizeFactor } = glOptions.pointsRenderOptions;

  return (
    <Popover>
      <PopoverTrigger
        disabled={!hasPoints}
        render={
          <IconButton
            label={hasPoints ? "Point size" : "Point size (no points)"}
            className={className}
          />
        }
      >
        <ChartScatterIcon />
      </PopoverTrigger>
      <PopoverContent align="end" className="w-auto items-center gap-1 p-2">
        <Slider
          orientation="vertical"
          className="my-2 h-32"
          thumbLabels={["Global point size"]}
          min={-1}
          max={1}
          step={0.01}
          value={pointSizeFactorToSliderPosition(globalPointSizeFactor)}
          onValueChange={(value) => {
            setGLOptions({
              ...glOptions,
              pointsRenderOptions: {
                ...glOptions.pointsRenderOptions,
                globalPointSizeFactor: sliderPositionToPointSizeFactor(value),
              },
            });
          }}
        />
        <span className="text-muted-foreground w-9 text-center text-xs tabular-nums">
          {percentFormat.format(globalPointSizeFactor)}
        </span>
      </PopoverContent>
    </Popover>
  );
}
