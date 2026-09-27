import {
  ChevronsLeftRightIcon,
  ChevronsRightLeftIcon,
  RotateCcwIcon,
} from "lucide-react";
import { useMemo, useState } from "react";

import { type ImageChannelHistogram, MathUtils } from "@tissuumaps/core";

import { Button } from "@/components/ui/button";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";

export type ContrastRangeWidgetProps = {
  contrastLimits: [number, number];
  onContrastLimitsChange: (contrastLimits: [number, number]) => void;
  onReset?: () => void;
  histogram?: ImageChannelHistogram;
  dataTypeRange?: [number, number];
  className?: string;
};

/** The number of steps the slider divides its range into */
const sliderStepCount = 1000;

/** The number of bars the histogram is drawn with */
const histogramBarCount = 128;

/** The drawn height of the histogram, in SVG units */
const histogramHeight = 40;

/** The narrowest integer range that the slider still steps in whole values */
const minWholeSpan = 10;

export function ContrastRangeWidget({
  contrastLimits,
  onContrastLimitsChange,
  onReset,
  histogram,
  dataTypeRange,
  className,
}: ContrastRangeWidgetProps) {
  const [min, max] = contrastLimits;
  // a range derived from the current limits would shrink with every drag
  const [initialContrastLimits] = useState(contrastLimits);
  const imageRange = histogram?.range ?? dataTypeRange ?? initialContrastLimits;
  const canWiden =
    dataTypeRange !== undefined &&
    (dataTypeRange[0] < imageRange[0] || dataTypeRange[1] > imageRange[1]);
  // a widen request only applies while the data type range is wider than the
  // image's, which can change when the histogram loads
  const [widenRequested, setWidenRequested] = useState(
    () => canWiden && (min < imageRange[0] || max > imageRange[1]),
  );
  const widened = widenRequested && canWiden;
  const baseRange = widened ? dataTypeRange : imageRange;
  const sliderMin = Math.min(min, baseRange[0]);
  const sliderMax = Math.max(max, baseRange[1]);
  const step = getStep(sliderMin, sliderMax);
  const round = (value: number) => MathUtils.roundToStepDecimals(value, step);
  const widenLabel = widened
    ? "Narrow the slider to the values in the image"
    : "Widen the slider to the range of the data type";

  const commitContrastLimits = (limit: number, otherLimit: number) => {
    const clamp = (value: number) =>
      dataTypeRange !== undefined
        ? MathUtils.clamp(value, dataTypeRange[0], dataTypeRange[1])
        : value;
    const newMin = clamp(Math.min(limit, otherLimit));
    const newMax = clamp(Math.max(limit, otherLimit));
    if (canWiden && (newMin < imageRange[0] || newMax > imageRange[1])) {
      setWidenRequested(true);
    }
    onContrastLimitsChange([newMin, newMax]);
  };

  const barHeights = useMemo(
    () =>
      histogram !== undefined
        ? getBarHeights(histogram, [sliderMin, sliderMax])
        : undefined,
    [histogram, sliderMin, sliderMax],
  );
  // the bars between the contrast limits, in fractional bar positions
  const [selectedBarsStart, selectedBarsEnd] =
    sliderMin < sliderMax
      ? [
          MathUtils.remap(min, [sliderMin, sliderMax], [0, histogramBarCount]),
          MathUtils.remap(max, [sliderMin, sliderMax], [0, histogramBarCount]),
        ]
      : [0, 0];

  return (
    <div className={cn("flex flex-col gap-y-1", className)}>
      <div className="grid grid-cols-[1fr_auto] items-center gap-x-1">
        {barHeights !== undefined ? (
          <svg
            className="text-muted-foreground/60 col-start-1 block h-10 w-full"
            viewBox={`0 0 ${histogramBarCount} ${histogramHeight}`}
            preserveAspectRatio="none"
            aria-hidden
          >
            {barHeights.map((height, i) => (
              <rect
                key={i}
                x={i}
                y={histogramHeight - height}
                width={1}
                height={height}
                className={
                  i + 1 > selectedBarsStart && i < selectedBarsEnd
                    ? "fill-primary/70"
                    : "fill-current"
                }
              />
            ))}
          </svg>
        ) : null}
        <Slider
          className="col-start-1"
          value={[min, max]}
          min={sliderMin}
          max={sliderMax > sliderMin ? sliderMax : sliderMin + 1}
          step={step}
          thumbCollisionBehavior="none"
          thumbLabels={["Minimum", "Maximum"]}
          onValueChange={(value) => {
            const [newMin, newMax] = value;
            if (newMin !== undefined && newMax !== undefined) {
              onContrastLimitsChange([round(newMin), round(newMax)]);
            }
          }}
        />
        <Button
          variant="ghost"
          size="icon-xs"
          className="col-start-2"
          aria-label={widenLabel}
          title={widenLabel}
          disabled={!canWiden}
          onClick={() => setWidenRequested(!widened)}
        >
          {widened ? <ChevronsRightLeftIcon /> : <ChevronsLeftRightIcon />}
        </Button>
        <div className="text-muted-foreground col-start-1 flex flex-row justify-between text-[10px] leading-3">
          <span>{round(sliderMin)}</span>
          <span>{round(sliderMax)}</span>
        </div>
      </div>
      <div className="flex flex-row items-center gap-x-2">
        <ContrastLimitInput
          label="min"
          value={round(min)}
          onValueChange={(newMin) => commitContrastLimits(newMin, max)}
        />
        <ContrastLimitInput
          label="max"
          value={round(max)}
          onValueChange={(newMax) => commitContrastLimits(min, newMax)}
        />
        <Button
          variant="ghost"
          size="icon-xs"
          className="ml-auto"
          aria-label="Reset to the limits estimated from the file"
          title="Reset to the limits estimated from the file"
          disabled={onReset === undefined}
          onClick={onReset}
        >
          <RotateCcwIcon />
        </Button>
      </div>
    </div>
  );
}

type ContrastLimitInputProps = {
  label: string;
  value: number;
  onValueChange: (value: number) => void;
  className?: string;
};

// Commits on blur or Enter rather than on every keystroke, as a partly typed
// number would be ordered against the other limit before it is complete
function ContrastLimitInput({
  label,
  value,
  onValueChange,
  className,
}: ContrastLimitInputProps) {
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    if (draft !== null) {
      const newValue = parseFloat(draft);
      if (!isNaN(newValue)) {
        onValueChange(newValue);
      }
      setDraft(null);
    }
  };
  return (
    <InputGroup className={cn("h-7 w-24", className)}>
      <InputGroupAddon className="text-xs">{label}</InputGroupAddon>
      <InputGroupInput
        className="h-full text-xs md:text-xs"
        type="number"
        inputMode="decimal"
        aria-label={label}
        value={draft ?? value}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            commit();
          }
        }}
      />
    </InputGroup>
  );
}

// Integer pixel values stay whole over a wide integer range; a narrow range is
// stepped finely even between whole bounds, as float channels usually span [0, 1]
function getStep(sliderMin: number, sliderMax: number): number {
  const span = sliderMax - sliderMin;
  if (!(span > 0)) {
    return 1;
  }
  const wholeBounds =
    Number.isInteger(sliderMin) && Number.isInteger(sliderMax);
  if (wholeBounds && span >= minWholeSpan) {
    return Math.max(1, Math.round(span / sliderStepCount));
  }
  return span / sliderStepCount;
}

// The bars span the slider's range rather than the histogram's, so that a bar
// and the slider position above it hold the same value; the square-root scale
// keeps sparse tails visible next to a dominant background peak
function getBarHeights(
  histogram: ImageChannelHistogram,
  sliderRange: [number, number],
): number[] {
  const { hist } = MathUtils.rebinHistogram(
    histogram,
    sliderRange,
    histogramBarCount,
  );
  const peak = Math.sqrt(Math.max(...hist, 1));
  return hist.map((count) => (Math.sqrt(count) / peak) * histogramHeight);
}
