import { Slider as SliderPrimitive } from "@base-ui/react/slider";

import { cn } from "@/lib/utils";

function Slider<Value extends number | readonly number[]>({
  className,
  value,
  defaultValue,
  thumbLabels,
  ...props
}: SliderPrimitive.Root.Props<Value> & { thumbLabels?: string[] }) {
  const values = value ?? defaultValue;
  const thumbCount = typeof values === "number" ? 1 : (values?.length ?? 1);
  return (
    <SliderPrimitive.Root
      data-slot="slider"
      value={value}
      defaultValue={defaultValue}
      className={cn(
        "relative flex w-full touch-none items-center data-[orientation=vertical]:w-auto data-[orientation=vertical]:flex-col",
        className,
      )}
      {...props}
    >
      <SliderPrimitive.Control
        data-slot="slider-control"
        className="flex w-full items-center py-2 data-[orientation=vertical]:h-full data-[orientation=vertical]:w-auto data-[orientation=vertical]:flex-col data-[orientation=vertical]:px-2 data-[orientation=vertical]:py-0"
      >
        <SliderPrimitive.Track
          data-slot="slider-track"
          className="bg-input relative h-1.5 w-full grow overflow-hidden rounded-full data-[orientation=vertical]:h-full data-[orientation=vertical]:w-1.5"
        >
          <SliderPrimitive.Indicator
            data-slot="slider-indicator"
            className="bg-primary absolute h-full data-[orientation=vertical]:w-full"
          />
        </SliderPrimitive.Track>
        {Array.from({ length: thumbCount }, (_, index) => (
          <SliderPrimitive.Thumb
            key={index}
            index={index}
            data-slot="slider-thumb"
            aria-label={thumbLabels?.[index]}
            // the thumb renders a nested input, which takes the focus
            className="border-primary bg-background has-[:focus-visible]:ring-ring/50 block size-4 cursor-pointer shrink-0 rounded-full border shadow-sm transition-[color,box-shadow] outline-none has-[:focus-visible]:ring-[3px] data-disabled:pointer-events-none data-disabled:opacity-50"
          />
        ))}
      </SliderPrimitive.Control>
    </SliderPrimitive.Root>
  );
}

export { Slider };
