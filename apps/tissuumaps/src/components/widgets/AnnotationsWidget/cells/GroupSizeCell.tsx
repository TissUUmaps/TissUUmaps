import { Input } from "@/components/ui/input";

export type GroupSizeCellProps = {
  size: number;
  onSizeChange: (size: number) => void;
};

export function GroupSizeCell({ size, onSizeChange }: GroupSizeCellProps) {
  return (
    <Input
      type="number"
      inputMode="decimal"
      step={0.1}
      min={0}
      aria-label="Size"
      className="h-6 text-xs md:text-xs"
      value={size}
      onChange={(event) => {
        const newSize = event.target.valueAsNumber;
        if (!isNaN(newSize)) {
          onSizeChange(Math.max(0, newSize));
        }
      }}
    />
  );
}
