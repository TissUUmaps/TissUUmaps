import {
  IconButton,
  type IconButtonProps,
} from "@/components/common/icon-button";
import { cn } from "@/lib/utils";

export type IconToggleProps = Omit<IconButtonProps, "onClick"> & {
  pressed: boolean;
  onPressedChange: (pressed: boolean) => void;
};

// The label names what the toggle switches, as aria-pressed tells its state.
// The icon shows the state too: a pressed background would look like hover.
export function IconToggle({
  pressed,
  onPressedChange,
  className,
  ...props
}: IconToggleProps) {
  return (
    <IconButton
      aria-pressed={pressed}
      className={cn("aria-pressed:text-foreground", className)}
      onClick={() => onPressedChange(!pressed)}
      {...props}
    />
  );
}
