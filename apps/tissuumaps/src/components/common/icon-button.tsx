import type { ComponentProps, ReactElement } from "react";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export type IconButtonProps = Omit<
  ComponentProps<typeof Button>,
  "aria-label" | "render"
> & {
  label: string;

  /**
   * The element to render instead of the button, such as an input group
   * button; it takes its own props, and only gets the label and the children
   */
  render?: ReactElement;
};

// Focusable when disabled, so that its tooltip still shows
export function IconButton({
  label,
  render,
  variant = "ghost",
  size = "icon-sm",
  className,
  children,
  ...props
}: IconButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger
        aria-label={label}
        render={
          render ?? (
            <Button
              variant={variant}
              size={size}
              focusableWhenDisabled
              className={cn(
                "data-disabled:opacity-50",
                variant === "ghost" &&
                  "text-muted-foreground data-disabled:hover:bg-transparent",
                className,
              )}
              {...props}
            />
          )
        }
      >
        {children}
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}
