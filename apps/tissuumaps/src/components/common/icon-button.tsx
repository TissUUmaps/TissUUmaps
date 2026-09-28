import { mergeProps } from "@base-ui/react/merge-props";
import { type ComponentProps, type ReactElement, cloneElement } from "react";

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
   * button; the other props are merged into it, but not the variant and size
   */
  render?: ReactElement<ComponentProps<"button">>;
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
          render ? (
            cloneElement(
              render,
              mergeProps(render.props, { className, ...props }),
            )
          ) : (
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
