import { HouseIcon } from "lucide-react";

import { IconButton } from "@/components/common/icon-button";
import { cn } from "@/lib/utils";

export type HomeViewerControlProps = {
  onClick: () => void;
  className?: string;
};

export function HomeViewerControl({
  onClick,
  className,
}: HomeViewerControlProps) {
  return (
    <IconButton
      label="Reset view"
      size="icon"
      // bg-clip-border: the button's default bg-clip-padding leaves a gray
      // seam between the border and the background along the rounded corners
      className={cn(
        "m-2 rounded-xl border-border bg-background bg-clip-border shadow-lg",
        className,
      )}
      onClick={onClick}
    >
      <HouseIcon />
    </IconButton>
  );
}
