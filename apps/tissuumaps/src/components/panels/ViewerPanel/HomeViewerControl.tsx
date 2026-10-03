import { HouseIcon } from "lucide-react";

import { useResetViewport } from "@tissuumaps/react";

import { IconButton } from "@/components/common/icon-button";
import { cn } from "@/lib/utils";

export type HomeViewerControlProps = { className?: string };

export function HomeViewerControl({ className }: HomeViewerControlProps) {
  const resetViewport = useResetViewport();

  return (
    <IconButton
      label="Reset view"
      size="icon"
      className={cn(
        "m-2 rounded-xl border-border bg-background shadow-lg",
        className,
      )}
      onClick={resetViewport}
    >
      <HouseIcon />
    </IconButton>
  );
}
