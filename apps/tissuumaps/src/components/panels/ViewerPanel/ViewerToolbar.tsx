import { HouseIcon } from "lucide-react";

import { IconButton } from "@/components/common/icon-button";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

import { PointSizePopover } from "./PointSizePopover";

export type ViewerToolbarProps = {
  onResetViewport: () => void;
  className?: string;
};

export function ViewerToolbar({
  onResetViewport,
  className,
}: ViewerToolbarProps) {
  const hasPoints = useProjectStore((state) => state.points.length > 0);

  return (
    <div
      className={cn(
        "m-2 flex items-center gap-1 rounded-xl border border-border bg-background p-1 shadow-lg",
        className,
      )}
    >
      <IconButton label="Reset view" onClick={onResetViewport}>
        <HouseIcon />
      </IconButton>
      {hasPoints && <PointSizePopover />}
    </div>
  );
}
