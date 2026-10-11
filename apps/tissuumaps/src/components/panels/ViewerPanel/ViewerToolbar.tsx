import { HouseIcon } from "lucide-react";

import { IconButton } from "@/components/common/icon-button";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

import { InteractionModeMenu } from "./InteractionModeMenu";
import { PointSizePopover } from "./PointSizePopover";
import { ViewMenu } from "./ViewMenu";

/** Hides the drawing modes until drawing is linked with actions */
const interactionModeMenuShown = false;

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
      {interactionModeMenuShown && (
        <>
          <InteractionModeMenu />
          <Separator orientation="vertical" className="mx-0.5" />
        </>
      )}
      <IconButton label="Reset view" onClick={onResetViewport}>
        <HouseIcon />
      </IconButton>
      {hasPoints && <PointSizePopover />}
      <ViewMenu />
    </div>
  );
}
