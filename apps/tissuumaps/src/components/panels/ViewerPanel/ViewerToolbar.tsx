import { HouseIcon } from "lucide-react";

import { IconButton } from "@/components/common/icon-button";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

import { InteractionModeMenu } from "./InteractionModeMenu";
import { PointSizePopover } from "./PointSizePopover";
import { ViewMenu } from "./ViewMenu";

/**
 * Whether the toolbar shows the interaction mode menu
 *
 * Hidden until drawing is linked with actions.
 */
const interactionModeMenuShown = false;

export type ViewerToolbarProps = {
  onResetViewport: () => void;
  className?: string;
};

export function ViewerToolbar({
  onResetViewport,
  className,
}: ViewerToolbarProps) {
  return (
    <div
      role="group"
      aria-label="Viewer controls"
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
      <PointSizePopover />
      <ViewMenu />
    </div>
  );
}
