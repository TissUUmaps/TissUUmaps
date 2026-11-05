import {
  HandIcon,
  type LucideIcon,
  PenToolIcon,
  PencilLineIcon,
  SquareDashedMousePointerIcon,
} from "lucide-react";

import { InteractionMode } from "@tissuumaps/core";

import { IconButton } from "@/components/common/icon-button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAppStore } from "@/stores/app";

/** The menu label of each interaction mode */
const interactionModeLabels: Record<InteractionMode, string> = {
  [InteractionMode.pan]: "Pan",
  [InteractionMode.drawRectangle]: "Draw rectangle",
  [InteractionMode.drawPolygon]: "Draw polygon",
  [InteractionMode.drawFreehand]: "Draw freehand",
};

/** The icon of each interaction mode */
const interactionModeIcons: Record<InteractionMode, LucideIcon> = {
  [InteractionMode.pan]: HandIcon,
  [InteractionMode.drawRectangle]: SquareDashedMousePointerIcon,
  [InteractionMode.drawPolygon]: PenToolIcon,
  [InteractionMode.drawFreehand]: PencilLineIcon,
};

export type InteractionModeMenuProps = { className?: string };

export function InteractionModeMenu({ className }: InteractionModeMenuProps) {
  const interactionMode = useAppStore((state) => state.interactionMode);
  const setInteractionMode = useAppStore((state) => state.setInteractionMode);

  const ActiveIcon = interactionModeIcons[interactionMode];

  return (
    // not modal: the viewer stays usable while the menu is open
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger
        render={
          <IconButton
            label={`Interaction mode: ${interactionModeLabels[interactionMode]}`}
            className={className}
          />
        }
      >
        <ActiveIcon />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        <DropdownMenuRadioGroup
          value={interactionMode}
          onValueChange={(value) =>
            setInteractionMode(value as InteractionMode)
          }
        >
          {Object.values(InteractionMode).map((mode) => {
            const Icon = interactionModeIcons[mode];
            return (
              <DropdownMenuRadioItem key={mode} value={mode}>
                <Icon />
                {interactionModeLabels[mode]}
              </DropdownMenuRadioItem>
            );
          })}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
