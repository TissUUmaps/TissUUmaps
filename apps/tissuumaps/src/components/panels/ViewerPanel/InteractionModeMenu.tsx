import {
  HandIcon,
  type LucideIcon,
  PenToolIcon,
  PencilLineIcon,
  SquareDashedMousePointerIcon,
} from "lucide-react";

import type { InteractionMode } from "@tissuumaps/core";

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
  pan: "Pan",
  drawRectangle: "Draw rectangle",
  drawPolygon: "Draw polygon",
  drawFreehand: "Draw freehand",
};

/** The icon of each interaction mode */
const interactionModeIcons: Record<InteractionMode, LucideIcon> = {
  pan: HandIcon,
  drawRectangle: SquareDashedMousePointerIcon,
  drawPolygon: PenToolIcon,
  drawFreehand: PencilLineIcon,
};

/** The interaction modes in menu order */
const interactionModes: InteractionMode[] = [
  "pan",
  "drawRectangle",
  "drawPolygon",
  "drawFreehand",
];

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
          {interactionModes.map((mode) => {
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
