import type { Shapes } from "@tissuumaps/core";

import { IconToggle } from "@/components/common/icon-toggle";
import { useProjectStore } from "@/stores/project";

type PartIconProps = { shown: boolean };

function FillIcon({ shown }: PartIconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden>
      <rect
        x="4"
        y="4"
        width="16"
        height="16"
        rx="4"
        className="fill-current"
        fillOpacity={shown ? 0.6 : 0.12}
      />
    </svg>
  );
}

function StrokeIcon({ shown }: PartIconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden>
      <rect
        x="4.5"
        y="4.5"
        width="15"
        height="15"
        rx="3.5"
        fill="none"
        stroke="currentColor"
        strokeWidth={shown ? 2.5 : 1.25}
        strokeDasharray={shown ? undefined : "2.5 2.5"}
      />
    </svg>
  );
}

export type FillStrokeTogglesProps = { shapes: Shapes };

/** Toggles that show or hide the fill and the stroke of a shape cloud */
export function FillStrokeToggles({ shapes }: FillStrokeTogglesProps) {
  const updateShapes = useProjectStore((state) => state.updateShapes);
  return (
    <>
      <IconToggle
        label={`Fill of ${shapes.name}`}
        pressed={shapes.fillVisibility}
        onPressedChange={(fillVisibility) =>
          updateShapes(shapes.id, { fillVisibility })
        }
      >
        <FillIcon shown={shapes.fillVisibility} />
      </IconToggle>
      <IconToggle
        label={`Stroke of ${shapes.name}`}
        pressed={shapes.strokeVisibility}
        onPressedChange={(strokeVisibility) =>
          updateShapes(shapes.id, { strokeVisibility })
        }
      >
        <StrokeIcon shown={shapes.strokeVisibility} />
      </IconToggle>
    </>
  );
}
