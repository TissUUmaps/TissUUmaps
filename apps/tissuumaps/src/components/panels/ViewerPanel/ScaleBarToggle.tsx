import { IconToggle } from "@/components/common/icon-toggle";
import { useProjectStore } from "@/stores/project";

function ScaleBarIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <g transform="rotate(-45 12 12)">
        <rect x="1.5" y="8" width="21" height="8" rx="1.5" />
        <path d="M7 8v2.5M12 8v4M17 8v2.5" />
      </g>
    </svg>
  );
}

export type ScaleBarToggleProps = { className?: string };

export function ScaleBarToggle({ className }: ScaleBarToggleProps) {
  const worldUnitSize = useProjectStore((state) => state.worldUnitSize);
  const scaleBarVisible = useProjectStore((state) => state.scaleBarVisible);
  const setScaleBarVisible = useProjectStore(
    (state) => state.setScaleBarVisible,
  );

  const scaleBarAvailable = worldUnitSize !== undefined && worldUnitSize > 0;

  return (
    <IconToggle
      label={
        scaleBarAvailable
          ? "Scale bar"
          : "Scale bar (set the world unit size in the render settings)"
      }
      disabled={!scaleBarAvailable}
      className={className}
      pressed={scaleBarVisible}
      onPressedChange={setScaleBarVisible}
    >
      <ScaleBarIcon />
    </IconToggle>
  );
}
