import { IconToggle } from "@/components/common/icon-toggle";
import { useProjectStore } from "@/stores/project";

type ScaleBarIconProps = { filled: boolean };

function ScaleBarIcon({ filled }: ScaleBarIconProps) {
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
        <rect
          x="1.5"
          y="8"
          width="21"
          height="8"
          rx="1.5"
          className={filled ? "fill-amber-300" : undefined}
        />
        <path d="M7 8v2.5M12 8v4M17 8v2.5" />
      </g>
    </svg>
  );
}

export type ScaleBarToggleProps = { className?: string };

export function ScaleBarToggle({ className }: ScaleBarToggleProps) {
  const scaleBarVisibility = useProjectStore(
    (state) => state.scaleBarVisibility,
  );
  const setScaleBarVisibility = useProjectStore(
    (state) => state.setScaleBarVisibility,
  );

  return (
    <IconToggle
      label="Scale bar"
      className={className}
      pressed={scaleBarVisibility}
      onPressedChange={setScaleBarVisibility}
    >
      <ScaleBarIcon filled={scaleBarVisibility} />
    </IconToggle>
  );
}
