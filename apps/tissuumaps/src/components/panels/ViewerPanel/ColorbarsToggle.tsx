import { useId } from "react";

import { IconToggle } from "@/components/common/icon-toggle";
import { useProjectStore } from "@/stores/project";

type ColorbarIconProps = { className?: string };

function ColorbarIcon({ className }: ColorbarIconProps) {
  const gradientId = useId();
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      className={className}
    >
      <defs>
        <linearGradient id={gradientId}>
          <stop offset="0" stopColor="#440154" />
          <stop offset="0.25" stopColor="#3b528b" />
          <stop offset="0.5" stopColor="#21918c" />
          <stop offset="0.75" stopColor="#5ec962" />
          <stop offset="1" stopColor="#fde725" />
        </linearGradient>
      </defs>
      <g transform="rotate(-45 12 12)">
        <rect
          x="0.5"
          y="8"
          width="23"
          height="8"
          rx="1.5"
          fill={`url(#${gradientId})`}
        />
        <path d="M6 16v3M12 16v3M18 16v3" />
      </g>
    </svg>
  );
}

export type ColorbarsToggleProps = { className?: string };

export function ColorbarsToggle({ className }: ColorbarsToggleProps) {
  const colorbarsVisible = useProjectStore((state) => state.colorbarsVisible);
  const setColorbarsVisible = useProjectStore(
    (state) => state.setColorbarsVisible,
  );

  return (
    <IconToggle
      label="Colorbars"
      className={className}
      pressed={colorbarsVisible}
      onPressedChange={setColorbarsVisible}
    >
      <ColorbarIcon className={colorbarsVisible ? undefined : "grayscale"} />
    </IconToggle>
  );
}
