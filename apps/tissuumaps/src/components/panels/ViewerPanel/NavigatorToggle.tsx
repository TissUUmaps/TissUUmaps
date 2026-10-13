import { IconToggle } from "@/components/common/icon-toggle";
import { useProjectStore } from "@/stores/project";

function NavigatorIcon() {
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
      <path d="M21.06 7.77A10 10 0 0 1 7.77 21.06M2.94 16.23A10 10 0 0 1 16.23 2.94" />
      <g transform="rotate(-45 12 12)">
        <path d="M12 9.5 19 12 12 14.5Z" fill="currentColor" />
        <path d="M12 9.5 5 12 12 14.5" />
      </g>
    </svg>
  );
}

export type NavigatorToggleProps = { className?: string };

export function NavigatorToggle({ className }: NavigatorToggleProps) {
  const osOptions = useProjectStore((state) => state.osOptions);
  const setOSOptions = useProjectStore((state) => state.setOSOptions);

  return (
    <IconToggle
      label="Navigator"
      className={className}
      pressed={osOptions.viewerOptions.showNavigator !== false}
      onPressedChange={(pressed) => {
        setOSOptions({
          ...osOptions,
          viewerOptions: { ...osOptions.viewerOptions, showNavigator: pressed },
        });
      }}
    >
      <NavigatorIcon />
    </IconToggle>
  );
}
