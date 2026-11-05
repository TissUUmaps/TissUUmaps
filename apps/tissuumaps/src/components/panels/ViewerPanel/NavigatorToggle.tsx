import { IconToggle } from "@/components/common/icon-toggle";
import { useProjectStore } from "@/stores/project";

type NavigatorIconProps = { filled: boolean };

function NavigatorIcon({ filled }: NavigatorIconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <circle cx="12" cy="12" r="10.25" />
      {/* a filled needle has no outline, which would hide its red half */}
      <g transform="rotate(-45 12 12)" stroke={filled ? "none" : undefined}>
        <path
          d="M12 8 20.5 12 12 16Z"
          className={filled ? "fill-red-500" : undefined}
        />
        <path
          d="M12 8 3.5 12 12 16Z"
          className={filled ? "fill-current" : undefined}
        />
      </g>
    </svg>
  );
}

export type NavigatorToggleProps = { className?: string };

export function NavigatorToggle({ className }: NavigatorToggleProps) {
  const osOptions = useProjectStore((state) => state.osOptions);
  const setOSOptions = useProjectStore((state) => state.setOSOptions);

  const navigatorVisible = osOptions.viewerOptions.showNavigator !== false;

  return (
    <IconToggle
      label="Navigator"
      className={className}
      pressed={navigatorVisible}
      onPressedChange={(pressed) => {
        setOSOptions({
          ...osOptions,
          viewerOptions: { ...osOptions.viewerOptions, showNavigator: pressed },
        });
      }}
    >
      <NavigatorIcon filled={navigatorVisible} />
    </IconToggle>
  );
}
