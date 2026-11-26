export type OpacityIconProps = { filled: boolean };

/** A checkerboard, the pattern of transparency, solid while opacities follow the column */
export function OpacityIcon({ filled }: OpacityIconProps) {
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
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path
        d="M5 3h4v6H3V5a2 2 0 0 1 2-2zM15 3h4a2 2 0 0 1 2 2v4h-6zM9 9h6v6H9zM3 15h6v6H5a2 2 0 0 1-2-2zM15 15h6v4a2 2 0 0 1-2 2h-4z"
        stroke="none"
        className={filled ? "fill-current" : "fill-current/35"}
      />
    </svg>
  );
}
