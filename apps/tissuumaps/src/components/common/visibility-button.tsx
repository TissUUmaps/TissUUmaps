import { EyeIcon, EyeOffIcon } from "lucide-react";

import {
  IconButton,
  type IconButtonProps,
} from "@/components/common/icon-button";

export type VisibilityButtonProps = Omit<
  IconButtonProps,
  "label" | "children" | "onClick"
> & {
  visible: boolean;
  onVisibleChange: (visible: boolean) => void;
  objectLabel: string;

  /** The name of the object, which tells the rows apart for screen readers */
  name?: string;
};

export function VisibilityButton({
  visible,
  onVisibleChange,
  objectLabel,
  name,
  ...props
}: VisibilityButtonProps) {
  const object = name === undefined ? objectLabel : `${objectLabel} ${name}`;
  return (
    <IconButton
      label={visible ? `Hide ${object}` : `Show ${object}`}
      onClick={() => onVisibleChange(!visible)}
      {...props}
    >
      {visible ? <EyeIcon /> : <EyeOffIcon />}
    </IconButton>
  );
}
