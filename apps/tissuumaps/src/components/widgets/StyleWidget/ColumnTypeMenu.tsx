import { HashIcon, TagIcon } from "lucide-react";

import { IconButton } from "@/components/common/icon-button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItemDescription,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

import type { ColumnType } from "./styleBy";

const columnTypeLabels: Record<ColumnType, string> = {
  categorical: "Categorical",
  numerical: "Numerical",
};

const columnTypeIcons: Record<ColumnType, typeof TagIcon> = {
  categorical: TagIcon,
  numerical: HashIcon,
};

export type ColumnTypeMenuProps = {
  value: ColumnType;
  onValueChange: (value: ColumnType) => void;
  /** Whether the column can only be categorical, e.g. a text column */
  isCategoricalOnly: boolean;
  disabled?: boolean;
};

/**
 * The type of the column styled by, as an icon button before its input that
 * opens a menu of both types
 */
export function ColumnTypeMenu({
  value,
  onValueChange,
  isCategoricalOnly,
  disabled = false,
}: ColumnTypeMenuProps) {
  const ActiveIcon = columnTypeIcons[value];
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        disabled={disabled}
        render={
          <IconButton label={`Column type: ${columnTypeLabels[value]}`} />
        }
      >
        <ActiveIcon />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-64">
        <DropdownMenuRadioGroup
          value={value}
          onValueChange={(newValue) => onValueChange(newValue as ColumnType)}
        >
          <DropdownMenuRadioItem
            value={"categorical" satisfies ColumnType}
            closeOnClick
            className="flex-wrap"
          >
            <TagIcon />
            {columnTypeLabels.categorical}
            <DropdownMenuItemDescription>
              Each value gets its own color or marker
            </DropdownMenuItemDescription>
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem
            value={"numerical" satisfies ColumnType}
            disabled={isCategoricalOnly}
            closeOnClick
            className="flex-wrap"
          >
            <HashIcon />
            {columnTypeLabels.numerical}
            <DropdownMenuItemDescription>
              {isCategoricalOnly
                ? "This column can only be categorical"
                : "The values set colors, sizes, opacities or visibility"}
            </DropdownMenuItemDescription>
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
