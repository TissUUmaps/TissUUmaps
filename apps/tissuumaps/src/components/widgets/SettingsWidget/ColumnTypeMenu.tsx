import { HashIcon, TagIcon } from "lucide-react";

import type { TableColumnType } from "@tissuumaps/core";

import { IconButton } from "@/components/common/icon-button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItemDescription,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

/** The name of each column type, as shown to the user */
const columnTypeLabels: Record<TableColumnType, string> = {
  categorical: "Categorical",
  numerical: "Numerical",
};

/** The icon of each column type */
const columnTypeIcons: Record<TableColumnType, typeof TagIcon> = {
  categorical: TagIcon,
  numerical: HashIcon,
};

export type ColumnTypeMenuProps = {
  value: TableColumnType;
  onValueChange: (value: TableColumnType) => void;
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
  const { categorical: CategoricalIcon, numerical: NumericalIcon } =
    columnTypeIcons;
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
          onValueChange={(newValue) =>
            onValueChange(newValue as TableColumnType)
          }
        >
          <DropdownMenuRadioItem
            value={"categorical" satisfies TableColumnType}
            closeOnClick
            className="flex-wrap"
          >
            <CategoricalIcon />
            {columnTypeLabels.categorical}
            <DropdownMenuItemDescription>
              Each value gets its own color or marker
            </DropdownMenuItemDescription>
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem
            value={"numerical" satisfies TableColumnType}
            disabled={isCategoricalOnly}
            closeOnClick
            className="flex-wrap"
          >
            <NumericalIcon />
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
