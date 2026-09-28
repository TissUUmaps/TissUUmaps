import { EllipsisIcon, PencilIcon, Trash2Icon } from "lucide-react";

import { IconButton } from "@/components/common/icon-button";
import { useConfirmDialog } from "@/components/dialogs/ConfirmDialog/hooks";
import { usePromptDialog } from "@/components/dialogs/PromptDialog/hooks";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuItemDescription,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export type ObjectMenuProps = {
  name: string;
  objectLabel: string;
  onRename: (name: string) => void;
  deleteDisabledReason?: string;
  onDelete: () => void;
  className?: string;
};

export function ObjectMenu({
  name,
  objectLabel,
  onRename,
  deleteDisabledReason,
  onDelete,
  className,
}: ObjectMenuProps) {
  const confirm = useConfirmDialog();
  const prompt = usePromptDialog();

  const displayName = name || "Untitled";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={
          <IconButton label={`${displayName} actions`} className={className} />
        }
      >
        <EllipsisIcon />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuItem
          onClick={() => {
            void prompt({
              title: `Rename ${objectLabel}`,
              defaultValue: name,
              inputProps: { required: true },
            }).then((value) => {
              const newName = value?.trim();
              if (newName) {
                onRename(newName);
              }
            });
          }}
        >
          <PencilIcon />
          Rename…
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          variant="destructive"
          disabled={deleteDisabledReason !== undefined}
          onClick={() => {
            void confirm({
              title: `Delete ${objectLabel}`,
              body: `Are you sure you want to delete "${displayName}"? This action cannot be undone.`,
            }).then((confirmed) => {
              if (confirmed) {
                onDelete();
              }
            });
          }}
        >
          <Trash2Icon />
          Delete {objectLabel}
          {deleteDisabledReason !== undefined && (
            <DropdownMenuItemDescription>
              {deleteDisabledReason}
            </DropdownMenuItemDescription>
          )}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
