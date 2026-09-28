import { EllipsisIcon, FolderIcon, FolderOpenIcon, XIcon } from "lucide-react";

import { IconButton } from "@/components/common/icon-button";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { isWorkspaceSupported } from "@/data/io/workspace";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";

import { useProjectActions } from "./useProjectActions";

export type WorkspaceWidgetProps = {
  className?: string;
};

export function WorkspaceWidget({ className }: WorkspaceWidgetProps) {
  const workspace = useAppStore((state) => state.workspace);
  const { openWorkspace, closeWorkspace } = useProjectActions();

  if (!isWorkspaceSupported()) {
    return null;
  }

  return (
    <div
      className={cn(
        "flex h-9 items-center gap-2 border-t px-2 text-xs",
        className,
      )}
    >
      <FolderIcon className="text-muted-foreground size-3.5 shrink-0" />
      {workspace !== null ? (
        <>
          <span className="truncate font-medium" title={workspace.name}>
            {workspace.name}
          </span>
          <DropdownMenu>
            <DropdownMenuTrigger
              render={
                <IconButton
                  label="Folder actions"
                  size="icon-xs"
                  className="ml-auto"
                />
              }
            >
              <EllipsisIcon />
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={openWorkspace}>
                <FolderOpenIcon />
                Change folder…
              </DropdownMenuItem>
              <DropdownMenuItem variant="destructive" onClick={closeWorkspace}>
                <XIcon />
                Disconnect
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </>
      ) : (
        <>
          <span className="text-muted-foreground">No folder connected</span>
          <Button
            variant="link"
            size="xs"
            className="ml-auto h-auto p-0"
            onClick={openWorkspace}
          >
            Open folder…
          </Button>
        </>
      )}
    </div>
  );
}
