import {
  DownloadIcon,
  EllipsisIcon,
  FileIcon,
  FolderIcon,
  LinkIcon,
  XIcon,
} from "lucide-react";

import { IconButton } from "@/components/common/icon-button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuItemDescription,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { isWorkspaceSupported } from "@/data/io/workspace";

import {
  useProjectActions,
  workspaceUnsupportedMessage,
} from "./useProjectActions";

export type ProjectMenuProps = {
  className?: string;
};

export function ProjectMenu({ className }: ProjectMenuProps) {
  const {
    openProjectFile,
    openProjectFromURL,
    downloadProject,
    closeProject,
    openWorkspace,
  } = useProjectActions();
  const workspaceSupported = isWorkspaceSupported();

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={<IconButton label="Project menu" className={className} />}
      >
        <EllipsisIcon />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuItem onClick={openProjectFile}>
          <FileIcon />
          Open project file…
        </DropdownMenuItem>
        <DropdownMenuItem onClick={openProjectFromURL}>
          <LinkIcon />
          Open from URL…
        </DropdownMenuItem>
        <DropdownMenuItem
          disabled={!workspaceSupported}
          onClick={openWorkspace}
        >
          <FolderIcon />
          Open folder…
          {!workspaceSupported && (
            <DropdownMenuItemDescription>
              {workspaceUnsupportedMessage}
            </DropdownMenuItemDescription>
          )}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={downloadProject}>
          <DownloadIcon />
          Download project
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem variant="destructive" onClick={closeProject}>
          <XIcon />
          Close project…
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
