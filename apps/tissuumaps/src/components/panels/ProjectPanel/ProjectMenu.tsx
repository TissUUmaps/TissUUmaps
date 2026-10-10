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
import { saveAndDownloadProjectToJSON } from "@/data/io/project";
import { isWorkspaceSupported } from "@/data/io/workspace";

import {
  useCloseProject,
  useOpenProjectFromFile,
  useOpenProjectFromURL,
  useOpenWorkspace,
  workspaceUnsupportedMessage,
} from "./hooks";

export type ProjectMenuProps = {
  className?: string;
};

export function ProjectMenu({ className }: ProjectMenuProps) {
  const openProjectFromFile = useOpenProjectFromFile();
  const openProjectFromURL = useOpenProjectFromURL();
  const closeProject = useCloseProject();
  const openWorkspace = useOpenWorkspace();
  const workspaceSupported = isWorkspaceSupported();

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        render={<IconButton label="Project menu" className={className} />}
      >
        <EllipsisIcon />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuItem onClick={openProjectFromFile}>
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
        <DropdownMenuItem onClick={() => saveAndDownloadProjectToJSON()}>
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
