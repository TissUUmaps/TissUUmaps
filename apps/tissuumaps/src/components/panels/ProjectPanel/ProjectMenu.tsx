import {
  DownloadIcon,
  EllipsisIcon,
  FileIcon,
  FolderIcon,
  LinkIcon,
  SaveAllIcon,
  SaveIcon,
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
  useOpenProjectFile,
  useOpenProjectFromURL,
  useOpenWorkspace,
  useSaveProjectToFolder,
  useSaveProjectToFolderAs,
  workspaceUnsupportedMessage,
} from "./hooks";

export type ProjectMenuProps = {
  className?: string;
};

export function ProjectMenu({ className }: ProjectMenuProps) {
  const openProjectFile = useOpenProjectFile();
  const openProjectFromURL = useOpenProjectFromURL();
  const saveProjectToFolder = useSaveProjectToFolder();
  const saveProjectToFolderAs = useSaveProjectToFolderAs();
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
        <DropdownMenuItem
          disabled={saveProjectToFolder === null}
          onClick={saveProjectToFolder ?? undefined}
        >
          <SaveIcon />
          Save project
          {saveProjectToFolder === null && (
            <DropdownMenuItemDescription>
              Only for projects opened from the connected folder
            </DropdownMenuItemDescription>
          )}
        </DropdownMenuItem>
        <DropdownMenuItem
          disabled={saveProjectToFolderAs === null}
          onClick={saveProjectToFolderAs ?? undefined}
        >
          <SaveAllIcon />
          Save as…
          {saveProjectToFolderAs === null && (
            <DropdownMenuItemDescription>
              Needs a connected folder
            </DropdownMenuItemDescription>
          )}
        </DropdownMenuItem>
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
