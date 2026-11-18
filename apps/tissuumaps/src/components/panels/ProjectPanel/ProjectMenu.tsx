import {
  CopyIcon,
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
  useCopyProjectLink,
  useOpenProjectFromFile,
  useOpenProjectFromURL,
  useOpenWorkspace,
  useSaveProjectAs,
  useSaveProjectToSourceFile,
  workspaceUnsupportedMessage,
} from "./hooks";

export type ProjectMenuProps = {
  className?: string;
};

export function ProjectMenu({ className }: ProjectMenuProps) {
  const openProjectFromFile = useOpenProjectFromFile();
  const openProjectFromURL = useOpenProjectFromURL();
  const saveProjectToSourceFile = useSaveProjectToSourceFile();
  const saveProjectAs = useSaveProjectAs();
  const copyProjectLink = useCopyProjectLink();
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
        <DropdownMenuItem
          disabled={saveProjectToSourceFile === null}
          onClick={saveProjectToSourceFile ?? undefined}
        >
          <SaveIcon />
          Save project
          {saveProjectToSourceFile === null && (
            <DropdownMenuItemDescription>
              Only for projects opened from or saved to the connected folder
            </DropdownMenuItemDescription>
          )}
        </DropdownMenuItem>
        <DropdownMenuItem
          disabled={saveProjectAs === null}
          onClick={saveProjectAs ?? undefined}
        >
          <SaveAllIcon />
          Save as…
          {saveProjectAs === null && (
            <DropdownMenuItemDescription>
              Needs a connected folder
            </DropdownMenuItemDescription>
          )}
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => saveAndDownloadProjectToJSON()}>
          <DownloadIcon />
          Download project
        </DropdownMenuItem>
        <DropdownMenuItem
          disabled={copyProjectLink === null || !window.isSecureContext}
          onClick={copyProjectLink ?? undefined}
        >
          <CopyIcon />
          Copy share link
          {copyProjectLink === null ? (
            <DropdownMenuItemDescription>
              Only for projects opened from a URL
            </DropdownMenuItemDescription>
          ) : (
            !window.isSecureContext && (
              <DropdownMenuItemDescription>
                Copying needs a secure (https) connection
              </DropdownMenuItemDescription>
            )
          )}
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
