import {
  DownloadIcon,
  EllipsisIcon,
  FileIcon,
  FileInputIcon,
  FolderIcon,
  ImportIcon,
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
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { isWorkspaceSupported } from "@/data/io/workspace";
import { useAppStore } from "@/stores/app";

import {
  useCloseProject,
  useDownloadProject,
  useImportProjectFromFile,
  useImportProjectFromURL,
  useOpenProjectFromFile,
  useOpenProjectFromURL,
  useOpenWorkspace,
  workspaceUnsupportedMessage,
} from "./hooks";

export type ProjectMenuProps = {
  className?: string;
};

export function ProjectMenu({ className }: ProjectMenuProps) {
  const projectProviders = useAppStore((state) => state.projectProviders);
  const writingProjectProviders = Array.from(projectProviders).filter(
    ([, projectProvider]) => projectProvider.toFile !== undefined,
  );

  const openProjectFromFile = useOpenProjectFromFile();
  const openProjectFromURL = useOpenProjectFromURL();
  const importProjectFromFile = useImportProjectFromFile();
  const importProjectFromURL = useImportProjectFromURL();
  const closeProject = useCloseProject();
  const downloadProject = useDownloadProject();
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
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={importProjectFromFile}>
          <FileInputIcon />
          Import project file…
        </DropdownMenuItem>
        <DropdownMenuItem onClick={importProjectFromURL}>
          <ImportIcon />
          Import from URL…
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          disabled={!workspaceSupported}
          onClick={openWorkspace}
        >
          <FolderIcon />
          Open workspace…
          {!workspaceSupported && (
            <DropdownMenuItemDescription>
              {workspaceUnsupportedMessage}
            </DropdownMenuItemDescription>
          )}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => downloadProject()}>
          <DownloadIcon />
          Download project
        </DropdownMenuItem>
        {/* a choice only once another project provider can write projects */}
        {writingProjectProviders.length > 1 && (
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>
              <DownloadIcon />
              Download as
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              {writingProjectProviders.map(([type, projectProvider]) => (
                <DropdownMenuItem
                  key={type}
                  onClick={() => downloadProject(projectProvider)}
                >
                  {projectProvider.name}
                </DropdownMenuItem>
              ))}
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        )}
        <DropdownMenuSeparator />
        <DropdownMenuItem variant="destructive" onClick={closeProject}>
          <XIcon />
          Close project…
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
