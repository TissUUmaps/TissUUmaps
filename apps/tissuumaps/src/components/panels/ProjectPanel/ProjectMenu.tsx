import {
  CopyIcon,
  DownloadIcon,
  EllipsisIcon,
  FileIcon,
  FileInputIcon,
  FolderIcon,
  ImportIcon,
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
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { hasUnsavedChanges } from "@/data/io/project";
import { isWorkspaceSupported } from "@/data/io/workspace";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

import {
  useCloseProject,
  useCopyProjectLink,
  useDownloadProject,
  useImportProjectFromFile,
  useImportProjectFromURL,
  useOpenProjectFromFile,
  useOpenProjectFromURL,
  useOpenWorkspace,
  workspaceUnsupportedMessage,
} from "./hooks";

export type ProjectMenuProps = {
  /** Whether the project is being saved, during which saving is disabled */
  saving: boolean;
  onSaveProject: () => void;
  onSaveProjectAs: () => void;
  className?: string;
};

export function ProjectMenu({
  saving,
  onSaveProject,
  onSaveProjectAs,
  className,
}: ProjectMenuProps) {
  const unsavedChanges = useProjectStore(hasUnsavedChanges);
  const canSave = useProjectStore(
    (state) => state.saveTarget?.projectProvider.toFile !== undefined,
  );
  const workspaceOpen = useAppStore((state) => state.workspace !== null);
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
  const copyProjectLink = useCopyProjectLink();
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
        <DropdownMenuItem
          disabled={!canSave || !unsavedChanges || saving}
          onClick={onSaveProject}
        >
          <SaveIcon />
          Save project
          {!canSave ? (
            <DropdownMenuItemDescription>
              Use Save as to save to a file in the workspace
            </DropdownMenuItemDescription>
          ) : (
            !unsavedChanges && (
              <DropdownMenuItemDescription>
                No unsaved changes
              </DropdownMenuItemDescription>
            )
          )}
        </DropdownMenuItem>
        <DropdownMenuItem
          disabled={!workspaceOpen || saving}
          onClick={onSaveProjectAs}
        >
          <SaveAllIcon />
          Save as…
          {!workspaceOpen && (
            <DropdownMenuItemDescription>
              Needs an open workspace
            </DropdownMenuItemDescription>
          )}
        </DropdownMenuItem>
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
        <DropdownMenuItem
          disabled={!window.isSecureContext}
          onClick={copyProjectLink}
        >
          <CopyIcon />
          Copy share link
          {!window.isSecureContext && (
            <DropdownMenuItemDescription>
              Copying needs a secure (https) connection
            </DropdownMenuItemDescription>
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
