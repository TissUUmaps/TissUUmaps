import { FileIcon, FilePlusIcon, FolderIcon, LinkIcon } from "lucide-react";

import logoUrl from "@/assets/logo.svg";
import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { isWorkspaceSupported } from "@/data/io/workspace";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";

import {
  useOpenEmptyProject,
  useOpenProjectFromFile,
  useOpenProjectFromURL,
  useOpenWorkspace,
  workspaceUnsupportedMessage,
} from "./hooks";

export type ProjectWelcomeViewProps = {
  className?: string;
};

export function ProjectWelcomeView({ className }: ProjectWelcomeViewProps) {
  const workspaceName = useAppStore((state) => state.workspace?.name ?? null);
  const openProjectFromFile = useOpenProjectFromFile();
  const openProjectFromURL = useOpenProjectFromURL();
  const openEmptyProject = useOpenEmptyProject();
  const openWorkspace = useOpenWorkspace();
  const workspaceSupported = isWorkspaceSupported();

  return (
    <Empty className={cn("flex-none justify-start px-6 py-8", className)}>
      <EmptyHeader>
        <EmptyMedia>
          <img src={logoUrl} alt="" className="h-24" />
        </EmptyMedia>
        <EmptyTitle>Welcome to TissUUmaps</EmptyTitle>
        <EmptyDescription>
          {workspaceName !== null
            ? `Open a project from ${workspaceName}, or start an empty project and add images, labels, points or shapes from the folder.`
            : "Load a TissUUmaps project (.tm4) from your computer or from a link, or open a folder with your data."}
        </EmptyDescription>
      </EmptyHeader>
      {workspaceName !== null ? (
        <EmptyContent>
          <Button className="w-full" onClick={openProjectFromFile}>
            <FileIcon />
            Open project from this folder…
          </Button>
          <div className="grid w-full grid-cols-2 gap-2">
            <Button variant="outline" onClick={openEmptyProject}>
              <FilePlusIcon />
              Start empty
            </Button>
            <Button variant="outline" onClick={openProjectFromURL}>
              <LinkIcon />
              From URL…
            </Button>
          </div>
        </EmptyContent>
      ) : (
        <EmptyContent>
          <Button className="w-full" onClick={openProjectFromFile}>
            <FileIcon />
            Open project file…
          </Button>
          <div className="grid w-full grid-cols-2 gap-2">
            <Button variant="outline" onClick={openProjectFromURL}>
              <LinkIcon />
              From URL…
            </Button>
            <Button
              variant="outline"
              disabled={!workspaceSupported}
              onClick={openWorkspace}
            >
              <FolderIcon />
              Open folder…
            </Button>
          </div>
          {!workspaceSupported && (
            <p className="text-muted-foreground text-xs">
              {workspaceUnsupportedMessage}
            </p>
          )}
          <p className="text-muted-foreground text-sm">
            No project yet?{" "}
            <Button
              variant="link"
              className="text-foreground h-auto p-0 underline"
              onClick={openEmptyProject}
            >
              Start empty
            </Button>{" "}
            and add data from the data tabs.
          </p>
        </EmptyContent>
      )}
    </Empty>
  );
}
