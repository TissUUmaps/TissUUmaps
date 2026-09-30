import {
  DownloadIcon,
  FolderIcon,
  LinkIcon,
  type LucideIcon,
  SaveIcon,
} from "lucide-react";

import { SourceUtils } from "@tissuumaps/core";

import { IconButton } from "@/components/common/icon-button";
import { objectKindIcons } from "@/components/object-kind-icons";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  hasUnsavedChanges,
  saveAndDownloadProjectToJSON,
} from "@/data/io/project";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

import { PanelId } from "../panelId";
import { ProjectMenu } from "./ProjectMenu";
import { formatProjectSource } from "./formatProjectSource";
import { useSaveProjectToFolder } from "./hooks";

export type ProjectHeaderProps = {
  onShowPanel: (panelId: PanelId) => void;
  className?: string;
};

export function ProjectHeader({ onShowPanel, className }: ProjectHeaderProps) {
  const name = useProjectStore((state) => state.name);
  const setName = useProjectStore((state) => state.setName);
  const source = useProjectStore((state) => state.source);
  // Only shown when Save can clear it
  const unsavedChanges = useProjectStore(
    (state) => state.sourceFile !== null && hasUnsavedChanges(state),
  );
  const workspaceName = useAppStore((state) => state.workspace?.name ?? null);
  const imageCount = useProjectStore((state) => state.images.length);
  const labelsCount = useProjectStore((state) => state.labels.length);
  const pointsCount = useProjectStore((state) => state.points.length);
  const shapesCount = useProjectStore((state) => state.shapes.length);
  const tableCount = useProjectStore((state) => state.tables.length);
  const saveProjectToFolder = useSaveProjectToFolder();

  const dataCounts: DataCount[] = [
    {
      panelId: PanelId.images,
      icon: objectKindIcons.image,
      count: imageCount,
      singularLabel: "image",
      pluralLabel: "images",
    },
    {
      panelId: PanelId.labels,
      icon: objectKindIcons.labels,
      count: labelsCount,
      singularLabel: "labels",
      pluralLabel: "labels",
    },
    {
      panelId: PanelId.points,
      icon: objectKindIcons.points,
      count: pointsCount,
      singularLabel: "point cloud",
      pluralLabel: "point clouds",
    },
    {
      panelId: PanelId.shapes,
      icon: objectKindIcons.shapes,
      count: shapesCount,
      singularLabel: "shape cloud",
      pluralLabel: "shape clouds",
    },
    {
      panelId: PanelId.tables,
      icon: objectKindIcons.table,
      count: tableCount,
      singularLabel: "table",
      pluralLabel: "tables",
    },
  ].filter((dataCount) => dataCount.count > 0);

  const sourceLabel =
    source !== null
      ? formatProjectSource(source, workspaceName, document.baseURI)
      : null;
  const SourceIcon =
    source !== null && SourceUtils.isWorkspacePath(source)
      ? FolderIcon
      : LinkIcon;

  return (
    <div className={cn("flex flex-col gap-1 px-1 pt-1 pb-3", className)}>
      <div className="flex items-center gap-0.5">
        <Input
          aria-label="Project name"
          value={name}
          onChange={(event) => setName(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.currentTarget.blur();
            }
          }}
          className="hover:border-input h-8 flex-1 border-transparent bg-transparent px-1.5 text-base font-semibold shadow-none md:text-base dark:bg-transparent"
        />
        {saveProjectToFolder !== null && (
          <IconButton label="Save project" onClick={saveProjectToFolder}>
            <SaveIcon />
          </IconButton>
        )}
        <IconButton
          label="Download project"
          onClick={() => saveAndDownloadProjectToJSON()}
        >
          <DownloadIcon />
        </IconButton>
        <ProjectMenu />
      </div>
      {(sourceLabel !== null || unsavedChanges) && (
        <div className="text-muted-foreground flex min-w-0 items-center gap-1.5 px-1.5 text-xs">
          {sourceLabel !== null && (
            <>
              <SourceIcon className="size-3.5 shrink-0" />
              <span className="truncate" title={source ?? undefined}>
                {sourceLabel}
              </span>
            </>
          )}
          {sourceLabel !== null && unsavedChanges && <span>·</span>}
          {unsavedChanges && (
            <span className="text-foreground shrink-0 font-medium">
              Unsaved changes
            </span>
          )}
        </div>
      )}
      {dataCounts.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5 px-1.5">
          {dataCounts.map(
            ({ panelId, icon: Icon, count, singularLabel, pluralLabel }) => (
              <Button
                key={panelId}
                variant="secondary"
                size="xs"
                className="font-normal"
                onClick={() => onShowPanel(panelId)}
              >
                <Icon className="text-muted-foreground" />
                {count} {count === 1 ? singularLabel : pluralLabel}
              </Button>
            ),
          )}
        </div>
      )}
    </div>
  );
}

type DataCount = {
  panelId: PanelId;
  icon: LucideIcon;
  count: number;
  singularLabel: string;
  pluralLabel: string;
};
