import {
  DownloadIcon,
  FolderIcon,
  LinkIcon,
  type LucideIcon,
} from "lucide-react";

import { SourceUtils } from "@tissuumaps/core";

import { IconButton } from "@/components/common/icon-button";
import { objectKindIcons } from "@/components/object-kind-icons";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

import { type PanelId, panelIds } from "../panelIds";
import { ProjectMenu } from "./ProjectMenu";
import { useProjectActions } from "./useProjectActions";

export type ProjectHeaderProps = {
  onShowPanel: (panelId: PanelId) => void;
  className?: string;
};

export function ProjectHeader({ onShowPanel, className }: ProjectHeaderProps) {
  const name = useProjectStore((state) => state.name);
  const setName = useProjectStore((state) => state.setName);
  const source = useProjectStore((state) => state.source);
  const workspaceName = useAppStore((state) => state.workspace?.name ?? null);
  const imageCount = useProjectStore((state) => state.images.length);
  const labelsCount = useProjectStore((state) => state.labels.length);
  const pointsCount = useProjectStore((state) => state.points.length);
  const shapesCount = useProjectStore((state) => state.shapes.length);
  const tableCount = useProjectStore((state) => state.tables.length);
  const { downloadProject } = useProjectActions();

  const dataCounts: DataCount[] = [
    {
      panelId: panelIds.images,
      icon: objectKindIcons.image,
      count: imageCount,
      singularLabel: "image",
      pluralLabel: "images",
    },
    {
      panelId: panelIds.labels,
      icon: objectKindIcons.labels,
      count: labelsCount,
      singularLabel: "labels",
      pluralLabel: "labels",
    },
    {
      panelId: panelIds.points,
      icon: objectKindIcons.points,
      count: pointsCount,
      singularLabel: "point cloud",
      pluralLabel: "point clouds",
    },
    {
      panelId: panelIds.shapes,
      icon: objectKindIcons.shapes,
      count: shapesCount,
      singularLabel: "shape cloud",
      pluralLabel: "shape clouds",
    },
    {
      panelId: panelIds.tables,
      icon: objectKindIcons.table,
      count: tableCount,
      singularLabel: "table",
      pluralLabel: "tables",
    },
  ].filter((dataCount) => dataCount.count > 0);

  const sourceLabel =
    source !== null ? formatProjectSource(source, workspaceName) : null;
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
        <IconButton label="Download project" onClick={downloadProject}>
          <DownloadIcon />
        </IconButton>
        <ProjectMenu />
      </div>
      {sourceLabel !== null && (
        <div className="text-muted-foreground flex min-w-0 items-center gap-1.5 px-1.5 text-xs">
          <SourceIcon className="size-3.5 shrink-0" />
          <span className="truncate" title={source ?? undefined}>
            {sourceLabel}
          </span>
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

// Shows workspace files relative to the folder, and URLs on this server
// relative to the app
function formatProjectSource(
  source: string,
  workspaceName: string | null,
): string {
  if (SourceUtils.isWorkspacePath(source)) {
    return `${workspaceName ?? "Folder"} › ${source.slice(1)}`;
  }
  if (!URL.canParse(source)) {
    return source;
  }
  const url = new URL(source);
  const appUrl = new URL(".", document.baseURI);
  return url.origin === appUrl.origin &&
    url.pathname.startsWith(appUrl.pathname)
    ? url.pathname.slice(appUrl.pathname.length)
    : `${url.host}${url.pathname}`;
}
