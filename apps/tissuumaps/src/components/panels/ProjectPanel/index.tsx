import { Section } from "@/components/common/section";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

import type { PanelId } from "../panelId";
import { DisplaySettingsWidget } from "./DisplaySettingsWidget";
import { LayersWidget } from "./LayersWidget";
import { ProjectFooter } from "./ProjectFooter";
import { ProjectHeader } from "./ProjectHeader";
import { ProjectStartPage } from "./ProjectStartPage";
import { WorkspaceWidget } from "./WorkspaceWidget";

export type ProjectPanelProps = {
  onShowPanel: (panelId: PanelId) => void;
  className?: string;
};

export function ProjectPanel({ onShowPanel, className }: ProjectPanelProps) {
  const startPageDismissed = useAppStore((state) => state.startPageDismissed);
  const layerCount = useProjectStore((state) => state.layers.length);

  return (
    <div className={cn("flex min-h-full flex-col", className)}>
      {!startPageDismissed ? (
        <ProjectStartPage />
      ) : (
        <>
          <ProjectHeader onShowPanel={onShowPanel} />
          <Section
            title="Layers"
            summary={`${layerCount} ${layerCount === 1 ? "layer" : "layers"}`}
            defaultOpen
          >
            <LayersWidget />
          </Section>
          <Section title="Display">
            <DisplaySettingsWidget />
          </Section>
        </>
      )}
      <div className="mt-auto pt-2">
        <WorkspaceWidget />
        <ProjectFooter />
      </div>
    </div>
  );
}
