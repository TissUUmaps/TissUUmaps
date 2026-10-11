import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  AccordionTrigger,
  AccordionTriggerRightDownIcon,
} from "@/components/common/accordion";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

import { LayersWidget } from "./LayersWidget";
import { ProjectFooter } from "./ProjectFooter";
import { ProjectHeader } from "./ProjectHeader";
import { ProjectWelcomeView } from "./ProjectWelcomeView";
import { RenderSettingsWidget } from "./RenderSettingsWidget";
import { WorkspaceWidget } from "./WorkspaceWidget";

export type ProjectPanelProps = {
  className?: string;
};

export function ProjectPanel({ className }: ProjectPanelProps) {
  const projectOpen = useAppStore((state) => state.projectOpen);
  const layerCount = useProjectStore((state) => state.layers.length);

  return (
    <div className={cn("flex min-h-full flex-col", className)}>
      {!projectOpen ? (
        <ProjectWelcomeView />
      ) : (
        <>
          <ProjectHeader />
          <Accordion multiple defaultValue={["layers"]}>
            <AccordionItem value="layers" className="border-t">
              <AccordionHeader className="text-muted-foreground h-10 gap-1 px-1">
                <AccordionTriggerRightDownIcon />
                <AccordionTrigger className="hover:text-foreground min-w-0 flex-1 gap-2 self-stretch">
                  <span className="text-xs font-semibold tracking-wider uppercase">
                    Layers
                  </span>
                  <span className="ml-auto truncate pr-1 text-xs group-aria-expanded/accordion-trigger:hidden">
                    {layerCount} {layerCount === 1 ? "layer" : "layers"}
                  </span>
                </AccordionTrigger>
              </AccordionHeader>
              <AccordionPanel className="px-1 pb-3">
                <LayersWidget />
              </AccordionPanel>
            </AccordionItem>
            <AccordionItem value="renderSettings" className="border-t">
              <AccordionHeader className="text-muted-foreground h-10 gap-1 px-1">
                <AccordionTriggerRightDownIcon />
                <AccordionTrigger className="hover:text-foreground min-w-0 flex-1 gap-2 self-stretch">
                  <span className="text-xs font-semibold tracking-wider uppercase">
                    Render settings
                  </span>
                </AccordionTrigger>
              </AccordionHeader>
              <AccordionPanel className="px-1 pb-3">
                <RenderSettingsWidget />
              </AccordionPanel>
            </AccordionItem>
          </Accordion>
        </>
      )}
      <div className="mt-auto pt-2">
        <WorkspaceWidget />
        <ProjectFooter />
      </div>
    </div>
  );
}
