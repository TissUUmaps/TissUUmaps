import { DragDropProvider } from "@dnd-kit/react";
import { isSortable, useSortable } from "@dnd-kit/react/sortable";
import { EyeIcon, EyeOffIcon, GripVertical, Trash2Icon } from "lucide-react";
import { useState } from "react";

import { type Labels, MathUtils, createLabels } from "@tissuumaps/core";

import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  AccordionTrigger,
  AccordionTriggerDownUpIcon,
} from "@/components/common/accordion";
import { useConfirmDialog } from "@/components/dialogs/ConfirmDialog/hooks";
import { Button } from "@/components/ui/button";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import { AddDataObjectDialog } from "@/components/widgets/AddDataObjectDialog";
import { DataSourceWidget } from "@/components/widgets/DataSourceWidget";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

import { LabelsAnnotationsWidget } from "./LabelsAnnotationsWidget";
import { LabelsSettingsWidget } from "./LabelsSettingsWidget";
import type { LabelsSettingsCategory } from "./category";

export type LabelsPanelProps = {
  className?: string;
};

export function LabelsPanel({ className }: LabelsPanelProps) {
  const labelsDataProviders = useAppStore((state) => state.labelsDataProviders);

  const layers = useProjectStore((state) => state.layers);
  const labels = useProjectStore((state) => state.labels);
  const addLabels = useProjectStore((state) => state.addLabels);
  const moveLabels = useProjectStore((state) => state.moveLabels);

  return (
    <div className={cn("flex flex-col gap-y-2", className)}>
      <DragDropProvider
        onDragEnd={(event) => {
          const { source, canceled } = event.operation;
          if (isSortable(source) && !canceled) {
            // dnd-kit optimistically updates the DOM
            // https://github.com/clauderic/dnd-kit/issues/1564
            moveLabels(source.id as string, source.index);
          }
        }}
      >
        <Accordion multiple className="gap-y-2">
          {labels.map((currentLabels, index) => (
            <LabelsAccordionItem
              key={currentLabels.id}
              labels={currentLabels}
              index={index}
            />
          ))}
        </Accordion>
      </DragDropProvider>
      <AddDataObjectDialog
        title="Add labels"
        layers={layers}
        dataProviders={labelsDataProviders}
        onAdd={(name, layerId, dataSource) => {
          if (!layerId) return;
          const newLabels = createLabels({
            id: crypto.randomUUID(),
            name,
            dataSource,
            layer: layerId,
          });
          addLabels(newLabels);
        }}
      />
    </div>
  );
}

type LabelsAccordionItemProps = {
  labels: Labels;
  index: number;
};

function LabelsAccordionItem({ labels, index }: LabelsAccordionItemProps) {
  const labelsDataProviders = useAppStore((state) => state.labelsDataProviders);

  const updateLabels = useProjectStore((state) => state.updateLabels);
  const deleteLabels = useProjectStore((state) => state.deleteLabels);
  const confirm = useConfirmDialog();

  const [activeSettingsCategory, setActiveSettingsCategory] =
    useState<LabelsSettingsCategory | null>(null);

  const { ref, handleRef } = useSortable({ id: labels.id, index });

  return (
    <div ref={ref}>
      <AccordionItem className="border rounded-md bg-sidebar p-2">
        <AccordionHeader>
          <GripVertical ref={handleRef} />
          <div className="flex-1 w-full">
            <AccordionTrigger className="w-full cursor-pointer">
              {labels.name}
            </AccordionTrigger>
          </div>
          <div className="ml-auto flex flex-row items-center gap-x-2">
            <InputGroup className="w-20">
              <InputGroupAddon>&alpha;</InputGroupAddon>
              <InputGroupInput
                type="number"
                aria-label="Opacity"
                inputMode="decimal"
                step={0.05}
                min={0}
                max={1}
                value={labels.opacity}
                onChange={(event) => {
                  const newValue = event.target.valueAsNumber;
                  if (!isNaN(newValue)) {
                    updateLabels(labels.id, {
                      opacity: MathUtils.clamp(newValue, 0, 1),
                    });
                  }
                }}
              />
            </InputGroup>
            <Button
              variant="ghost"
              aria-label={labels.visibility ? "Hide labels" : "Show labels"}
              onClick={() =>
                updateLabels(labels.id, { visibility: !labels.visibility })
              }
            >
              {labels.visibility ? <EyeIcon /> : <EyeOffIcon />}
            </Button>
            <Button
              variant="ghost"
              aria-label="Delete labels"
              onClick={() => {
                void confirm({
                  title: "Delete labels",
                  body: "Are you sure you want to delete these labels? This action cannot be undone.",
                }).then((confirmed) => {
                  if (confirmed) {
                    deleteLabels(labels.id);
                  }
                });
              }}
              title="Delete labels"
            >
              <Trash2Icon />
            </Button>
          </div>
          <AccordionTriggerDownUpIcon />
        </AccordionHeader>
        <AccordionPanel className="pt-2 flex flex-col gap-y-2">
          <DataSourceWidget
            dataSource={labels.dataSource}
            dataProviders={labelsDataProviders}
            onDataSourceChange={(newDataSource) => {
              updateLabels(labels.id, { dataSource: newDataSource });
            }}
            className="bg-card"
          />
          <LabelsSettingsWidget
            labels={labels}
            activeCategory={activeSettingsCategory}
            onActiveCategoryChange={setActiveSettingsCategory}
            className="bg-card"
          />
          {labels.dataSource.table !== undefined && (
            <LabelsAnnotationsWidget
              labels={labels}
              activeSettingsCategory={activeSettingsCategory}
              className="bg-card"
            />
          )}
        </AccordionPanel>
      </AccordionItem>
    </div>
  );
}
