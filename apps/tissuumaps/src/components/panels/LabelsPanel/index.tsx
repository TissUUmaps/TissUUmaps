import { useState } from "react";

import { type Labels, createLabels } from "@tissuumaps/core";

import { OpacityControl } from "@/components/common/opacity-control";
import { VisibilityButton } from "@/components/common/visibility-button";
import { AddDataObjectDialog } from "@/components/widgets/AddDataObjectDialog";
import { DataSourceWidget } from "@/components/widgets/DataSourceWidget";
import {
  SortableObjectList,
  SortableObjectListItem,
} from "@/components/widgets/ObjectList";
import { useExpandedObjectIds } from "@/hooks/useExpandedObjectIds";
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
  const [expandedIds, setExpandedIds] = useExpandedObjectIds("labels");
  const labelsDataProviders = useAppStore((state) => state.labelsDataProviders);

  const layers = useProjectStore((state) => state.layers);
  const labels = useProjectStore((state) => state.labels);
  const addLabels = useProjectStore((state) => state.addLabels);
  const moveLabels = useProjectStore((state) => state.moveLabels);

  return (
    <div className={cn("flex flex-col gap-y-2", className)}>
      <SortableObjectList
        objects={labels}
        onMove={moveLabels}
        expandedIds={expandedIds}
        onExpandedIdsChange={setExpandedIds}
      >
        {(currentLabels, index) => (
          <LabelsAccordionItem
            key={currentLabels.id}
            labels={currentLabels}
            index={index}
          />
        )}
      </SortableObjectList>
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

  const [activeSettingsCategory, setActiveSettingsCategory] =
    useState<LabelsSettingsCategory | null>(null);

  return (
    <SortableObjectListItem
      id={labels.id}
      index={index}
      name={labels.name}
      objectLabel="labels"
      onRename={(name) => updateLabels(labels.id, { name })}
      dimmed={!labels.visibility}
      leadingControls={
        <VisibilityButton
          visible={labels.visibility}
          onVisibleChange={(visibility) =>
            updateLabels(labels.id, { visibility })
          }
          objectLabel="labels"
          name={labels.name}
        />
      }
      trailingControls={
        <OpacityControl
          opacity={labels.opacity}
          name={labels.name}
          onOpacityChange={(opacity) => updateLabels(labels.id, { opacity })}
        />
      }
      onDelete={() => deleteLabels(labels.id)}
    >
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
    </SortableObjectListItem>
  );
}
