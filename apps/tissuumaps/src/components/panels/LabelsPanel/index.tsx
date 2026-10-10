import { useState } from "react";

import type { Labels } from "@tissuumaps/core";

import { OpacityControl } from "@/components/common/opacity-control";
import { VisibilityButton } from "@/components/common/visibility-button";
import { AddDataObjectButton } from "@/components/controls/AddDataObjectButton";
import {
  SortableObjectList,
  SortableObjectListItem,
} from "@/components/controls/ObjectList";
import { useAddLabelsDialogParams } from "@/components/dialogs/AddDataObjectDialog/hooks";
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
  const expandedLabelsIds = useAppStore((state) => state.expandedLabelsIds);
  const setExpandedLabelsIds = useAppStore(
    (state) => state.setExpandedLabelsIds,
  );
  const addDialogParams = useAddLabelsDialogParams();

  const labels = useProjectStore((state) => state.labels);
  const moveLabels = useProjectStore((state) => state.moveLabels);

  return (
    <div className={cn("flex flex-col gap-y-2", className)}>
      <SortableObjectList
        objects={labels}
        onMove={moveLabels}
        expandedIds={expandedLabelsIds}
        onExpandedIdsChange={setExpandedLabelsIds}
      >
        {(currentLabels, index) => (
          <LabelsAccordionItem
            key={currentLabels.id}
            labels={currentLabels}
            index={index}
          />
        )}
      </SortableObjectList>
      <AddDataObjectButton {...addDialogParams} />
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
