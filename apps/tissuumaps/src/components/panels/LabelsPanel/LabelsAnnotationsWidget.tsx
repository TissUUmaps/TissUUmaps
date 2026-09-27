import { useMemo } from "react";

import {
  type Labels,
  defaultLabelColor,
  defaultLabelOpacity,
  defaultLabelVisibility,
} from "@tissuumaps/core";

import { AnnotationsWidget } from "@/components/widgets/AnnotationsWidget";
import { useColorGroupValues } from "@/components/widgets/AnnotationsWidget/useColorGroupValues";
import { useGroupColumn } from "@/components/widgets/AnnotationsWidget/useGroupColumn";
import { useGroupTable } from "@/components/widgets/AnnotationsWidget/useGroupTable";
import { useGroupVisibility } from "@/components/widgets/AnnotationsWidget/useGroupVisibility";
import { useOpacityGroupValues } from "@/components/widgets/AnnotationsWidget/useOpacityGroupValues";
import { useVisibilityGroupValues } from "@/components/widgets/AnnotationsWidget/useVisibilityGroupValues";
import { useProjectStore } from "@/stores/project";

import { LabelsSettingsCategory } from "./category";

export type LabelsAnnotationsWidgetProps = {
  labels: Labels;
  activeSettingsCategory: LabelsSettingsCategory | null;
  className?: string;
};

export function LabelsAnnotationsWidget({
  labels,
  activeSettingsCategory,
  className,
}: LabelsAnnotationsWidgetProps) {
  const updateLabels = useProjectStore((state) => state.updateLabels);

  const tableId = labels.dataSource.table ?? null;
  const annotatedObject = useMemo(() => ({ labelsId: labels.id }), [labels.id]);
  const groupTable = useGroupTable(
    labels.name,
    tableId,
    [
      {
        category: LabelsSettingsCategory.labelColor,
        config: labels.labelColor,
      },
      {
        category: LabelsSettingsCategory.labelVisibility,
        config: labels.labelVisibility,
      },
      {
        category: LabelsSettingsCategory.labelOpacity,
        config: labels.labelOpacity,
      },
    ],
    activeSettingsCategory,
  );

  const colorAdapter = useColorGroupValues();
  const visibilityAdapter = useVisibilityGroupValues();
  const opacityAdapter = useOpacityGroupValues();

  const groupVisibility = useGroupVisibility(groupTable, {
    name: "visibility",
    default: defaultLabelVisibility,
    config: labels.labelVisibility,
    onConfigChange: (labelVisibility) =>
      updateLabels(labels.id, { labelVisibility }),
    adapter: visibilityAdapter,
  });
  const colorColumn = useGroupColumn(groupTable, {
    name: "color",
    isShownByDefault: true,
    default: defaultLabelColor,
    config: labels.labelColor,
    onConfigChange: (labelColor) => updateLabels(labels.id, { labelColor }),
    adapter: colorAdapter,
  });
  const opacityColumn = useGroupColumn(groupTable, {
    name: "opacity",
    default: defaultLabelOpacity,
    config: labels.labelOpacity,
    onConfigChange: (labelOpacity) => updateLabels(labels.id, { labelOpacity }),
    adapter: opacityAdapter,
  });

  const groupColumnDefs = useMemo(
    () =>
      [colorColumn, opacityColumn].filter(
        (columnDef) => columnDef !== undefined,
      ),
    [colorColumn, opacityColumn],
  );

  return (
    <AnnotationsWidget
      tableHeight={300}
      tableId={tableId}
      annotatedObject={annotatedObject}
      selectedGroupByColumn={groupTable.column}
      onSelectedGroupByColumnChange={groupTable.setColumn}
      groupCounts={groupTable.groupCounts}
      groupVisibility={groupVisibility}
      groupColumnDefs={groupColumnDefs}
      className={className}
    />
  );
}
