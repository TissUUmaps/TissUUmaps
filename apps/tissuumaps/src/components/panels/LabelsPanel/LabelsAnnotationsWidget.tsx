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
import { useOpacityGroupValues } from "@/components/widgets/AnnotationsWidget/useOpacityGroupValues";
import { useVisibilityGroupValues } from "@/components/widgets/AnnotationsWidget/useVisibilityGroupValues";

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
  const tableId = labels.dataSource.table ?? null;
  const groupTable = useGroupTable(
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

  const colorColumn = useGroupColumn(groupTable, {
    name: "color",
    default: defaultLabelColor,
    config: labels.labelColor,
    adapter: colorAdapter,
  });
  const visibilityColumn = useGroupColumn(groupTable, {
    name: "visibility",
    default: defaultLabelVisibility,
    config: labels.labelVisibility,
    adapter: visibilityAdapter,
  });
  const opacityColumn = useGroupColumn(groupTable, {
    name: "opacity",
    default: defaultLabelOpacity,
    config: labels.labelOpacity,
    adapter: opacityAdapter,
  });

  const groupColumnDefs = useMemo(
    () =>
      [colorColumn, visibilityColumn, opacityColumn].filter(
        (columnDef) => columnDef !== undefined,
      ),
    [colorColumn, visibilityColumn, opacityColumn],
  );

  return (
    <AnnotationsWidget
      tableHeight={200}
      table={tableId}
      selectedGroupByColumn={groupTable.column}
      onSelectedGroupByColumnChange={groupTable.setColumn}
      extraGroupColumnDefs={groupColumnDefs}
      className={className}
    />
  );
}
