import { useMemo } from "react";

import {
  type ItemsData,
  type Shapes,
  defaultShapeFillColor,
  defaultShapeFillOpacity,
  defaultShapeFillVisibility,
  defaultShapeStrokeColor,
  defaultShapeStrokeOpacity,
  defaultShapeStrokeVisibility,
} from "@tissuumaps/core";

import { AnnotationsWidget } from "@/components/widgets/AnnotationsWidget";
import { useColorGroupValues } from "@/components/widgets/AnnotationsWidget/useColorGroupValues";
import { useGroupColumn } from "@/components/widgets/AnnotationsWidget/useGroupColumn";
import { useGroupTable } from "@/components/widgets/AnnotationsWidget/useGroupTable";
import { useOpacityGroupValues } from "@/components/widgets/AnnotationsWidget/useOpacityGroupValues";
import { useVisibilityGroupValues } from "@/components/widgets/AnnotationsWidget/useVisibilityGroupValues";

import { ShapesSettingsCategory } from "./category";

export type ShapesAnnotationsWidgetProps = {
  shapes: Shapes;
  data: ItemsData;
  activeSettingsCategory: ShapesSettingsCategory | null;
  className?: string;
};

export function ShapesAnnotationsWidget({
  shapes,
  data,
  activeSettingsCategory,
  className,
}: ShapesAnnotationsWidgetProps) {
  const tableId = shapes.dataSource.table ?? null;
  const groupTable = useGroupTable(
    tableId,
    [
      {
        category: ShapesSettingsCategory.shapeFillColor,
        config: shapes.shapeFillColor,
      },
      {
        category: ShapesSettingsCategory.shapeFillVisibility,
        config: shapes.shapeFillVisibility,
      },
      {
        category: ShapesSettingsCategory.shapeFillOpacity,
        config: shapes.shapeFillOpacity,
      },
      {
        category: ShapesSettingsCategory.shapeStrokeColor,
        config: shapes.shapeStrokeColor,
      },
      {
        category: ShapesSettingsCategory.shapeStrokeVisibility,
        config: shapes.shapeStrokeVisibility,
      },
      {
        category: ShapesSettingsCategory.shapeStrokeOpacity,
        config: shapes.shapeStrokeOpacity,
      },
    ],
    activeSettingsCategory,
  );

  const colorAdapter = useColorGroupValues();
  const visibilityAdapter = useVisibilityGroupValues();
  const opacityAdapter = useOpacityGroupValues();

  const fillColorColumn = useGroupColumn(groupTable, {
    name: "fill color",
    default: defaultShapeFillColor,
    config: shapes.shapeFillColor,
    adapter: colorAdapter,
  });
  const fillVisibilityColumn = useGroupColumn(groupTable, {
    name: "fill visibility",
    default: defaultShapeFillVisibility,
    config: shapes.shapeFillVisibility,
    adapter: visibilityAdapter,
  });
  const fillOpacityColumn = useGroupColumn(groupTable, {
    name: "fill opacity",
    default: defaultShapeFillOpacity,
    config: shapes.shapeFillOpacity,
    adapter: opacityAdapter,
  });
  const strokeColorColumn = useGroupColumn(groupTable, {
    name: "outline color",
    default: defaultShapeStrokeColor,
    config: shapes.shapeStrokeColor,
    adapter: colorAdapter,
  });
  const strokeVisibilityColumn = useGroupColumn(groupTable, {
    name: "outline visibility",
    default: defaultShapeStrokeVisibility,
    config: shapes.shapeStrokeVisibility,
    adapter: visibilityAdapter,
  });
  const strokeOpacityColumn = useGroupColumn(groupTable, {
    name: "outline opacity",
    default: defaultShapeStrokeOpacity,
    config: shapes.shapeStrokeOpacity,
    adapter: opacityAdapter,
  });

  const groupColumnDefs = useMemo(
    () =>
      [
        fillColorColumn,
        fillVisibilityColumn,
        fillOpacityColumn,
        strokeColorColumn,
        strokeVisibilityColumn,
        strokeOpacityColumn,
      ].filter((columnDef) => columnDef !== undefined),
    [
      fillColorColumn,
      fillVisibilityColumn,
      fillOpacityColumn,
      strokeColorColumn,
      strokeVisibilityColumn,
      strokeOpacityColumn,
    ],
  );

  return (
    <AnnotationsWidget
      data={data}
      tableHeight={200}
      table={tableId}
      selectedGroupByColumn={groupTable.column}
      onSelectedGroupByColumnChange={groupTable.setColumn}
      extraGroupColumnDefs={groupColumnDefs}
      className={className}
    />
  );
}
