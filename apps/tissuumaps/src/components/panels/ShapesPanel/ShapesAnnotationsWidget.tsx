import { useMemo } from "react";

import {
  type ItemsData,
  type Shapes,
  defaultShapeFillColor,
  defaultShapeFillOpacity,
  defaultShapeFillVisibility,
  defaultShapeOpacity,
  defaultShapeStrokeColor,
  defaultShapeStrokeOpacity,
  defaultShapeStrokeVisibility,
  defaultShapeVisibility,
} from "@tissuumaps/core";

import { AnnotationsWidget } from "@/components/widgets/AnnotationsWidget";
import { useColorGroupValues } from "@/components/widgets/AnnotationsWidget/useColorGroupValues";
import { useGroupColumn } from "@/components/widgets/AnnotationsWidget/useGroupColumn";
import { useGroupTable } from "@/components/widgets/AnnotationsWidget/useGroupTable";
import { useGroupVisibility } from "@/components/widgets/AnnotationsWidget/useGroupVisibility";
import { useOpacityGroupValues } from "@/components/widgets/AnnotationsWidget/useOpacityGroupValues";
import { useVisibilityGroupValues } from "@/components/widgets/AnnotationsWidget/useVisibilityGroupValues";
import { useProjectStore } from "@/stores/project";

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
  const updateShapes = useProjectStore((state) => state.updateShapes);

  const tableId = shapes.dataSource.table ?? null;
  const annotatedObject = useMemo(() => ({ shapesId: shapes.id }), [shapes.id]);
  const groupTable = useGroupTable(
    shapes.name,
    tableId,
    // in the order of the style section, whose first property that groups by
    // a column the group table follows while no settings category is open
    [
      {
        category: ShapesSettingsCategory.shapeFillColor,
        config: shapes.shapeFillColor,
      },
      {
        category: ShapesSettingsCategory.shapeFillOpacity,
        config: shapes.shapeFillOpacity,
      },
      {
        category: ShapesSettingsCategory.shapeFillVisibility,
        config: shapes.shapeFillVisibility,
      },
      {
        category: ShapesSettingsCategory.shapeStrokeColor,
        config: shapes.shapeStrokeColor,
      },
      {
        category: ShapesSettingsCategory.shapeStrokeOpacity,
        config: shapes.shapeStrokeOpacity,
      },
      {
        category: ShapesSettingsCategory.shapeStrokeVisibility,
        config: shapes.shapeStrokeVisibility,
      },
      {
        category: ShapesSettingsCategory.shapeOpacity,
        config: shapes.shapeOpacity,
      },
      {
        category: ShapesSettingsCategory.shapeVisibility,
        config: shapes.shapeVisibility,
      },
    ],
    activeSettingsCategory,
  );

  const colorAdapter = useColorGroupValues();
  const visibilityAdapter = useVisibilityGroupValues();
  const opacityAdapter = useOpacityGroupValues();

  const groupVisibility = useGroupVisibility(groupTable, {
    name: "visibility",
    default: defaultShapeVisibility,
    config: shapes.shapeVisibility,
    onConfigChange: (shapeVisibility) =>
      updateShapes(shapes.id, { shapeVisibility }),
    adapter: visibilityAdapter,
  });
  const opacityColumn = useGroupColumn(groupTable, {
    name: "opacity",
    default: defaultShapeOpacity,
    config: shapes.shapeOpacity,
    onConfigChange: (shapeOpacity) => updateShapes(shapes.id, { shapeOpacity }),
    adapter: opacityAdapter,
  });
  const fillColorColumn = useGroupColumn(groupTable, {
    name: "fill color",
    isShownByDefault: true,
    default: defaultShapeFillColor,
    config: shapes.shapeFillColor,
    onConfigChange: (shapeFillColor) =>
      updateShapes(shapes.id, { shapeFillColor }),
    adapter: colorAdapter,
  });
  const fillVisibilityColumn = useGroupColumn(groupTable, {
    name: "fill visibility",
    default: defaultShapeFillVisibility,
    config: shapes.shapeFillVisibility,
    onConfigChange: (shapeFillVisibility) =>
      updateShapes(shapes.id, { shapeFillVisibility }),
    adapter: visibilityAdapter,
  });
  const fillOpacityColumn = useGroupColumn(groupTable, {
    name: "fill opacity",
    default: defaultShapeFillOpacity,
    config: shapes.shapeFillOpacity,
    onConfigChange: (shapeFillOpacity) =>
      updateShapes(shapes.id, { shapeFillOpacity }),
    adapter: opacityAdapter,
  });
  const strokeColorColumn = useGroupColumn(groupTable, {
    name: "stroke color",
    default: defaultShapeStrokeColor,
    config: shapes.shapeStrokeColor,
    onConfigChange: (shapeStrokeColor) =>
      updateShapes(shapes.id, { shapeStrokeColor }),
    adapter: colorAdapter,
  });
  const strokeVisibilityColumn = useGroupColumn(groupTable, {
    name: "stroke visibility",
    default: defaultShapeStrokeVisibility,
    config: shapes.shapeStrokeVisibility,
    onConfigChange: (shapeStrokeVisibility) =>
      updateShapes(shapes.id, { shapeStrokeVisibility }),
    adapter: visibilityAdapter,
  });
  const strokeOpacityColumn = useGroupColumn(groupTable, {
    name: "stroke opacity",
    default: defaultShapeStrokeOpacity,
    config: shapes.shapeStrokeOpacity,
    onConfigChange: (shapeStrokeOpacity) =>
      updateShapes(shapes.id, { shapeStrokeOpacity }),
    adapter: opacityAdapter,
  });

  const groupColumnDefs = useMemo(
    () =>
      [
        opacityColumn,
        fillColorColumn,
        fillVisibilityColumn,
        fillOpacityColumn,
        strokeColorColumn,
        strokeVisibilityColumn,
        strokeOpacityColumn,
      ].filter((columnDef) => columnDef !== undefined),
    [
      opacityColumn,
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
      tableHeight={300}
      tableId={tableId}
      annotatedObject={annotatedObject}
      selectedGroupBy={groupTable.groupBy}
      onSelectedGroupByChange={groupTable.setGroupBy}
      groupCounts={groupTable.groupCounts}
      groupVisibility={groupVisibility}
      groupColumnDefs={groupColumnDefs}
      className={className}
    />
  );
}
