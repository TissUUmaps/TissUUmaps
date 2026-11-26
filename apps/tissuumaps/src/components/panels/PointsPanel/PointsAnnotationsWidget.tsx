import { useMemo } from "react";

import {
  type ItemsData,
  type Points,
  defaultPointColor,
  defaultPointMarker,
  defaultPointOpacity,
  defaultPointSize,
  defaultPointVisibility,
} from "@tissuumaps/core";

import { AnnotationsWidget } from "@/components/widgets/AnnotationsWidget";
import { useColorGroupValues } from "@/components/widgets/AnnotationsWidget/useColorGroupValues";
import { useGroupColumn } from "@/components/widgets/AnnotationsWidget/useGroupColumn";
import { useGroupTable } from "@/components/widgets/AnnotationsWidget/useGroupTable";
import { useGroupVisibility } from "@/components/widgets/AnnotationsWidget/useGroupVisibility";
import { useMarkerGroupValues } from "@/components/widgets/AnnotationsWidget/useMarkerGroupValues";
import { useOpacityGroupValues } from "@/components/widgets/AnnotationsWidget/useOpacityGroupValues";
import { useSizeGroupValues } from "@/components/widgets/AnnotationsWidget/useSizeGroupValues";
import { useVisibilityGroupValues } from "@/components/widgets/AnnotationsWidget/useVisibilityGroupValues";
import { useProjectStore } from "@/stores/project";

export type PointsAnnotationsWidgetProps = {
  points: Points;
  data: ItemsData;
  className?: string;
};

export function PointsAnnotationsWidget({
  points,
  data,
  className,
}: PointsAnnotationsWidgetProps) {
  const updatePoints = useProjectStore((state) => state.updatePoints);

  const tableId = points.dataSource.table ?? null;
  const annotatedObject = useMemo(() => ({ pointsId: points.id }), [points.id]);
  const groupTable = useGroupTable(points.name, tableId, [
    points.pointMarker,
    points.pointSize,
    points.pointColor,
    points.pointVisibility,
    points.pointOpacity,
  ]);

  const markerAdapter = useMarkerGroupValues();
  const sizeAdapter = useSizeGroupValues();
  const colorAdapter = useColorGroupValues();
  const visibilityAdapter = useVisibilityGroupValues();
  const opacityAdapter = useOpacityGroupValues();

  const groupVisibility = useGroupVisibility(groupTable, {
    name: "visibility",
    default: defaultPointVisibility,
    config: points.pointVisibility,
    onConfigChange: (pointVisibility) =>
      updatePoints(points.id, { pointVisibility }),
    adapter: visibilityAdapter,
  });
  const markerColumn = useGroupColumn(groupTable, {
    name: "marker",
    isShownByDefault: true,
    default: defaultPointMarker,
    config: points.pointMarker,
    onConfigChange: (pointMarker) => updatePoints(points.id, { pointMarker }),
    adapter: markerAdapter,
  });
  const sizeColumn = useGroupColumn(groupTable, {
    name: "size",
    default: defaultPointSize,
    config: points.pointSize,
    onConfigChange: (pointSize) => updatePoints(points.id, { pointSize }),
    adapter: sizeAdapter,
  });
  const colorColumn = useGroupColumn(groupTable, {
    name: "color",
    isShownByDefault: true,
    default: defaultPointColor,
    config: points.pointColor,
    onConfigChange: (pointColor) => updatePoints(points.id, { pointColor }),
    adapter: colorAdapter,
  });
  const opacityColumn = useGroupColumn(groupTable, {
    name: "opacity",
    default: defaultPointOpacity,
    config: points.pointOpacity,
    onConfigChange: (pointOpacity) => updatePoints(points.id, { pointOpacity }),
    adapter: opacityAdapter,
  });

  const groupColumnDefs = useMemo(
    () =>
      [markerColumn, sizeColumn, colorColumn, opacityColumn].filter(
        (columnDef) => columnDef !== undefined,
      ),
    [markerColumn, sizeColumn, colorColumn, opacityColumn],
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
