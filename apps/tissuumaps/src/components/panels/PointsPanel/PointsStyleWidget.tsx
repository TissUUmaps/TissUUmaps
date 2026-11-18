import {
  type Points,
  defaultPointColor,
  pointsDefaults,
} from "@tissuumaps/core";

import { useColorConfigWidget } from "@/components/widgets/ColorConfigWidget/hooks";
import { StyleWidget } from "@/components/widgets/StyleWidget";
import {
  readColorBy,
  withColorColumn,
  withoutColorColumn,
} from "@/components/widgets/StyleWidget/colorBy";
import { projectStore, useProjectStore } from "@/stores/project";

export type PointsStyleWidgetProps = {
  points: Points;
  className?: string;
};

/** The always available settings of points: what they are colored by */
export function PointsStyleWidget({
  points,
  className,
}: PointsStyleWidgetProps) {
  const updatePoints = useProjectStore((state) => state.updatePoints);
  const tableId = points.dataSource.table ?? null;
  const colorAdapter = useColorConfigWidget(
    points.pointColor,
    (newColorConfig) => updatePoints(points.id, { pointColor: newColorConfig }),
    defaultPointColor,
    tableId,
  );
  const colorBy = readColorBy(points.pointColor, tableId);

  return (
    <StyleWidget
      tableId={tableId}
      colorBy={colorBy}
      onColorByChange={(newColorBy) => {
        // the points as they are now: the change may follow a column load
        const current = projectStore
          .getState()
          .points.find(({ id }) => id === points.id);
        if (current === undefined) {
          return;
        }
        updatePoints(current.id, {
          pointColor:
            newColorBy.kind === "column"
              ? withColorColumn(
                  current.pointColor,
                  newColorBy.column,
                  newColorBy.encoding,
                  current.dataSource.table ?? null,
                )
              : withoutColorColumn(
                  current.pointColor,
                  pointsDefaults.pointColor,
                ),
        });
      }}
      colorAdapter={colorAdapter}
      className={className}
    />
  );
}
