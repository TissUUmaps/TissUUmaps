import { type Points, pointsDefaults } from "@tissuumaps/core";

import { StyleWidget } from "@/components/widgets/StyleWidget";
import {
  colorByColumn,
  colorByNone,
  readColorBy,
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
              ? colorByColumn(
                  current.pointColor,
                  newColorBy.column,
                  newColorBy.encoding,
                  current.dataSource.table ?? null,
                )
              : colorByNone(current.pointColor, pointsDefaults.pointColor),
        });
      }}
      colorConfig={points.pointColor}
      onColorConfigChange={(pointColor) =>
        updatePoints(points.id, { pointColor })
      }
      className={className}
    />
  );
}
