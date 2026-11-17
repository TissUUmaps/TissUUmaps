import {
  type Points,
  defaultPointColor,
  pointsDefaults,
} from "@tissuumaps/core";

import { Field, FieldLabel } from "@/components/common/field";
import { Switch } from "@/components/ui/switch";
import { useColorConfigWidget } from "@/components/widgets/ColorConfigWidget/hooks";
import { StyleWidget } from "@/components/widgets/StyleWidget";
import {
  isMarkerByCategory,
  markerByCategory,
  markerFollowingColor,
  markerWithoutCategories,
  readColorBy,
  withColorColumn,
  withoutColorColumn,
} from "@/components/widgets/StyleWidget/colorBy";
import { cn } from "@/lib/utils";
import { projectStore, useProjectStore } from "@/stores/project";

export type PointsStyleWidgetProps = {
  points: Points;
  className?: string;
};

/**
 * The always available settings of points: what they are colored by and, for
 * categories, whether their shapes tell them apart
 *
 * "Shapes by category" is off by default; once switched on, the marker
 * follows the column colored by categories (see `markerFollowingColor`).
 */
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
        const currentTableId = current.dataSource.table ?? null;
        updatePoints(current.id, {
          pointColor:
            newColorBy.kind === "column"
              ? withColorColumn(
                  current.pointColor,
                  newColorBy.column,
                  newColorBy.encoding,
                  currentTableId,
                )
              : withoutColorColumn(
                  current.pointColor,
                  pointsDefaults.pointColor,
                ),
          pointMarker: markerFollowingColor(
            current.pointMarker,
            current.pointColor,
            newColorBy,
            pointsDefaults.pointMarker,
            currentTableId,
          ),
        });
      }}
      colorAdapter={colorAdapter}
      categoryOptions={
        colorBy.kind === "column" && (
          <MarkerByCategoryField
            points={points}
            column={colorBy.column}
            className="col-span-2"
          />
        )
      }
      className={className}
    />
  );
}

type MarkerByCategoryFieldProps = {
  points: Points;
  /** The column colored by categories */
  column: string;
  className?: string;
};

/** The "Shapes by category" switch, which groups the marker by the column */
function MarkerByCategoryField({
  points,
  column,
  className,
}: MarkerByCategoryFieldProps) {
  const updatePoints = useProjectStore((state) => state.updatePoints);
  const tableId = points.dataSource.table ?? null;
  const isOn = isMarkerByCategory(points.pointMarker, column, tableId);
  return (
    <Field
      className={cn("flex min-w-0 flex-row items-center gap-x-2", className)}
    >
      <Switch
        checked={isOn}
        onCheckedChange={(checked) =>
          updatePoints(points.id, {
            pointMarker: checked
              ? markerByCategory(
                  points.pointMarker,
                  column,
                  pointsDefaults.pointMarker,
                  tableId,
                )
              : markerWithoutCategories(
                  points.pointMarker,
                  pointsDefaults.pointMarker,
                ),
          })
        }
      />
      <FieldLabel className="truncate text-sm font-normal">
        Shapes by category
      </FieldLabel>
    </Field>
  );
}
