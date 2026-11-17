import {
  type Shapes,
  defaultShapeFillColor,
  shapesDefaults,
} from "@tissuumaps/core";

import { useColorConfigWidget } from "@/components/widgets/ColorConfigWidget/hooks";
import { StyleWidget } from "@/components/widgets/StyleWidget";
import {
  hide,
  isHiddenByConstant,
  readColorBy,
  showIfHidden,
  withColorColumn,
  withoutColorColumn,
} from "@/components/widgets/StyleWidget/colorBy";
import { projectStore, useProjectStore } from "@/stores/project";

export type ShapesStyleWidgetProps = {
  shapes: Shapes;
  className?: string;
};

/**
 * The always available settings of shapes: what their fill is colored by
 *
 * Coloring by a column also shows a hidden fill, and coloring by none hides
 * it again.
 */
export function ShapesStyleWidget({
  shapes,
  className,
}: ShapesStyleWidgetProps) {
  const updateShapes = useProjectStore((state) => state.updateShapes);
  const tableId = shapes.dataSource.table ?? null;
  const colorAdapter = useColorConfigWidget(
    shapes.shapeFillColor,
    (newColorConfig) =>
      updateShapes(shapes.id, { shapeFillColor: newColorConfig }),
    defaultShapeFillColor,
    tableId,
  );

  return (
    <StyleWidget
      tableId={tableId}
      colorBy={readColorBy(shapes.shapeFillColor, tableId, {
        isHidden: isHiddenByConstant(shapes.shapeFillVisibility),
      })}
      onColorByChange={(colorBy) => {
        // the shapes as they are now: the change may follow a column load
        const current = projectStore
          .getState()
          .shapes.find(({ id }) => id === shapes.id);
        if (current === undefined) {
          return;
        }
        updateShapes(
          current.id,
          colorBy.kind === "column"
            ? {
                shapeFillColor: withColorColumn(
                  current.shapeFillColor,
                  colorBy.column,
                  colorBy.encoding,
                  current.dataSource.table ?? null,
                ),
                shapeFillVisibility: showIfHidden(current.shapeFillVisibility),
              }
            : {
                shapeFillColor: withoutColorColumn(
                  current.shapeFillColor,
                  shapesDefaults.shapeFillColor,
                ),
                // the mirror of picking a column, which shows a hidden fill
                shapeFillVisibility: hide(current.shapeFillVisibility),
              },
        );
      }}
      colorAdapter={colorAdapter}
      className={className}
    />
  );
}
