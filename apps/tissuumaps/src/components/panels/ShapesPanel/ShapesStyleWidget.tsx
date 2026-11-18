import {
  type Shapes,
  defaultShapeFillColor,
  shapesDefaults,
} from "@tissuumaps/core";

import { useColorConfigWidget } from "@/components/widgets/ColorConfigWidget/hooks";
import { StyleWidget } from "@/components/widgets/StyleWidget";
import {
  readColorBy,
  withColorColumn,
  withoutColorColumn,
} from "@/components/widgets/StyleWidget/colorBy";
import { projectStore, useProjectStore } from "@/stores/project";

export type ShapesStyleWidgetProps = {
  shapes: Shapes;
  className?: string;
};

/** The always available settings of shapes: what their fill is colored by */
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
      colorBy={readColorBy(shapes.shapeFillColor, tableId)}
      onColorByChange={(colorBy) => {
        // the shapes as they are now: the change may follow a column load
        const current = projectStore
          .getState()
          .shapes.find(({ id }) => id === shapes.id);
        if (current === undefined) {
          return;
        }
        updateShapes(current.id, {
          shapeFillColor:
            colorBy.kind === "column"
              ? withColorColumn(
                  current.shapeFillColor,
                  colorBy.column,
                  colorBy.encoding,
                  current.dataSource.table ?? null,
                )
              : withoutColorColumn(
                  current.shapeFillColor,
                  shapesDefaults.shapeFillColor,
                ),
        });
      }}
      colorAdapter={colorAdapter}
      className={className}
    />
  );
}
