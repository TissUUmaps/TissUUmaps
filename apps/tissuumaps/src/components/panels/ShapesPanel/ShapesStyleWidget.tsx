import { type Shapes, shapesDefaults } from "@tissuumaps/core";

import { StyleWidget } from "@/components/widgets/StyleWidget";
import {
  colorByColumn,
  colorByNone,
  readColorBy,
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
              ? colorByColumn(
                  current.shapeFillColor,
                  colorBy.column,
                  colorBy.encoding,
                  current.dataSource.table ?? null,
                )
              : colorByNone(
                  current.shapeFillColor,
                  shapesDefaults.shapeFillColor,
                ),
        });
      }}
      colorConfig={shapes.shapeFillColor}
      onColorConfigChange={(shapeFillColor) =>
        updateShapes(shapes.id, { shapeFillColor })
      }
      className={className}
    />
  );
}
