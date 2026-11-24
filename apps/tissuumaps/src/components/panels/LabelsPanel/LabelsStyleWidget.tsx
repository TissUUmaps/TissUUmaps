import { type Labels, labelsDefaults } from "@tissuumaps/core";

import { StyleWidget } from "@/components/widgets/StyleWidget";
import {
  colorByColumn,
  colorByNone,
  readColorBy,
} from "@/components/widgets/StyleWidget/colorBy";
import { projectStore, useProjectStore } from "@/stores/project";

export type LabelsStyleWidgetProps = {
  labels: Labels;
  className?: string;
};

/**
 * The always available settings of labels: what they are colored by
 *
 * Coloring by no column returns to the default random colors of labels.
 */
export function LabelsStyleWidget({
  labels,
  className,
}: LabelsStyleWidgetProps) {
  const updateLabels = useProjectStore((state) => state.updateLabels);
  const tableId = labels.dataSource.table ?? null;

  return (
    <StyleWidget
      tableId={tableId}
      colorBy={readColorBy(labels.labelColor, tableId, {
        noneSource: "random",
      })}
      onColorByChange={(colorBy) => {
        // the labels as they are now: the change may follow a column load
        const current = projectStore
          .getState()
          .labels.find(({ id }) => id === labels.id);
        if (current === undefined) {
          return;
        }
        updateLabels(current.id, {
          labelColor:
            colorBy.kind === "column"
              ? colorByColumn(
                  current.labelColor,
                  colorBy.column,
                  colorBy.encoding,
                  current.dataSource.table ?? null,
                )
              : colorByNone(current.labelColor, labelsDefaults.labelColor),
        });
      }}
      noneLabel="Random colors"
      colorConfig={labels.labelColor}
      onColorConfigChange={(labelColor) =>
        updateLabels(labels.id, { labelColor })
      }
      className={className}
    />
  );
}
