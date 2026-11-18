import {
  type Labels,
  defaultLabelColor,
  labelsDefaults,
} from "@tissuumaps/core";

import { useColorConfigWidget } from "@/components/widgets/ColorConfigWidget/hooks";
import { StyleWidget } from "@/components/widgets/StyleWidget";
import {
  readColorBy,
  withColorColumn,
  withoutColorColumn,
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
  const colorAdapter = useColorConfigWidget(
    labels.labelColor,
    (newColorConfig) => updateLabels(labels.id, { labelColor: newColorConfig }),
    defaultLabelColor,
    tableId,
  );

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
              ? withColorColumn(
                  current.labelColor,
                  colorBy.column,
                  colorBy.encoding,
                  current.dataSource.table ?? null,
                )
              : withoutColorColumn(
                  current.labelColor,
                  labelsDefaults.labelColor,
                ),
        });
      }}
      noneLabel="Random colors"
      colorAdapter={colorAdapter}
      className={className}
    />
  );
}
