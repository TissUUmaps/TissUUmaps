import { EyeIcon, PaletteIcon } from "lucide-react";
import type { ReactNode } from "react";

import {
  type ColorConfig,
  type Labels,
  type OpacityConfig,
  type VisibilityConfig,
  defaultLabelColor,
  labelsDefaults,
} from "@tissuumaps/core";

import { useColorConfigWidget } from "@/components/widgets/ColorConfigWidget/hooks";
import { StyleWidget } from "@/components/widgets/StyleWidget";
import { ColorRangeRows } from "@/components/widgets/StyleWidget/ColorRangeRows";
import { OpacityIcon } from "@/components/widgets/StyleWidget/OpacityIcon";
import {
  findStyleBy,
  followColumn,
  getFollowState,
  isSameColumnStyleBy,
  readStyleBy,
  withColorColumn,
  withFromColumn,
  withoutColumn,
} from "@/components/widgets/StyleWidget/styleBy";
import { projectStore, useProjectStore } from "@/stores/project";

import { LabelsSettingsCategory } from "./category";

export type LabelsStyleWidgetProps = {
  labels: Labels;
  /** The settings below the style, e.g. the advanced settings */
  children?: ReactNode;
  className?: string;
};

/**
 * The always available settings of labels: the column that their color,
 * opacity and visibility follow
 *
 * If no property follows the column, a newly picked one styles the color.
 * The color stopping to follow
 * returns to the default random colors of labels.
 */
export function LabelsStyleWidget({
  labels,
  children,
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

  const styleBys = readStyleBys(labels);
  const styleBy = findStyleBy(Object.values(styleBys));
  const isOn = (name: keyof typeof styleBys) =>
    isSameColumnStyleBy(styleBys[name], styleBy);

  return (
    <StyleWidget
      tableId={tableId}
      styleBy={styleBy}
      items={[
        {
          name: LabelsSettingsCategory.labelColor,
          label: "Color",
          icon: <PaletteIcon />,
          columnTypes: ["categorical", "numerical"],
          isOn: isOn(LabelsSettingsCategory.labelColor),
          numericalOptions: <ColorRangeRows adapter={colorAdapter} />,
        },
        {
          name: LabelsSettingsCategory.labelOpacity,
          label: "Opacity",
          icon: (
            <OpacityIcon filled={isOn(LabelsSettingsCategory.labelOpacity)} />
          ),
          columnTypes: ["numerical"],
          isOn: isOn(LabelsSettingsCategory.labelOpacity),
        },
        {
          name: LabelsSettingsCategory.labelVisibility,
          label: "Visibility",
          icon: <EyeIcon />,
          columnTypes: ["numerical"],
          hint: "hides items whose value is 0 or less",
          isOn: isOn(LabelsSettingsCategory.labelVisibility),
        },
      ]}
      onStyleChange={(newStyleBy, propertyNames) => {
        // the labels as they are now: the change may follow a column load
        const current = projectStore
          .getState()
          .labels.find(({ id }) => id === labels.id);
        if (current === undefined) {
          return;
        }
        const currentTableId = current.dataSource.table ?? null;
        const followState = getFollowState(
          readStyleBys(current),
          newStyleBy,
          propertyNames,
        );
        updateLabels(current.id, {
          labelColor: followColumn(
            current.labelColor,
            followState(LabelsSettingsCategory.labelColor),
            {
              withColumn: ({ column, columnType }) =>
                withColorColumn(
                  current.labelColor,
                  column,
                  columnType,
                  currentTableId,
                ),
              withoutColumn: () =>
                withoutColumn<ColorConfig>(
                  current.labelColor,
                  labelsDefaults.labelColor,
                ),
            },
          ),
          labelOpacity: followColumn(
            current.labelOpacity,
            followState(LabelsSettingsCategory.labelOpacity),
            {
              withColumn: ({ column }) =>
                withFromColumn(current.labelOpacity, column, currentTableId),
              withoutColumn: () =>
                withoutColumn<OpacityConfig>(
                  current.labelOpacity,
                  labelsDefaults.labelOpacity,
                ),
            },
          ),
          labelVisibility: followColumn(
            current.labelVisibility,
            followState(LabelsSettingsCategory.labelVisibility),
            {
              withColumn: ({ column }) =>
                withFromColumn(current.labelVisibility, column, currentTableId),
              withoutColumn: () =>
                withoutColumn<VisibilityConfig>(
                  current.labelVisibility,
                  labelsDefaults.labelVisibility,
                ),
            },
          ),
        });
      }}
      noneLabel="Random colors"
      className={className}
    >
      {children}
    </StyleWidget>
  );
}

/**
 * @param labels - The labels
 * @returns What the properties of the labels follow, by name, in the order
 * of the style widget
 */
function readStyleBys(labels: Labels) {
  const tableId = labels.dataSource.table ?? null;
  return {
    [LabelsSettingsCategory.labelColor]: readStyleBy(
      labels.labelColor,
      tableId,
      {
        defaultConfig: labelsDefaults.labelColor,
      },
    ),
    [LabelsSettingsCategory.labelOpacity]: readStyleBy(
      labels.labelOpacity,
      tableId,
    ),
    [LabelsSettingsCategory.labelVisibility]: readStyleBy(
      labels.labelVisibility,
      tableId,
    ),
  };
}
