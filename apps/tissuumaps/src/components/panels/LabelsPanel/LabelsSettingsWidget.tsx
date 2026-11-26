import { EyeIcon, PaletteIcon } from "lucide-react";

import {
  type Labels,
  type TableColumnType,
  defaultLabelColor,
  labelsDefaults,
} from "@tissuumaps/core";

import { useColorConfigWidget } from "@/components/widgets/ColorConfigWidget/hooks";
import { SettingsWidget } from "@/components/widgets/SettingsWidget";
import { CategoricalColorPaletteRow } from "@/components/widgets/SettingsWidget/CategoricalColorPaletteRow";
import { ColorRangeRows } from "@/components/widgets/SettingsWidget/ColorRangeRows";
import { OpacityIcon } from "@/components/widgets/SettingsWidget/OpacityIcon";
import {
  describeColorWithoutColumn,
  findFollowedColumn,
  getFollower,
  isSameFollowedColumn,
  readFollowedColumn,
  withColorColumn,
  withFromColumn,
  withOpacityColumn,
} from "@/components/widgets/SettingsWidget/follow";
import { projectStore, useProjectStore } from "@/stores/project";

import { LabelsAdvancedSettingsWidget } from "./LabelsAdvancedSettingsWidget";
import { LabelsSettingsCategory } from "./category";

/** The column types each property can follow */
const columnTypes = {
  [LabelsSettingsCategory.labelColor]: ["categorical", "numerical"],
  [LabelsSettingsCategory.labelOpacity]: ["numerical"],
  [LabelsSettingsCategory.labelVisibility]: ["numerical"],
} satisfies Record<string, TableColumnType[]>;

export type LabelsSettingsWidgetProps = {
  labels: Labels;
  className?: string;
};

/**
 * The always available settings of labels: the column that their color,
 * opacity and visibility follow
 *
 * If no property follows the column, the color follows a newly picked one.
 * The color stopping to follow returns to the default random colors of labels.
 */
export function LabelsSettingsWidget({
  labels,
  className,
}: LabelsSettingsWidgetProps) {
  const updateLabels = useProjectStore((state) => state.updateLabels);
  const tableId = labels.dataSource.table ?? null;
  const colorAdapter = useColorConfigWidget(
    labels.labelColor,
    (newColorConfig) => updateLabels(labels.id, { labelColor: newColorConfig }),
    defaultLabelColor,
    tableId,
  );

  const followedColumns = readFollowedColumns(labels);
  const followedColumn = findFollowedColumn(Object.values(followedColumns));
  const isOn = (name: keyof typeof followedColumns) =>
    isSameFollowedColumn(followedColumns[name], followedColumn);

  return (
    <SettingsWidget
      tableId={tableId}
      followedColumn={followedColumn}
      placeholder={describeColorWithoutColumn(
        labels.labelColor,
        labelsDefaults.labelColor,
      )}
      items={[
        {
          name: LabelsSettingsCategory.labelColor,
          label: "Color",
          icon: <PaletteIcon />,
          columnTypes: columnTypes[LabelsSettingsCategory.labelColor],
          isOn: isOn(LabelsSettingsCategory.labelColor),
          categoricalOptions: (
            <CategoricalColorPaletteRow adapter={colorAdapter} />
          ),
          numericalOptions: <ColorRangeRows adapter={colorAdapter} />,
        },
        {
          name: LabelsSettingsCategory.labelOpacity,
          label: "Opacity",
          icon: (
            <OpacityIcon filled={isOn(LabelsSettingsCategory.labelOpacity)} />
          ),
          columnTypes: columnTypes[LabelsSettingsCategory.labelOpacity],
          isOn: isOn(LabelsSettingsCategory.labelOpacity),
        },
        {
          name: LabelsSettingsCategory.labelVisibility,
          label: "Visibility",
          icon: <EyeIcon />,
          columnTypes: columnTypes[LabelsSettingsCategory.labelVisibility],
          hint: "hides items whose value is 0 or less",
          isOn: isOn(LabelsSettingsCategory.labelVisibility),
        },
      ]}
      onFollowedColumnChange={(newFollowedColumn, propertyNames) => {
        // the labels as they are now: the change may come after a column load
        const current = projectStore
          .getState()
          .labels.find(({ id }) => id === labels.id);
        if (current === undefined) {
          return;
        }
        const currentTableId = current.dataSource.table ?? null;
        const follow = getFollower(
          readFollowedColumns(current),
          newFollowedColumn,
          propertyNames,
        );
        updateLabels(current.id, {
          labelColor: follow(
            LabelsSettingsCategory.labelColor,
            current.labelColor,
            labelsDefaults.labelColor,
            ({ column, columnType }) =>
              withColorColumn(
                current.labelColor,
                column,
                columnType,
                currentTableId,
              ),
          ),
          labelOpacity: follow(
            LabelsSettingsCategory.labelOpacity,
            current.labelOpacity,
            labelsDefaults.labelOpacity,
            ({ column }) =>
              withOpacityColumn(current.labelOpacity, column, currentTableId),
          ),
          labelVisibility: follow(
            LabelsSettingsCategory.labelVisibility,
            current.labelVisibility,
            labelsDefaults.labelVisibility,
            ({ column }) =>
              withFromColumn(current.labelVisibility, column, currentTableId),
          ),
        });
      }}
      advancedSettings={<LabelsAdvancedSettingsWidget labels={labels} />}
      className={className}
    />
  );
}

/**
 * @param labels - The labels
 * @returns What the properties of the labels follow, by name, in the order
 * of the settings widget
 */
function readFollowedColumns(labels: Labels) {
  const tableId = labels.dataSource.table ?? null;
  return {
    [LabelsSettingsCategory.labelColor]: readFollowedColumn(
      labels.labelColor,
      tableId,
      columnTypes[LabelsSettingsCategory.labelColor],
    ),
    [LabelsSettingsCategory.labelOpacity]: readFollowedColumn(
      labels.labelOpacity,
      tableId,
      columnTypes[LabelsSettingsCategory.labelOpacity],
    ),
    [LabelsSettingsCategory.labelVisibility]: readFollowedColumn(
      labels.labelVisibility,
      tableId,
      columnTypes[LabelsSettingsCategory.labelVisibility],
    ),
  };
}
