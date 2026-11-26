import { EyeIcon, PaletteIcon, ShapesIcon } from "lucide-react";
import type { ReactNode } from "react";

import {
  type Points,
  type TableColumnType,
  defaultPointColor,
  defaultPointSize,
  defaultPointSizeUnit,
  pointsDefaults,
} from "@tissuumaps/core";

import { useColorConfigWidget } from "@/components/widgets/ColorConfigWidget/hooks";
import { SettingsWidget } from "@/components/widgets/SettingsWidget";
import { CategoricalColorPaletteRow } from "@/components/widgets/SettingsWidget/CategoricalColorPaletteRow";
import { ColorRangeRows } from "@/components/widgets/SettingsWidget/ColorRangeRows";
import { OpacityIcon } from "@/components/widgets/SettingsWidget/OpacityIcon";
import { StyleRow } from "@/components/widgets/SettingsWidget/StyleRow";
import {
  describeColorWithoutColumn,
  findStyleBy,
  getFollower,
  isSameColumnStyleBy,
  readStyleBy,
  withColorColumn,
  withFromColumn,
  withMarkerColumn,
  withOpacityColumn,
} from "@/components/widgets/SettingsWidget/styleBy";
import { SizeUnitRadioGroup } from "@/components/widgets/SizeConfigWidget";
import { useSizeConfigWidget } from "@/components/widgets/SizeConfigWidget/hooks";
import { projectStore, useProjectStore } from "@/stores/project";

import { PointsSettingsCategory } from "./category";

/** The column types each property can follow */
const columnTypes = {
  [PointsSettingsCategory.pointColor]: ["categorical", "numerical"],
  [PointsSettingsCategory.pointMarker]: ["categorical"],
  [PointsSettingsCategory.pointSize]: ["numerical"],
  [PointsSettingsCategory.pointOpacity]: ["numerical"],
  [PointsSettingsCategory.pointVisibility]: ["numerical"],
} satisfies Record<string, TableColumnType[]>;

export type PointsSettingsWidgetProps = {
  points: Points;
  /** The settings below the style, e.g. the advanced settings */
  children?: ReactNode;
  className?: string;
};

/**
 * The always available settings of points: the column that their color,
 * marker, size, opacity and visibility follow
 *
 * If no property follows the column, a newly picked one styles the color.
 */
export function PointsSettingsWidget({
  points,
  children,
  className,
}: PointsSettingsWidgetProps) {
  const updatePoints = useProjectStore((state) => state.updatePoints);
  const tableId = points.dataSource.table ?? null;
  const colorAdapter = useColorConfigWidget(
    points.pointColor,
    (newColorConfig) => updatePoints(points.id, { pointColor: newColorConfig }),
    defaultPointColor,
    tableId,
  );
  const sizeAdapter = useSizeConfigWidget(
    points.pointSize,
    (newSizeConfig) => updatePoints(points.id, { pointSize: newSizeConfig }),
    defaultPointSize,
    defaultPointSizeUnit,
    tableId,
  );

  const styleBys = readStyleBys(points);
  const styleBy = findStyleBy(Object.values(styleBys));
  const isOn = (name: keyof typeof styleBys) =>
    isSameColumnStyleBy(styleBys[name], styleBy);

  return (
    <SettingsWidget
      tableId={tableId}
      styleBy={styleBy}
      placeholder={describeColorWithoutColumn(
        points.pointColor,
        pointsDefaults.pointColor,
      )}
      items={[
        {
          name: PointsSettingsCategory.pointColor,
          label: "Color",
          icon: <PaletteIcon />,
          columnTypes: columnTypes[PointsSettingsCategory.pointColor],
          isOn: isOn(PointsSettingsCategory.pointColor),
          categoricalOptions: (
            <CategoricalColorPaletteRow adapter={colorAdapter} />
          ),
          numericalOptions: <ColorRangeRows adapter={colorAdapter} />,
        },
        {
          name: PointsSettingsCategory.pointMarker,
          label: "Marker",
          icon: <ShapesIcon />,
          columnTypes: columnTypes[PointsSettingsCategory.pointMarker],
          isOn: isOn(PointsSettingsCategory.pointMarker),
        },
        {
          name: PointsSettingsCategory.pointSize,
          label: "Size",
          icon: <SizeIcon filled={isOn(PointsSettingsCategory.pointSize)} />,
          columnTypes: columnTypes[PointsSettingsCategory.pointSize],
          isOn: isOn(PointsSettingsCategory.pointSize),
          numericalOptions: (
            <StyleRow label="Unit">
              <SizeUnitRadioGroup
                value={sizeAdapter.currentFromUnit}
                onValueChange={sizeAdapter.setCurrentFromUnit}
                className="text-xs"
              />
            </StyleRow>
          ),
        },
        {
          name: PointsSettingsCategory.pointOpacity,
          label: "Opacity",
          icon: (
            <OpacityIcon filled={isOn(PointsSettingsCategory.pointOpacity)} />
          ),
          columnTypes: columnTypes[PointsSettingsCategory.pointOpacity],
          isOn: isOn(PointsSettingsCategory.pointOpacity),
        },
        {
          name: PointsSettingsCategory.pointVisibility,
          label: "Visibility",
          icon: <EyeIcon />,
          columnTypes: columnTypes[PointsSettingsCategory.pointVisibility],
          hint: "hides items whose value is 0 or less",
          isOn: isOn(PointsSettingsCategory.pointVisibility),
        },
      ]}
      onStyleChange={(newStyleBy, propertyNames) => {
        // the points as they are now: the change may follow a column load
        const current = projectStore
          .getState()
          .points.find(({ id }) => id === points.id);
        if (current === undefined) {
          return;
        }
        const currentTableId = current.dataSource.table ?? null;
        const follow = getFollower(
          readStyleBys(current),
          newStyleBy,
          propertyNames,
        );
        updatePoints(current.id, {
          pointColor: follow(
            PointsSettingsCategory.pointColor,
            current.pointColor,
            pointsDefaults.pointColor,
            ({ column, columnType }) =>
              withColorColumn(
                current.pointColor,
                column,
                columnType,
                currentTableId,
              ),
          ),
          pointMarker: follow(
            PointsSettingsCategory.pointMarker,
            current.pointMarker,
            pointsDefaults.pointMarker,
            ({ column }) =>
              withMarkerColumn(current.pointMarker, column, currentTableId),
          ),
          pointSize: follow(
            PointsSettingsCategory.pointSize,
            current.pointSize,
            pointsDefaults.pointSize,
            ({ column }) =>
              withFromColumn(current.pointSize, column, currentTableId),
          ),
          pointOpacity: follow(
            PointsSettingsCategory.pointOpacity,
            current.pointOpacity,
            pointsDefaults.pointOpacity,
            ({ column }) =>
              withOpacityColumn(current.pointOpacity, column, currentTableId),
          ),
          pointVisibility: follow(
            PointsSettingsCategory.pointVisibility,
            current.pointVisibility,
            pointsDefaults.pointVisibility,
            ({ column }) =>
              withFromColumn(current.pointVisibility, column, currentTableId),
          ),
        });
      }}
      className={className}
    >
      {children}
    </SettingsWidget>
  );
}

/**
 * @param points - The points
 * @returns What the properties of the points follow, by name, in the order
 * of the settings widget
 */
function readStyleBys(points: Points) {
  const tableId = points.dataSource.table ?? null;
  return {
    [PointsSettingsCategory.pointColor]: readStyleBy(
      points.pointColor,
      tableId,
      columnTypes[PointsSettingsCategory.pointColor],
    ),
    [PointsSettingsCategory.pointMarker]: readStyleBy(
      points.pointMarker,
      tableId,
      columnTypes[PointsSettingsCategory.pointMarker],
    ),
    [PointsSettingsCategory.pointSize]: readStyleBy(
      points.pointSize,
      tableId,
      columnTypes[PointsSettingsCategory.pointSize],
    ),
    [PointsSettingsCategory.pointOpacity]: readStyleBy(
      points.pointOpacity,
      tableId,
      columnTypes[PointsSettingsCategory.pointOpacity],
    ),
    [PointsSettingsCategory.pointVisibility]: readStyleBy(
      points.pointVisibility,
      tableId,
      columnTypes[PointsSettingsCategory.pointVisibility],
    ),
  };
}

type SizeIconProps = { filled: boolean };

/** A small and a large point, filled while sizes follow the column */
function SizeIcon({ filled }: SizeIconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <circle
        cx="6.5"
        cy="17.5"
        r="2.5"
        className={filled ? "fill-current" : undefined}
      />
      <circle
        cx="15"
        cy="9"
        r="6"
        className={filled ? "fill-current" : undefined}
      />
    </svg>
  );
}
