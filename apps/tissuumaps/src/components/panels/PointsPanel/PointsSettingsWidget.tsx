import { EyeIcon, PaletteIcon, ShapesIcon } from "lucide-react";
import type { ReactNode } from "react";

import {
  type ColorConfig,
  type MarkerConfig,
  type OpacityConfig,
  type Points,
  type SizeConfig,
  type VisibilityConfig,
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
  followColumn,
  getFollowState,
  isSameColumnStyleBy,
  readStyleBy,
  withColorColumn,
  withFromColumn,
  withMarkerColumn,
  withoutColumn,
} from "@/components/widgets/SettingsWidget/styleBy";
import { SizeUnitRadioGroup } from "@/components/widgets/SizeConfigWidget";
import { useSizeConfigWidget } from "@/components/widgets/SizeConfigWidget/hooks";
import { projectStore, useProjectStore } from "@/stores/project";

import { PointsSettingsCategory } from "./category";

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
          columnTypes: ["categorical", "numerical"],
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
          columnTypes: ["categorical"],
          isOn: isOn(PointsSettingsCategory.pointMarker),
        },
        {
          name: PointsSettingsCategory.pointSize,
          label: "Size",
          icon: <SizeIcon filled={isOn(PointsSettingsCategory.pointSize)} />,
          columnTypes: ["numerical"],
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
          columnTypes: ["numerical"],
          isOn: isOn(PointsSettingsCategory.pointOpacity),
        },
        {
          name: PointsSettingsCategory.pointVisibility,
          label: "Visibility",
          icon: <EyeIcon />,
          columnTypes: ["numerical"],
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
        const followState = getFollowState(
          readStyleBys(current),
          newStyleBy,
          propertyNames,
        );
        updatePoints(current.id, {
          pointColor: followColumn(
            current.pointColor,
            followState(PointsSettingsCategory.pointColor),
            {
              withColumn: ({ column, columnType }) =>
                withColorColumn(
                  current.pointColor,
                  column,
                  columnType,
                  currentTableId,
                ),
              withoutColumn: () =>
                withoutColumn<ColorConfig>(
                  current.pointColor,
                  pointsDefaults.pointColor,
                ),
            },
          ),
          pointMarker: followColumn(
            current.pointMarker,
            followState(PointsSettingsCategory.pointMarker),
            {
              withColumn: ({ column }) =>
                withMarkerColumn(current.pointMarker, column, currentTableId),
              withoutColumn: () =>
                withoutColumn<MarkerConfig>(
                  current.pointMarker,
                  pointsDefaults.pointMarker,
                ),
            },
          ),
          pointSize: followColumn(
            current.pointSize,
            followState(PointsSettingsCategory.pointSize),
            {
              withColumn: ({ column }) =>
                withFromColumn(current.pointSize, column, currentTableId),
              withoutColumn: () =>
                withoutColumn<SizeConfig>(
                  current.pointSize,
                  pointsDefaults.pointSize,
                ),
            },
          ),
          pointOpacity: followColumn(
            current.pointOpacity,
            followState(PointsSettingsCategory.pointOpacity),
            {
              withColumn: ({ column }) =>
                withFromColumn(current.pointOpacity, column, currentTableId),
              withoutColumn: () =>
                withoutColumn<OpacityConfig>(
                  current.pointOpacity,
                  pointsDefaults.pointOpacity,
                ),
            },
          ),
          pointVisibility: followColumn(
            current.pointVisibility,
            followState(PointsSettingsCategory.pointVisibility),
            {
              withColumn: ({ column }) =>
                withFromColumn(current.pointVisibility, column, currentTableId),
              withoutColumn: () =>
                withoutColumn<VisibilityConfig>(
                  current.pointVisibility,
                  pointsDefaults.pointVisibility,
                ),
            },
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
    ),
    [PointsSettingsCategory.pointMarker]: readStyleBy(
      points.pointMarker,
      tableId,
    ),
    [PointsSettingsCategory.pointSize]: readStyleBy(points.pointSize, tableId),
    [PointsSettingsCategory.pointOpacity]: readStyleBy(
      points.pointOpacity,
      tableId,
    ),
    [PointsSettingsCategory.pointVisibility]: readStyleBy(
      points.pointVisibility,
      tableId,
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
