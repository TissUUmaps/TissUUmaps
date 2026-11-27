import { EyeIcon, PaletteIcon, ShapesIcon } from "lucide-react";

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
import { SettingsRow } from "@/components/widgets/SettingsWidget/SettingsRow";
import {
  describeColorWithoutColumn,
  findFollowedColumn,
  getFollower,
  isSameFollowedColumn,
  readFollowedColumn,
  startsFollowingNumerical,
  withColorColumn,
  withFromColumn,
  withMarkerColumn,
  withOpacityColumn,
} from "@/components/widgets/SettingsWidget/follow";
import { SizeUnitRadioGroup } from "@/components/widgets/SizeConfigWidget";
import { useSizeConfigWidget } from "@/components/widgets/SizeConfigWidget/hooks";
import { projectStore, useProjectStore } from "@/stores/project";

import { PointsAdvancedSettingsWidget } from "./PointsAdvancedSettingsWidget";
import { PointsSettingsCategory } from "./category";

/** The column types each property can follow */
const columnTypes = {
  [PointsSettingsCategory.pointColor]: ["categorical", "numerical"],
  [PointsSettingsCategory.pointMarker]: ["categorical"],
  [PointsSettingsCategory.pointSize]: ["numerical"],
  [PointsSettingsCategory.pointOpacity]: ["numerical"],
  [PointsSettingsCategory.pointVisibility]: ["numerical"],
  [PointsSettingsCategory.pointOrder]: ["numerical"],
} satisfies Record<string, TableColumnType[]>;

export type PointsSettingsWidgetProps = {
  points: Points;
  className?: string;
};

/**
 * The always available settings of points: the column that their color,
 * marker, size, opacity, visibility and draw order follow
 *
 * If no property follows the column, the color follows a newly picked one.
 * The draw order starts together with the color on a numerical column, unless
 * an order is set already.
 */
export function PointsSettingsWidget({
  points,
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

  const followedColumns = readFollowedColumns(points);
  const followedColumn = findFollowedColumn(Object.values(followedColumns));
  const isOn = (name: keyof typeof followedColumns) =>
    isSameFollowedColumn(followedColumns[name], followedColumn);

  return (
    <SettingsWidget
      tableId={tableId}
      followedColumn={followedColumn}
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
            <SettingsRow label="Unit">
              <SizeUnitRadioGroup
                value={sizeAdapter.currentFromUnit}
                onValueChange={sizeAdapter.setCurrentFromUnit}
                className="text-xs"
              />
            </SettingsRow>
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
        {
          name: PointsSettingsCategory.pointOrder,
          label: "Order",
          icon: <OrderIcon filled={isOn(PointsSettingsCategory.pointOrder)} />,
          columnTypes: columnTypes[PointsSettingsCategory.pointOrder],
          hint: "draws items with higher values on top",
          isOn: isOn(PointsSettingsCategory.pointOrder),
        },
      ]}
      onFollowedColumnChange={(newFollowedColumn, propertyNames) => {
        // the points as they are now: the change may come after a column load
        const current = projectStore
          .getState()
          .points.find(({ id }) => id === points.id);
        if (current === undefined) {
          return;
        }
        const currentTableId = current.dataSource.table ?? null;
        const previousFollowedColumns = readFollowedColumns(current);
        const follow = getFollower(
          previousFollowedColumns,
          newFollowedColumn,
          // the order starts with the color (higher values are drawn on top),
          // unless an order is set already
          current.pointOrder === undefined &&
            startsFollowingNumerical(
              previousFollowedColumns,
              newFollowedColumn,
              propertyNames,
              PointsSettingsCategory.pointColor,
            )
            ? [...propertyNames, PointsSettingsCategory.pointOrder]
            : propertyNames,
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
          pointOrder: follow(
            PointsSettingsCategory.pointOrder,
            current.pointOrder,
            undefined,
            ({ column }) => ({ source: "from", from: { column } }),
          ),
        });
      }}
      advancedSettings={<PointsAdvancedSettingsWidget points={points} />}
      className={className}
    />
  );
}

/**
 * @param points - The points
 * @returns What the properties of the points follow, by name, in the order
 * of the settings widget
 */
function readFollowedColumns(points: Points) {
  const tableId = points.dataSource.table ?? null;
  return {
    [PointsSettingsCategory.pointColor]: readFollowedColumn(
      points.pointColor,
      tableId,
      columnTypes[PointsSettingsCategory.pointColor],
    ),
    [PointsSettingsCategory.pointMarker]: readFollowedColumn(
      points.pointMarker,
      tableId,
      columnTypes[PointsSettingsCategory.pointMarker],
    ),
    [PointsSettingsCategory.pointSize]: readFollowedColumn(
      points.pointSize,
      tableId,
      columnTypes[PointsSettingsCategory.pointSize],
    ),
    [PointsSettingsCategory.pointOpacity]: readFollowedColumn(
      points.pointOpacity,
      tableId,
      columnTypes[PointsSettingsCategory.pointOpacity],
    ),
    [PointsSettingsCategory.pointVisibility]: readFollowedColumn(
      points.pointVisibility,
      tableId,
      columnTypes[PointsSettingsCategory.pointVisibility],
    ),
    [PointsSettingsCategory.pointOrder]: readFollowedColumn(
      points.pointOrder,
      tableId,
      columnTypes[PointsSettingsCategory.pointOrder],
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

type OrderIconProps = { filled: boolean };

/** Three overlapping points, the top one filled while the draw order follows the column */
function OrderIcon({ filled }: OrderIconProps) {
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
      <path d="M7.08 11.08A5 5 0 1 0 12.92 16.92" />
      <path d="M11.08 7.08A5 5 0 1 0 16.92 12.92" />
      <circle
        cx="16"
        cy="8"
        r="5"
        className={filled ? "fill-current" : undefined}
      />
    </svg>
  );
}
