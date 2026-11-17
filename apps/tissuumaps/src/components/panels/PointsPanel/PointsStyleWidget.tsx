import { EyeIcon, PaletteIcon, ShapesIcon } from "lucide-react";
import { type ReactNode, useState } from "react";

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

import { SimpleSelect } from "@/components/common/simple-select";
import { TableColumnInput } from "@/components/controls/TableColumnField/TableColumnInput";
import { useColorConfigWidget } from "@/components/widgets/ColorConfigWidget/hooks";
import { SizeUnitRadioGroup } from "@/components/widgets/SizeConfigWidget";
import { useSizeConfigWidget } from "@/components/widgets/SizeConfigWidget/hooks";
import { StyleWidget } from "@/components/widgets/StyleWidget";
import { ColorRangeRows } from "@/components/widgets/StyleWidget/ColorRangeRows";
import { OpacityIcon } from "@/components/widgets/StyleWidget/OpacityIcon";
import { StyleRow } from "@/components/widgets/StyleWidget/StyleRow";
import {
  findStyleBy,
  followColumn,
  getFollowState,
  isSameColumnStyleBy,
  readStyleBy,
  withColorColumn,
  withFromColumn,
  withMarkerColumn,
  withoutColumn,
} from "@/components/widgets/StyleWidget/styleBy";
import { useTableData } from "@/data/hooks/useData";
import { useAppStore } from "@/stores/app";
import { projectStore, useProjectStore } from "@/stores/project";

import { PointsSettingsCategory } from "./category";
import { getCoordinateProperties } from "./getCoordinateProperties";

export type PointsStyleWidgetProps = {
  points: Points;
  /** The settings below the style, e.g. the advanced settings */
  children?: ReactNode;
  className?: string;
};

/**
 * The always available settings of points: their position, and the column
 * that their color, marker, size, opacity and visibility follow
 *
 * If no property follows the column, a newly picked one styles the color.
 */
export function PointsStyleWidget({
  points,
  children,
  className,
}: PointsStyleWidgetProps) {
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
    <StyleWidget
      tableId={tableId}
      styleBy={styleBy}
      items={[
        {
          name: PointsSettingsCategory.pointColor,
          label: "Color",
          icon: <PaletteIcon />,
          columnTypes: ["categorical", "numerical"],
          isOn: isOn(PointsSettingsCategory.pointColor),
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
      leadingRows={<PositionField points={points} />}
      className={className}
    >
      {children}
    </StyleWidget>
  );
}

/**
 * @param points - The points
 * @returns What the properties of the points follow, by name, in the order
 * of the style widget
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

/**
 * @param pair - A coordinate column pair
 * @returns The key of the pair, by its columns rather than its label, which is
 * for display only
 */
function getPairKey(pair: { x: string; y: string }): string {
  return JSON.stringify([pair.x, pair.y]);
}

type PositionFieldProps = {
  points: Points;
  className?: string;
};

/**
 * The coordinate columns of points, for data sources that declare them (see
 * `getCoordinateProperties`), as the data provider normalizes them, so that
 * the columns in use are shown even where the data source leaves them unset
 *
 * Coordinate pairs recorded by the table, such as the embeddings of an
 * AnnData object, are offered in a select; "Custom…" picks the two columns
 * separately.
 */
function PositionField({ points, className }: PositionFieldProps) {
  const pointsDataProviders = useAppStore((state) => state.pointsDataProviders);
  const workspace = useAppStore((state) => state.workspace);
  const projectSource = useProjectStore((state) => state.source);
  const updatePoints = useProjectStore((state) => state.updatePoints);

  const dataProvider = pointsDataProviders.get(points.dataSource.type);
  const coordinateProperties =
    dataProvider !== undefined
      ? getCoordinateProperties(dataProvider.uischema)
      : null;

  const tableId = points.dataSource.table ?? null;
  const tableData = useTableData(tableId);
  const pairs = tableData?.getCoordinateColumns?.() ?? [];
  const [isCustom, setCustom] = useState(false);

  if (dataProvider === undefined || coordinateProperties === null) {
    return null;
  }

  let normalizedDataSource: Record<string, unknown>;
  try {
    normalizedDataSource = {
      ...dataProvider.normalize(points.dataSource, workspace, projectSource),
    };
  } catch {
    normalizedDataSource = { ...points.dataSource };
  }
  const readCoordinate = (property: string) => {
    const value = normalizedDataSource[property];
    return typeof value === "string" ? value : undefined;
  };
  const x = readCoordinate(coordinateProperties.x);
  const y = readCoordinate(coordinateProperties.y);
  const setCoordinates = (newX: string | undefined, newY: string | undefined) =>
    updatePoints(points.id, {
      dataSource: {
        ...points.dataSource,
        [coordinateProperties.x]: newX,
        [coordinateProperties.y]: newY,
      },
    });

  const selectedPair = pairs.find((pair) => pair.x === x && pair.y === y);

  const columnInputs = (
    <div className="grid min-w-0 grid-cols-2 gap-x-1.5">
      <TableColumnInput
        tableId={tableId}
        value={x ?? null}
        onValueChange={(newX) => setCoordinates(newX ?? undefined, y)}
        startAddon="X"
        placeholder="Default"
      />
      <TableColumnInput
        tableId={tableId}
        value={y ?? null}
        onValueChange={(newY) => setCoordinates(x, newY ?? undefined)}
        startAddon="Y"
        placeholder="Default"
      />
    </div>
  );

  if (pairs.length === 0) {
    return (
      <StyleRow label="Position" className={className}>
        {columnInputs}
      </StyleRow>
    );
  }
  // the unselected value (null) stands for coordinates of no recorded pair
  return (
    <>
      <StyleRow label="Position" className={className}>
        <SimpleSelect
          items={pairs}
          itemLabel={(pair) => pair.name}
          itemValue={getPairKey}
          nullable
          placeholder="Custom…"
          value={!isCustom && selectedPair ? getPairKey(selectedPair) : null}
          onValueChange={(key) => {
            const pair = pairs.find((pair) => getPairKey(pair) === key);
            setCustom(pair === undefined);
            if (pair !== undefined) {
              setCoordinates(pair.x, pair.y);
            }
          }}
          variant="compact"
          triggerLabel="Position"
        />
      </StyleRow>
      {(isCustom || !selectedPair) && (
        <StyleRow className={className}>{columnInputs}</StyleRow>
      )}
    </>
  );
}
