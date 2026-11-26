import { EyeIcon, PaletteIcon } from "lucide-react";
import type { ReactNode } from "react";

import {
  type ColorConfig,
  type OpacityConfig,
  type Shapes,
  type VisibilityConfig,
  defaultShapeFillColor,
  defaultShapeStrokeColor,
  shapesDefaults,
} from "@tissuumaps/core";

import { useColorConfigWidget } from "@/components/widgets/ColorConfigWidget/hooks";
import { SettingsWidget } from "@/components/widgets/SettingsWidget";
import { ColorRangeRows } from "@/components/widgets/SettingsWidget/ColorRangeRows";
import { OpacityIcon } from "@/components/widgets/SettingsWidget/OpacityIcon";
import {
  findStyleBy,
  followColumn,
  getFollowState,
  hide,
  isHiddenByConstant,
  isSameColumnStyleBy,
  readStyleBy,
  showIfHidden,
  withColorColumn,
  withFromColumn,
  withoutColumn,
} from "@/components/widgets/SettingsWidget/styleBy";
import { projectStore, useProjectStore } from "@/stores/project";

import { ShapesSettingsCategory } from "./category";

export type ShapesSettingsWidgetProps = {
  shapes: Shapes;
  /** The settings below the style, e.g. the advanced settings */
  children?: ReactNode;
  className?: string;
};

/**
 * The always available settings of shapes: the column that their fill and
 * stroke (each with its color, opacity and visibility), opacity and
 * visibility follow
 *
 * If no property follows the column, a newly picked one styles the fill
 * color. Following a column also shows
 * a hidden fill or stroke; the fill stopping to follow hides it again.
 */
export function ShapesSettingsWidget({
  shapes,
  children,
  className,
}: ShapesSettingsWidgetProps) {
  const updateShapes = useProjectStore((state) => state.updateShapes);
  const tableId = shapes.dataSource.table ?? null;
  const fillColorAdapter = useColorConfigWidget(
    shapes.shapeFillColor,
    (newColorConfig) =>
      updateShapes(shapes.id, { shapeFillColor: newColorConfig }),
    defaultShapeFillColor,
    tableId,
  );
  const strokeColorAdapter = useColorConfigWidget(
    shapes.shapeStrokeColor,
    (newColorConfig) =>
      updateShapes(shapes.id, { shapeStrokeColor: newColorConfig }),
    defaultShapeStrokeColor,
    tableId,
  );

  const styleBys = readStyleBys(shapes);
  const styleBy = findStyleBy(Object.values(styleBys));
  const isOn = (name: keyof typeof styleBys) =>
    isSameColumnStyleBy(styleBys[name], styleBy);

  return (
    <SettingsWidget
      tableId={tableId}
      styleBy={styleBy}
      items={[
        {
          name: "shapeFill",
          label: "Shape fill",
          icon: (
            <FillIcon
              filled={
                isOn(ShapesSettingsCategory.shapeFillColor) ||
                isOn(ShapesSettingsCategory.shapeFillOpacity) ||
                isOn(ShapesSettingsCategory.shapeFillVisibility)
              }
            />
          ),
          properties: [
            {
              name: ShapesSettingsCategory.shapeFillColor,
              label: "Color",
              icon: <PaletteIcon />,
              columnTypes: ["categorical", "numerical"],
              isOn: isOn(ShapesSettingsCategory.shapeFillColor),
              numericalOptions: <ColorRangeRows adapter={fillColorAdapter} />,
            },
            {
              name: ShapesSettingsCategory.shapeFillOpacity,
              label: "Opacity",
              icon: (
                <OpacityIcon
                  filled={isOn(ShapesSettingsCategory.shapeFillOpacity)}
                />
              ),
              columnTypes: ["numerical"],
              isOn: isOn(ShapesSettingsCategory.shapeFillOpacity),
            },
            {
              name: ShapesSettingsCategory.shapeFillVisibility,
              label: "Visibility",
              icon: <EyeIcon />,
              columnTypes: ["numerical"],
              hint: "hides items whose value is 0 or less",
              isOn: isOn(ShapesSettingsCategory.shapeFillVisibility),
            },
          ],
        },
        {
          name: "shapeStroke",
          label: "Shape stroke",
          icon: (
            <StrokeIcon
              filled={
                isOn(ShapesSettingsCategory.shapeStrokeColor) ||
                isOn(ShapesSettingsCategory.shapeStrokeOpacity) ||
                isOn(ShapesSettingsCategory.shapeStrokeVisibility)
              }
            />
          ),
          properties: [
            {
              name: ShapesSettingsCategory.shapeStrokeColor,
              label: "Color",
              icon: <PaletteIcon />,
              columnTypes: ["categorical", "numerical"],
              isOn: isOn(ShapesSettingsCategory.shapeStrokeColor),
              numericalOptions: <ColorRangeRows adapter={strokeColorAdapter} />,
            },
            {
              name: ShapesSettingsCategory.shapeStrokeOpacity,
              label: "Opacity",
              icon: (
                <OpacityIcon
                  filled={isOn(ShapesSettingsCategory.shapeStrokeOpacity)}
                />
              ),
              columnTypes: ["numerical"],
              isOn: isOn(ShapesSettingsCategory.shapeStrokeOpacity),
            },
            {
              name: ShapesSettingsCategory.shapeStrokeVisibility,
              label: "Visibility",
              icon: <EyeIcon />,
              columnTypes: ["numerical"],
              hint: "hides items whose value is 0 or less",
              isOn: isOn(ShapesSettingsCategory.shapeStrokeVisibility),
            },
          ],
        },
        {
          name: ShapesSettingsCategory.shapeOpacity,
          label: "Opacity",
          icon: (
            <OpacityIcon filled={isOn(ShapesSettingsCategory.shapeOpacity)} />
          ),
          columnTypes: ["numerical"],
          isOn: isOn(ShapesSettingsCategory.shapeOpacity),
        },
        {
          name: ShapesSettingsCategory.shapeVisibility,
          label: "Visibility",
          icon: <EyeIcon />,
          columnTypes: ["numerical"],
          hint: "hides items whose value is 0 or less",
          isOn: isOn(ShapesSettingsCategory.shapeVisibility),
        },
      ]}
      onStyleChange={(newStyleBy, propertyNames) => {
        // the shapes as they are now: the change may follow a column load
        const current = projectStore
          .getState()
          .shapes.find(({ id }) => id === shapes.id);
        if (current === undefined) {
          return;
        }
        const currentTableId = current.dataSource.table ?? null;
        const followState = getFollowState(
          readStyleBys(current),
          newStyleBy,
          propertyNames,
        );
        const fillStates = [
          followState(ShapesSettingsCategory.shapeFillColor),
          followState(ShapesSettingsCategory.shapeFillOpacity),
          followState(ShapesSettingsCategory.shapeFillVisibility),
        ];
        const strokeStates = [
          followState(ShapesSettingsCategory.shapeStrokeColor),
          followState(ShapesSettingsCategory.shapeStrokeOpacity),
          followState(ShapesSettingsCategory.shapeStrokeVisibility),
        ];
        const isFillOn = fillStates.some(({ column }) => column !== null);
        const wasFillOn = fillStates.some(({ wasFollowing }) => wasFollowing);
        const isStrokeOn = strokeStates.some(({ column }) => column !== null);

        const fillVisibilityState = followState(
          ShapesSettingsCategory.shapeFillVisibility,
        );
        const fillVisibility = followColumn(
          current.shapeFillVisibility,
          fillVisibilityState,
          {
            withColumn: ({ column }) =>
              withFromColumn(
                current.shapeFillVisibility,
                column,
                currentTableId,
              ),
            withoutColumn: () =>
              withoutColumn<VisibilityConfig>(
                current.shapeFillVisibility,
                shapesDefaults.shapeFillVisibility,
              ),
          },
        );
        const strokeVisibilityState = followState(
          ShapesSettingsCategory.shapeStrokeVisibility,
        );
        const strokeVisibility = followColumn(
          current.shapeStrokeVisibility,
          strokeVisibilityState,
          {
            withColumn: ({ column }) =>
              withFromColumn(
                current.shapeStrokeVisibility,
                column,
                currentTableId,
              ),
            withoutColumn: () =>
              withoutColumn<VisibilityConfig>(
                current.shapeStrokeVisibility,
                shapesDefaults.shapeStrokeVisibility,
              ),
          },
        );

        updateShapes(current.id, {
          shapeFillColor: followColumn(
            current.shapeFillColor,
            followState(ShapesSettingsCategory.shapeFillColor),
            {
              withColumn: ({ column, columnType }) =>
                withColorColumn(
                  current.shapeFillColor,
                  column,
                  columnType,
                  currentTableId,
                ),
              withoutColumn: () =>
                withoutColumn<ColorConfig>(
                  current.shapeFillColor,
                  shapesDefaults.shapeFillColor,
                ),
            },
          ),
          shapeFillOpacity: followColumn(
            current.shapeFillOpacity,
            followState(ShapesSettingsCategory.shapeFillOpacity),
            {
              withColumn: ({ column }) =>
                withFromColumn(
                  current.shapeFillOpacity,
                  column,
                  currentTableId,
                ),
              withoutColumn: () =>
                withoutColumn<OpacityConfig>(
                  current.shapeFillOpacity,
                  shapesDefaults.shapeFillOpacity,
                ),
            },
          ),
          // a fill that follows the column is shown; one that stops is hidden,
          // the mirror of following a column, which shows a hidden fill
          shapeFillVisibility:
            fillVisibilityState.column !== null
              ? fillVisibility
              : isFillOn
                ? showIfHidden(fillVisibility)
                : wasFillOn
                  ? hide(fillVisibility)
                  : fillVisibility,
          shapeStrokeColor: followColumn(
            current.shapeStrokeColor,
            followState(ShapesSettingsCategory.shapeStrokeColor),
            {
              withColumn: ({ column, columnType }) =>
                withColorColumn(
                  current.shapeStrokeColor,
                  column,
                  columnType,
                  currentTableId,
                ),
              withoutColumn: () =>
                withoutColumn<ColorConfig>(
                  current.shapeStrokeColor,
                  shapesDefaults.shapeStrokeColor,
                ),
            },
          ),
          shapeStrokeOpacity: followColumn(
            current.shapeStrokeOpacity,
            followState(ShapesSettingsCategory.shapeStrokeOpacity),
            {
              withColumn: ({ column }) =>
                withFromColumn(
                  current.shapeStrokeOpacity,
                  column,
                  currentTableId,
                ),
              withoutColumn: () =>
                withoutColumn<OpacityConfig>(
                  current.shapeStrokeOpacity,
                  shapesDefaults.shapeStrokeOpacity,
                ),
            },
          ),
          // a stroke that follows the column is shown
          shapeStrokeVisibility:
            strokeVisibilityState.column === null && isStrokeOn
              ? showIfHidden(strokeVisibility)
              : strokeVisibility,
          shapeOpacity: followColumn(
            current.shapeOpacity,
            followState(ShapesSettingsCategory.shapeOpacity),
            {
              withColumn: ({ column }) =>
                withFromColumn(current.shapeOpacity, column, currentTableId),
              withoutColumn: () =>
                withoutColumn<OpacityConfig>(
                  current.shapeOpacity,
                  shapesDefaults.shapeOpacity,
                ),
            },
          ),
          shapeVisibility: followColumn(
            current.shapeVisibility,
            followState(ShapesSettingsCategory.shapeVisibility),
            {
              withColumn: ({ column }) =>
                withFromColumn(current.shapeVisibility, column, currentTableId),
              withoutColumn: () =>
                withoutColumn<VisibilityConfig>(
                  current.shapeVisibility,
                  shapesDefaults.shapeVisibility,
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
 * @param shapes - The shapes
 * @returns What the properties of the shapes follow, by name, in the order of
 * the settings widget; the color and opacity of a hidden fill or stroke follow
 * none
 */
function readStyleBys(shapes: Shapes) {
  const tableId = shapes.dataSource.table ?? null;
  const isFillHidden = isHiddenByConstant(shapes.shapeFillVisibility);
  const isStrokeHidden = isHiddenByConstant(shapes.shapeStrokeVisibility);
  return {
    [ShapesSettingsCategory.shapeFillColor]: readStyleBy(
      shapes.shapeFillColor,
      tableId,
      {
        isHidden: isFillHidden,
      },
    ),
    [ShapesSettingsCategory.shapeFillOpacity]: readStyleBy(
      shapes.shapeFillOpacity,
      tableId,
      {
        isHidden: isFillHidden,
      },
    ),
    [ShapesSettingsCategory.shapeFillVisibility]: readStyleBy(
      shapes.shapeFillVisibility,
      tableId,
    ),
    [ShapesSettingsCategory.shapeStrokeColor]: readStyleBy(
      shapes.shapeStrokeColor,
      tableId,
      {
        isHidden: isStrokeHidden,
      },
    ),
    [ShapesSettingsCategory.shapeStrokeOpacity]: readStyleBy(
      shapes.shapeStrokeOpacity,
      tableId,
      {
        isHidden: isStrokeHidden,
      },
    ),
    [ShapesSettingsCategory.shapeStrokeVisibility]: readStyleBy(
      shapes.shapeStrokeVisibility,
      tableId,
    ),
    [ShapesSettingsCategory.shapeOpacity]: readStyleBy(
      shapes.shapeOpacity,
      tableId,
    ),
    [ShapesSettingsCategory.shapeVisibility]: readStyleBy(
      shapes.shapeVisibility,
      tableId,
    ),
  };
}

type FillIconProps = { filled: boolean };

/** A filled square, colored while the fill follows the column */
function FillIcon({ filled }: FillIconProps) {
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
      <rect
        x="4"
        y="4"
        width="16"
        height="16"
        rx="3"
        className={filled ? "fill-blue-500/70" : "fill-current/25"}
      />
    </svg>
  );
}

type StrokeIconProps = { filled: boolean };

/** A bold outline, colored while the stroke follows the column */
function StrokeIcon({ filled }: StrokeIconProps) {
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
      <rect
        x="4.5"
        y="4.5"
        width="15"
        height="15"
        rx="3"
        strokeWidth={3}
        className={filled ? "stroke-blue-500" : undefined}
      />
    </svg>
  );
}
