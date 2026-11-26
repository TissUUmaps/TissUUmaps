import { EyeIcon, PaletteIcon } from "lucide-react";
import type { ReactNode } from "react";

import {
  type Shapes,
  type TableColumnType,
  defaultShapeFillColor,
  defaultShapeStrokeColor,
  shapesDefaults,
} from "@tissuumaps/core";

import { useColorConfigWidget } from "@/components/widgets/ColorConfigWidget/hooks";
import { SettingsWidget } from "@/components/widgets/SettingsWidget";
import { CategoricalColorPaletteRow } from "@/components/widgets/SettingsWidget/CategoricalColorPaletteRow";
import { ColorRangeRows } from "@/components/widgets/SettingsWidget/ColorRangeRows";
import { OpacityIcon } from "@/components/widgets/SettingsWidget/OpacityIcon";
import {
  describeColorWithoutColumn,
  findStyleBy,
  getFollower,
  isSameColumnStyleBy,
  readStyleBy,
  withColorColumn,
  withFromColumn,
} from "@/components/widgets/SettingsWidget/styleBy";
import { projectStore, useProjectStore } from "@/stores/project";

import { ShapesSettingsCategory } from "./category";

/** The column types each property can follow */
const columnTypes = {
  [ShapesSettingsCategory.shapeFillColor]: ["categorical", "numerical"],
  [ShapesSettingsCategory.shapeFillOpacity]: ["numerical"],
  [ShapesSettingsCategory.shapeFillVisibility]: ["numerical"],
  [ShapesSettingsCategory.shapeStrokeColor]: ["categorical", "numerical"],
  [ShapesSettingsCategory.shapeStrokeOpacity]: ["numerical"],
  [ShapesSettingsCategory.shapeStrokeVisibility]: ["numerical"],
  [ShapesSettingsCategory.shapeOpacity]: ["numerical"],
  [ShapesSettingsCategory.shapeVisibility]: ["numerical"],
} satisfies Record<string, TableColumnType[]>;

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
 * color.
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
      placeholder={describeColorWithoutColumn(
        shapes.shapeFillColor,
        shapesDefaults.shapeFillColor,
      )}
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
              columnTypes: columnTypes[ShapesSettingsCategory.shapeFillColor],
              isOn: isOn(ShapesSettingsCategory.shapeFillColor),
              categoricalOptions: (
                <CategoricalColorPaletteRow adapter={fillColorAdapter} />
              ),
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
              columnTypes: columnTypes[ShapesSettingsCategory.shapeFillOpacity],
              isOn: isOn(ShapesSettingsCategory.shapeFillOpacity),
            },
            {
              name: ShapesSettingsCategory.shapeFillVisibility,
              label: "Visibility",
              icon: <EyeIcon />,
              columnTypes:
                columnTypes[ShapesSettingsCategory.shapeFillVisibility],
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
              columnTypes: columnTypes[ShapesSettingsCategory.shapeStrokeColor],
              isOn: isOn(ShapesSettingsCategory.shapeStrokeColor),
              categoricalOptions: (
                <CategoricalColorPaletteRow adapter={strokeColorAdapter} />
              ),
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
              columnTypes:
                columnTypes[ShapesSettingsCategory.shapeStrokeOpacity],
              isOn: isOn(ShapesSettingsCategory.shapeStrokeOpacity),
            },
            {
              name: ShapesSettingsCategory.shapeStrokeVisibility,
              label: "Visibility",
              icon: <EyeIcon />,
              columnTypes:
                columnTypes[ShapesSettingsCategory.shapeStrokeVisibility],
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
          columnTypes: columnTypes[ShapesSettingsCategory.shapeOpacity],
          isOn: isOn(ShapesSettingsCategory.shapeOpacity),
        },
        {
          name: ShapesSettingsCategory.shapeVisibility,
          label: "Visibility",
          icon: <EyeIcon />,
          columnTypes: columnTypes[ShapesSettingsCategory.shapeVisibility],
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
        const follow = getFollower(
          readStyleBys(current),
          newStyleBy,
          propertyNames,
        );
        updateShapes(current.id, {
          shapeFillColor: follow(
            ShapesSettingsCategory.shapeFillColor,
            current.shapeFillColor,
            shapesDefaults.shapeFillColor,
            ({ column, columnType }) =>
              withColorColumn(
                current.shapeFillColor,
                column,
                columnType,
                currentTableId,
              ),
          ),
          shapeFillOpacity: follow(
            ShapesSettingsCategory.shapeFillOpacity,
            current.shapeFillOpacity,
            shapesDefaults.shapeFillOpacity,
            ({ column }) =>
              withFromColumn(current.shapeFillOpacity, column, currentTableId),
          ),
          shapeFillVisibility: follow(
            ShapesSettingsCategory.shapeFillVisibility,
            current.shapeFillVisibility,
            shapesDefaults.shapeFillVisibility,
            ({ column }) =>
              withFromColumn(
                current.shapeFillVisibility,
                column,
                currentTableId,
              ),
          ),
          shapeStrokeColor: follow(
            ShapesSettingsCategory.shapeStrokeColor,
            current.shapeStrokeColor,
            shapesDefaults.shapeStrokeColor,
            ({ column, columnType }) =>
              withColorColumn(
                current.shapeStrokeColor,
                column,
                columnType,
                currentTableId,
              ),
          ),
          shapeStrokeOpacity: follow(
            ShapesSettingsCategory.shapeStrokeOpacity,
            current.shapeStrokeOpacity,
            shapesDefaults.shapeStrokeOpacity,
            ({ column }) =>
              withFromColumn(
                current.shapeStrokeOpacity,
                column,
                currentTableId,
              ),
          ),
          shapeStrokeVisibility: follow(
            ShapesSettingsCategory.shapeStrokeVisibility,
            current.shapeStrokeVisibility,
            shapesDefaults.shapeStrokeVisibility,
            ({ column }) =>
              withFromColumn(
                current.shapeStrokeVisibility,
                column,
                currentTableId,
              ),
          ),
          shapeOpacity: follow(
            ShapesSettingsCategory.shapeOpacity,
            current.shapeOpacity,
            shapesDefaults.shapeOpacity,
            ({ column }) =>
              withFromColumn(current.shapeOpacity, column, currentTableId),
          ),
          shapeVisibility: follow(
            ShapesSettingsCategory.shapeVisibility,
            current.shapeVisibility,
            shapesDefaults.shapeVisibility,
            ({ column }) =>
              withFromColumn(current.shapeVisibility, column, currentTableId),
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
 * the settings widget
 */
function readStyleBys(shapes: Shapes) {
  const tableId = shapes.dataSource.table ?? null;
  return {
    [ShapesSettingsCategory.shapeFillColor]: readStyleBy(
      shapes.shapeFillColor,
      tableId,
      columnTypes[ShapesSettingsCategory.shapeFillColor],
    ),
    [ShapesSettingsCategory.shapeFillOpacity]: readStyleBy(
      shapes.shapeFillOpacity,
      tableId,
      columnTypes[ShapesSettingsCategory.shapeFillOpacity],
    ),
    [ShapesSettingsCategory.shapeFillVisibility]: readStyleBy(
      shapes.shapeFillVisibility,
      tableId,
      columnTypes[ShapesSettingsCategory.shapeFillVisibility],
    ),
    [ShapesSettingsCategory.shapeStrokeColor]: readStyleBy(
      shapes.shapeStrokeColor,
      tableId,
      columnTypes[ShapesSettingsCategory.shapeStrokeColor],
    ),
    [ShapesSettingsCategory.shapeStrokeOpacity]: readStyleBy(
      shapes.shapeStrokeOpacity,
      tableId,
      columnTypes[ShapesSettingsCategory.shapeStrokeOpacity],
    ),
    [ShapesSettingsCategory.shapeStrokeVisibility]: readStyleBy(
      shapes.shapeStrokeVisibility,
      tableId,
      columnTypes[ShapesSettingsCategory.shapeStrokeVisibility],
    ),
    [ShapesSettingsCategory.shapeOpacity]: readStyleBy(
      shapes.shapeOpacity,
      tableId,
      columnTypes[ShapesSettingsCategory.shapeOpacity],
    ),
    [ShapesSettingsCategory.shapeVisibility]: readStyleBy(
      shapes.shapeVisibility,
      tableId,
      columnTypes[ShapesSettingsCategory.shapeVisibility],
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
