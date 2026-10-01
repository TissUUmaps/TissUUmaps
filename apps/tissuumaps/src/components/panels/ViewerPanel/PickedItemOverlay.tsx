import { useMemo } from "react";

import {
  ArrayUtils,
  type ColorConfig,
  ConfigUtils,
  type TableColumnRef,
  getActiveConfigSource,
  isFromConfig,
} from "@tissuumaps/core";
import { ViewerOverlay } from "@tissuumaps/react";

import { formatTableColumn } from "@/components/controls/TableColumnField/formatTableColumn";
import {
  usePointsData,
  useShapesData,
  useTableData,
} from "@/data/hooks/useData";
import { useItemValue } from "@/data/hooks/useItemValue";
import { numberFormat } from "@/lib/format";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

/**
 * Returns the table column a color configuration takes its colors from
 *
 * @param config - The color configuration
 * @returns The group-by or from column of the active source, or `undefined`
 * if the active source uses no column
 */
function getColorColumn(config: ColorConfig): TableColumnRef | undefined {
  return (
    ConfigUtils.getGroupByColumn(config) ??
    (getActiveConfigSource(config) === "from" && isFromConfig(config)
      ? config.from
      : undefined)
  );
}

/**
 * Formats the value of an item in a table column
 *
 * @param value - The value
 * @returns The value as text; non-integer numbers are rounded to six
 * significant digits, everything else is shown as it is
 */
function formatValue(value: unknown): string {
  return typeof value === "number" && !Number.isInteger(value)
    ? numberFormat.format(value)
    : String(value);
}

export type PickedItemOverlayProps = {
  className?: string;
};

/**
 * Shows the item picked in the viewer at the picked position: its object, its
 * ID and name, and its value in the column it is colored by
 */
export function PickedItemOverlay({ className }: PickedItemOverlayProps) {
  const pickedItem = useAppStore((state) => state.pickedItem);
  const annotatedObject = pickedItem?.annotatedObject;
  const labelsId =
    annotatedObject !== undefined && "labelsId" in annotatedObject
      ? annotatedObject.labelsId
      : null;
  const pointsId =
    annotatedObject !== undefined && "pointsId" in annotatedObject
      ? annotatedObject.pointsId
      : null;
  const shapesId =
    annotatedObject !== undefined && "shapesId" in annotatedObject
      ? annotatedObject.shapesId
      : null;

  const labels = useProjectStore(
    (state) => state.labels.find((labels) => labels.id === labelsId) ?? null,
  );
  const points = useProjectStore(
    (state) => state.points.find((points) => points.id === pointsId) ?? null,
  );
  const shapes = useProjectStore(
    (state) => state.shapes.find((shapes) => shapes.id === shapesId) ?? null,
  );
  const tables = useProjectStore((state) => state.tables);
  const pointsData = usePointsData(pointsId);
  const shapesData = useShapesData(shapesId);

  const object = labels ?? points ?? shapes;
  const tableId = object?.dataSource.table ?? null;
  const tableData = useTableData(tableId);
  const colorColumn =
    labels !== null
      ? getColorColumn(labels.labelColor)
      : points !== null
        ? getColorColumn(points.pointColor)
        : shapes !== null
          ? (getColorColumn(shapes.shapeFillColor) ??
            getColorColumn(shapes.shapeStrokeColor))
          : undefined;
  const itemId = pickedItem?.itemId ?? null;
  const itemValue = useItemValue(
    colorColumn !== undefined ? (colorColumn.table ?? tableId) : null,
    colorColumn?.column ?? null,
    itemId,
  );

  // the object's table governs the names, as in the annotations table
  const namedData = tableId !== null ? tableData : (pointsData ?? shapesData);
  const itemName = useMemo(() => {
    const itemNames = namedData?.getNames?.();
    if (namedData === null || itemNames === undefined || itemId === null) {
      return undefined;
    }
    return itemNames[ArrayUtils.indexOfId(namedData.getIds(), itemId)];
  }, [namedData, itemId]);

  if (pickedItem === null || object === null) {
    return null;
  }
  return (
    <ViewerOverlay worldPosition={pickedItem.worldPosition}>
      <div
        role="status"
        className={cn(
          "m-2 max-w-64 rounded-md border bg-popover px-2 py-1 text-xs text-popover-foreground shadow-md",
          className,
        )}
      >
        <div className="font-medium">{object.name}</div>
        <div className="text-muted-foreground">
          ID {pickedItem.itemId}
          {itemName !== undefined && ` (${itemName})`}
        </div>
        {colorColumn !== undefined && itemValue !== null && (
          <div>
            {formatTableColumn(colorColumn, tables)}:{" "}
            {formatValue(itemValue.value)}
          </div>
        )}
      </div>
    </ViewerOverlay>
  );
}
