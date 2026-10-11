import {
  type ColorConfig,
  type Layer,
  type TableColumnRef,
  type VisibilityConfig,
  findColorPalette,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
} from "@tissuumaps/core";

import { formatTableColumn } from "@/components/controls/TableColumnField/formatTableColumn";
import { useValueRange } from "@/data/hooks/useValueRange";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

/** Formats the ends of a colorbar's value range */
const colorbarValueFormat = new Intl.NumberFormat(undefined, {
  maximumSignificantDigits: 6,
});

// The color settings keep the inactive sources as drafts, so a "from" source
// only gets a colorbar while it is the active one.
function getActiveFrom(colorConfig: ColorConfig) {
  return getActiveConfigSource(colorConfig) === "from" &&
    isFromConfig(colorConfig)
    ? colorConfig.from
    : null;
}

// Only a constant visibility is checked, as a colorbar is still useful for the
// items that a table column leaves visible.
function isConstantlyHidden(visibilityConfig: VisibilityConfig): boolean {
  return (
    getActiveConfigSource(visibilityConfig) === "constant" &&
    isConstantConfig(visibilityConfig) &&
    !visibilityConfig.constant.value
  );
}

// Only a layer ID is checked: with a layer per item, a colorbar is still useful
// for the items on the visible layers.
function isOnHiddenLayer(
  layer: string | TableColumnRef,
  layers: Layer[],
): boolean {
  return (
    typeof layer === "string" &&
    layers.find(({ id }) => id === layer)?.visibility === false
  );
}

type ColorbarProps = {
  name: string;
  colorConfig: ColorConfig;
  tableId: string | null;
  className?: string;
};

function Colorbar({ name, colorConfig, tableId, className }: ColorbarProps) {
  const tables = useProjectStore((state) => state.tables);

  const from = getActiveFrom(colorConfig);

  // like in the renderers, a configured range takes precedence over the
  // column's value range
  const columnValueRange = useValueRange(
    from !== null && from.range === undefined ? (from.table ?? tableId) : null,
    from?.column ?? null,
  );

  const valueRange = from?.range ?? columnValueRange;
  const colorPalette = findColorPalette(from?.palette);
  if (from === null || valueRange === null || colorPalette === undefined) {
    return null;
  }

  const columnLabel = formatTableColumn(from, tables);
  // every palette color is a stop of its own, so that the gradient blends the
  // colors linearly like the renderers do
  const gradientStops = colorPalette.colors.map(
    ({ r, g, b }, i) =>
      `rgb(${r}, ${g}, ${b}) ${(100 * i) / (colorPalette.colors.length - 1)}%`,
  );

  return (
    <div className={className}>
      <div className="flex justify-between gap-2">
        <span title={name} className="max-w-1/2 truncate font-medium">
          {name}
        </span>
        <span title={columnLabel} className="max-w-1/2 truncate">
          {columnLabel}
        </span>
      </div>
      <div
        className="mt-0.5 h-2 rounded-xs border border-input bg-origin-border"
        style={{
          backgroundImage: `linear-gradient(to right, ${gradientStops.join(", ")})`,
        }}
      />
      <div className="flex justify-between tabular-nums">
        <span>{colorbarValueFormat.format(valueRange[0])}</span>
        <span>{colorbarValueFormat.format(valueRange[1])}</span>
      </div>
    </div>
  );
}

export type ColorbarViewerControlProps = { className?: string };

export function ColorbarViewerControl({
  className,
}: ColorbarViewerControlProps) {
  const layers = useProjectStore((state) => state.layers);
  const labels = useProjectStore((state) => state.labels);
  const points = useProjectStore((state) => state.points);
  const shapes = useProjectStore((state) => state.shapes);
  const colorbarsVisible = useProjectStore((state) => state.colorbarsVisible);

  const colorbars = [
    ...labels
      .filter(
        (labels) => labels.visibility && !isOnHiddenLayer(labels.layer, layers),
      )
      .map((labels) => ({
        key: `labels-${labels.id}`,
        name: labels.name,
        colorConfig: labels.labelColor,
        visibilityConfig: labels.labelVisibility,
        tableId: labels.dataSource.table ?? null,
      })),
    ...points
      .filter(
        (points) => points.visibility && !isOnHiddenLayer(points.layer, layers),
      )
      .map((points) => ({
        key: `points-${points.id}`,
        name: points.name,
        colorConfig: points.pointColor,
        visibilityConfig: points.pointVisibility,
        tableId: points.dataSource.table ?? null,
      })),
    ...shapes
      .filter(
        (shapes) =>
          shapes.visibility &&
          !isOnHiddenLayer(shapes.layer, layers) &&
          !isConstantlyHidden(shapes.shapeVisibility),
      )
      .flatMap((shapes) => [
        {
          key: `shapes-${shapes.id}-fill`,
          name: `${shapes.name} (fill)`,
          colorConfig: shapes.shapeFillColor,
          visibilityConfig: shapes.shapeFillVisibility,
          tableId: shapes.dataSource.table ?? null,
        },
        {
          key: `shapes-${shapes.id}-stroke`,
          name: `${shapes.name} (stroke)`,
          colorConfig: shapes.shapeStrokeColor,
          visibilityConfig: shapes.shapeStrokeVisibility,
          tableId: shapes.dataSource.table ?? null,
        },
      ]),
  ].filter(
    ({ colorConfig, visibilityConfig }) =>
      getActiveFrom(colorConfig) !== null &&
      !isConstantlyHidden(visibilityConfig),
  );

  if (!colorbarsVisible || colorbars.length === 0) {
    return null;
  }

  // the card hides while none of its colorbars can be drawn, e.g. while their
  // value ranges load
  return (
    <div
      className={cn(
        "mr-2 mb-2 flex w-80 empty:hidden flex-col gap-1.5 rounded-lg border border-border bg-background/75 px-2 py-1 text-xs shadow-lg",
        className,
      )}
    >
      {colorbars.map(({ key, name, colorConfig, tableId }) => (
        <Colorbar
          key={key}
          name={name}
          colorConfig={colorConfig}
          tableId={tableId}
        />
      ))}
    </div>
  );
}
