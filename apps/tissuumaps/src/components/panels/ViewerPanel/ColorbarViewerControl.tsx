import {
  type ColorConfig,
  type FromConfig,
  findColorPalette,
} from "@tissuumaps/core";

import { formatTableColumn } from "@/components/controls/TableColumnField/formatTableColumn";
import { useValueRange } from "@/data/hooks/useValueRange";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

import { getColorbars } from "./getColorbars";

/** Formats the ends of a colorbar's value range */
const colorbarValueFormat = new Intl.NumberFormat(undefined, {
  maximumSignificantDigits: 6,
});

type ColorbarProps = {
  name: string;
  from: Extract<ColorConfig, FromConfig>["from"];
  tableId: string;
  className?: string;
};

function Colorbar({ name, from, tableId, className }: ColorbarProps) {
  const tables = useProjectStore((state) => state.tables);

  // like in the renderers, a configured range end takes precedence over the
  // column's value range
  const [rangeMin, rangeMax] = from.range ?? [null, null];
  const columnValueRange = useValueRange(
    rangeMin === null || rangeMax === null ? tableId : null,
    from.column,
  );

  const colorPalette = findColorPalette(from.palette);
  const valueMin = rangeMin ?? columnValueRange?.[0];
  const valueMax = rangeMax ?? columnValueRange?.[1];
  if (
    valueMin === undefined ||
    valueMax === undefined ||
    colorPalette === undefined
  ) {
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
        <span>{colorbarValueFormat.format(valueMin)}</span>
        <span>{colorbarValueFormat.format(valueMax)}</span>
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

  const colorbars = getColorbars({ layers, labels, points, shapes });

  if (colorbars.length === 0) {
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
      {colorbars.map(({ key, name, from, tableId }) => (
        <Colorbar key={key} name={name} from={from} tableId={tableId} />
      ))}
    </div>
  );
}
