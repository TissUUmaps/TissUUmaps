import type { ReactNode } from "react";

import { continuousColorPalettes } from "@tissuumaps/core";

import { Fieldset, FieldsetLegend } from "@/components/common/fieldset";
import { Input } from "@/components/ui/input";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { ColorPaletteSelect } from "@/components/widgets/ColorConfigWidget/ColorPaletteSelect";
import type { ColorConfigWidgetAdapter } from "@/components/widgets/ColorConfigWidget/adapter";
import { cn } from "@/lib/utils";

import { ColorByPicker } from "./ColorByPicker";
import { StyleRow } from "./StyleRow";
import type { ColorBy, ColorByEncoding } from "./colorBy";
import { useColorBy } from "./useColorBy";

export type StyleWidgetProps = {
  /** The ID of the object's own table, if it has one */
  tableId: string | null;

  /** What the object is colored by (see `readColorBy`) */
  colorBy: ColorBy;

  /**
   * Writes what the object is colored by, to the object as it is when called
   * (the call may follow a load)
   */
  onColorByChange: (colorBy: ColorBy) => void;

  /** The color configuration, whose gradient the widget edits */
  colorAdapter: ColorConfigWidgetAdapter;

  /**
   * What the object looks like while colored by no column, e.g. "Random
   * colors" for labels
   *
   * @defaultValue "None"
   */
  noneLabel?: string;

  /**
   * A row shown while coloring by categories, below the encoding, e.g. the
   * marker of points
   */
  categoryOptions?: ReactNode;

  className?: string;
};

/**
 * The always available settings of an object annotated by a table: what to
 * color it by, as categories or as a gradient, in compact labelled rows
 *
 * The encoding follows the picked column and determines the rows below it.
 * The colors of the categories are edited in the group table below the
 * widget; the other settings in the advanced settings.
 */
export function StyleWidget({
  tableId,
  colorBy,
  onColorByChange,
  colorAdapter,
  noneLabel = "None",
  categoryOptions,
  className,
}: StyleWidgetProps) {
  const { isCategorical, isPending, error, pick, setEncoding } = useColorBy(
    tableId,
    colorBy,
    onColorByChange,
  );

  const isGradientDisabled = isCategorical === true;
  const encoding = colorBy.kind === "column" ? colorBy.encoding : null;

  return (
    <Fieldset
      disabled={tableId === null}
      className={cn(
        "grid grid-cols-[4.5rem_1fr] items-center gap-x-2 gap-y-1.5 border rounded-md p-2",
        className,
      )}
    >
      <FieldsetLegend className="sr-only">Style</FieldsetLegend>
      <ColorByPicker
        tableId={tableId}
        column={colorBy.kind === "column" ? colorBy.column : null}
        onColumnChange={pick}
        placeholder={
          tableId === null
            ? "No table"
            : colorBy.kind === "custom"
              ? colorBy.label
              : noneLabel
        }
      />
      {encoding !== null && (
        <StyleRow label="Show as">
          <ToggleGroup
            value={[encoding]}
            onValueChange={(value) => {
              if (value.length === 1 && value[0] !== undefined) {
                setEncoding(value[0] as ColorByEncoding);
              }
            }}
            size="sm"
            className="border rounded"
            // the encodings a column allows are known with its type
            disabled={isPending || isCategorical === null}
          >
            <ToggleGroupItem
              value={"categories" satisfies ColorByEncoding}
              className={
                encoding === "categories" ? "font-medium" : "font-normal"
              }
            >
              Categories
            </ToggleGroupItem>
            <ToggleGroupItem
              value={"gradient" satisfies ColorByEncoding}
              className={
                encoding === "gradient" ? "font-medium" : "font-normal"
              }
              disabled={isGradientDisabled}
              title={
                isGradientDisabled
                  ? "Categorical columns cannot be shown as a gradient"
                  : undefined
              }
            >
              Gradient
            </ToggleGroupItem>
          </ToggleGroup>
        </StyleRow>
      )}
      {encoding === "categories" && categoryOptions}
      {encoding === "gradient" && colorAdapter.currentSource === "from" && (
        <>
          <StyleRow label="Palette">
            <ColorPaletteSelect
              colorPalettes={continuousColorPalettes}
              value={colorAdapter.currentFromPalette}
              onValueChange={colorAdapter.setCurrentFromPalette}
            />
          </StyleRow>
          <StyleRow label="Range" className="grid grid-cols-2 gap-x-1">
            <RangeEndInput
              label="Min. value"
              value={colorAdapter.currentFromRangeMin}
              onValueChange={colorAdapter.setCurrentFromRangeMin}
              placeholder={colorAdapter.fromColumnValueRange?.[0]}
            />
            <RangeEndInput
              label="Max. value"
              value={colorAdapter.currentFromRangeMax}
              onValueChange={colorAdapter.setCurrentFromRangeMax}
              placeholder={colorAdapter.fromColumnValueRange?.[1]}
            />
          </StyleRow>
        </>
      )}
      {isPending && (
        <StyleRow>
          <span className="text-xs text-muted-foreground">Loading column…</span>
        </StyleRow>
      )}
      {error !== null && (
        <StyleRow>
          <span className="text-xs text-destructive">{error}</span>
        </StyleRow>
      )}
    </Fieldset>
  );
}

type RangeEndInputProps = {
  label: string;
  value: number | null;
  onValueChange: (value: number | null) => void;
  /** The column's minimum or maximum value, shown while the end is unset */
  placeholder: number | undefined;
};

/** One end of the gradient's value range; an empty end is the column's own */
function RangeEndInput({
  label,
  value,
  onValueChange,
  placeholder,
}: RangeEndInputProps) {
  return (
    <Input
      type="number"
      inputMode="decimal"
      aria-label={label}
      title={label}
      placeholder={placeholder?.toPrecision(6)}
      value={value ?? ""}
      onChange={(event) => {
        if (event.target.value === "") {
          onValueChange(null);
        } else {
          const newValue = event.target.valueAsNumber;
          if (!isNaN(newValue)) {
            onValueChange(newValue);
          }
        }
      }}
      className="h-8 min-w-0"
    />
  );
}
