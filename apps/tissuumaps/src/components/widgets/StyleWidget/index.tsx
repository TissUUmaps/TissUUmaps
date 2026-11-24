import { RotateCcwIcon } from "lucide-react";
import { useEffect, useState } from "react";

import {
  type ColorConfig,
  TableUtils,
  continuousColorPalettes,
  isFromConfig,
} from "@tissuumaps/core";

import { Fieldset, FieldsetLegend } from "@/components/common/fieldset";
import { IconButton } from "@/components/common/icon-button";
import { Input } from "@/components/ui/input";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { ColorPaletteSelect } from "@/components/widgets/ColorConfigWidget/ColorPaletteSelect";
import { useTableData } from "@/data/hooks/useData";
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
  colorConfig: ColorConfig;

  onColorConfigChange: (colorConfig: ColorConfig) => void;

  /**
   * What the object looks like while colored by no column, e.g. "Random
   * colors" for labels
   *
   * @defaultValue "None"
   */
  noneLabel?: string;

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
  colorConfig,
  onColorConfigChange,
  noneLabel = "None",
  className,
}: StyleWidgetProps) {
  const { dataType, isPending, error, pick, setEncoding } = useColorBy(
    tableId,
    colorBy,
    onColorByChange,
  );

  const isGradientDisabled = dataType !== null && dataType !== "number";
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
            // the encodings a column allows are known with its data type
            disabled={isPending || dataType === null}
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
                  ? "Only numeric columns can be shown as a gradient"
                  : undefined
              }
            >
              Gradient
            </ToggleGroupItem>
          </ToggleGroup>
        </StyleRow>
      )}
      {encoding === "gradient" &&
        isFromConfig<{ palette: string; range?: [number, number] }>(
          colorConfig,
        ) && (
          <GradientSettings
            tableId={tableId}
            colorConfig={colorConfig}
            onColorConfigChange={onColorConfigChange}
            className="col-span-2"
          />
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

type GradientSettingsProps = {
  tableId: string | null;
  colorConfig: ColorConfig & {
    from: { column: string; palette: string; range?: [number, number] };
  };
  onColorConfigChange: (colorConfig: ColorConfig) => void;
  className?: string;
};

/**
 * `[min] ▓▓ palette ▓▓ [max]`: the range is the data range unless set
 *
 * The bounds can only be edited once the data range is known: editing one
 * stores the other as the data bound, as a range always has both; a cleared
 * bound returns to the data bound, a minimum above the maximum is refused,
 * and the reset button returns to the data range.
 */
function GradientSettings({
  tableId,
  colorConfig,
  onColorConfigChange,
  className,
}: GradientSettingsProps) {
  const { from } = colorConfig;
  const tableData = useTableData(tableId);
  const [valueRange, setValueRange] = useState<[number, number] | null>(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- no stale range
    setValueRange(null);
    if (tableData === null) {
      return;
    }
    const abortController = new AbortController();
    TableUtils.loadValueRange(tableData, from.column, {
      signal: abortController.signal,
    })
      .then((range) => {
        if (!abortController.signal.aborted) {
          setValueRange(range ?? null);
        }
      })
      .catch((error: unknown) => {
        if (!abortController.signal.aborted) {
          console.error("Error loading table value range", error);
        }
      });
    return () => abortController.abort();
  }, [tableData, from.column]);

  const [drafts, setDrafts] = useState<[string | null, string | null]>([
    null,
    null,
  ]);
  const [rangeError, setRangeError] = useState<string | null>(null);

  const setRange = (range: [number, number] | undefined) =>
    onColorConfigChange({ ...colorConfig, from: { ...from, range } });

  // a bound equal to the data bound is automatic, and shown empty
  const getBoundText = (index: 0 | 1) => {
    const bound = from.range?.[index];
    return bound === undefined || bound === valueRange?.[index]
      ? ""
      : String(bound);
  };

  const commitBound = (index: 0 | 1) => {
    const draft = drafts[index];
    if (draft === null || valueRange === null) {
      return;
    }
    const newRange: [number, number] = [...(from.range ?? valueRange)];
    const newValue = draft.trim() === "" ? valueRange[index] : Number(draft);
    if (isNaN(newValue)) {
      return;
    }
    newRange[index] = newValue;
    if (newRange[0] > newRange[1]) {
      setRangeError("The minimum must not exceed the maximum");
      return;
    }
    setRangeError(null);
    setDrafts((drafts) => {
      const newDrafts: [string | null, string | null] = [...drafts];
      newDrafts[index] = null;
      return newDrafts;
    });
    setRange(
      newRange[0] === valueRange[0] && newRange[1] === valueRange[1]
        ? undefined
        : newRange,
    );
  };

  const rangeInput = (index: 0 | 1) => (
    <Input
      type="number"
      inputMode="decimal"
      aria-label={index === 0 ? "Minimum value" : "Maximum value"}
      title={index === 0 ? "Minimum value" : "Maximum value"}
      aria-invalid={rangeError !== null || undefined}
      disabled={valueRange === null}
      placeholder={valueRange?.[index].toPrecision(4)}
      value={drafts[index] ?? getBoundText(index)}
      onChange={(event) => {
        const draft = event.target.value;
        setDrafts((drafts) => {
          const newDrafts: [string | null, string | null] = [...drafts];
          newDrafts[index] = draft;
          return newDrafts;
        });
      }}
      onBlur={() => commitBound(index)}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          commitBound(index);
        }
      }}
      className="h-8 w-20 shrink-0 px-1.5 text-xs placeholder:text-muted-foreground/70 md:text-xs"
    />
  );

  return (
    <div
      className={cn(
        "flex min-w-0 flex-row flex-wrap items-center gap-1",
        className,
      )}
    >
      {rangeInput(0)}
      <div className="min-w-12 flex-1">
        <ColorPaletteSelect
          colorPalettes={continuousColorPalettes}
          value={from.palette}
          onValueChange={(palette) => {
            if (palette !== null) {
              onColorConfigChange({
                ...colorConfig,
                from: { ...from, palette },
              });
            }
          }}
          asColorBar
        />
      </div>
      {rangeInput(1)}
      {from.range !== undefined && (
        <IconButton
          label="Reset to the data range"
          size="icon-xs"
          onClick={() => {
            setDrafts([null, null]);
            setRangeError(null);
            setRange(undefined);
          }}
        >
          <RotateCcwIcon />
        </IconButton>
      )}
      {rangeError !== null && (
        <span className="basis-full text-xs text-destructive">
          {rangeError}
        </span>
      )}
    </div>
  );
}
