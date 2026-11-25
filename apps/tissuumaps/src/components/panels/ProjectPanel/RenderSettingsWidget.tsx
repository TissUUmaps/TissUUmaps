import { RotateCcwIcon, SquareIcon } from "lucide-react";

import {
  ColorUtils,
  PhysicalUnit,
  physicalUnits,
  projectDefaults,
} from "@tissuumaps/core";

import {
  Autocomplete,
  AutocompleteInput,
  AutocompleteInputGroup,
  AutocompleteItem,
  AutocompleteList,
  AutocompletePopup,
  AutocompleteTrigger,
} from "@/components/common/autocomplete";
import {
  Collapsible,
  CollapsiblePanel,
  CollapsibleTrigger,
  CollapsibleTriggerRightDownIcon,
} from "@/components/common/collapsible";
import { ColorPicker } from "@/components/common/color-picker";
import {
  Field,
  FieldControl,
  FieldDescription,
  FieldLabel,
} from "@/components/common/field";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { InputGroupAddon } from "@/components/ui/input-group";
import { cn } from "@/lib/utils";
import { useProjectStore } from "@/stores/project";

export type RenderSettingsWidgetProps = {
  className?: string;
};

export function RenderSettingsWidget({ className }: RenderSettingsWidgetProps) {
  const glOptions = useProjectStore((state) => state.glOptions);
  const setGLOptions = useProjectStore((state) => state.setGLOptions);
  const viewerBackgroundColor = useProjectStore(
    (state) => state.viewerBackgroundColor,
  );
  const setViewerBackgroundColor = useProjectStore(
    (state) => state.setViewerBackgroundColor,
  );
  const physicalUnitSize = useProjectStore((state) => state.physicalUnitSize);
  const setPhysicalUnitSize = useProjectStore(
    (state) => state.setPhysicalUnitSize,
  );
  const physicalUnit = useProjectStore((state) => state.physicalUnit);
  const setPhysicalUnit = useProjectStore((state) => state.setPhysicalUnit);

  const knownPhysicalUnit = Object.values(PhysicalUnit).find(
    (unit) => unit === physicalUnit,
  );
  const { globalPointSizeFactor } = glOptions.pointsRenderOptions;
  const viewerBackgroundColorHex = ColorUtils.toHex(viewerBackgroundColor);

  return (
    <div className={cn("flex flex-col gap-2 pl-6", className)}>
      <Field className="flex flex-col items-start">
        <FieldLabel>Viewer background color</FieldLabel>
        <ColorPicker
          color={viewerBackgroundColor}
          onColorChange={setViewerBackgroundColor}
          className="font-mono"
        >
          <span className="sr-only">Viewer background color</span>
          <SquareIcon fill={viewerBackgroundColorHex} />
          {viewerBackgroundColorHex}
        </ColorPicker>
      </Field>
      <Field>
        <FieldLabel>World unit size</FieldLabel>
        <div className="flex flex-row gap-x-1">
          <FieldControl
            render={
              <Input
                type="number"
                inputMode="decimal"
                min={0}
                step="any"
                placeholder="Unknown"
                className="min-w-0 flex-1"
                value={physicalUnitSize ?? ""}
                onChange={(event) => {
                  const newValue = event.target.valueAsNumber;
                  setPhysicalUnitSize(
                    isNaN(newValue) ? undefined : Math.max(0, newValue),
                  );
                }}
              />
            }
          />
          <Field className="flex w-24 shrink-0">
            <FieldLabel className="sr-only">Physical unit</FieldLabel>
            <Autocomplete
              value={
                knownPhysicalUnit !== undefined
                  ? physicalUnits[knownPhysicalUnit].symbol
                  : (physicalUnit ?? "")
              }
              onValueChange={(text) => {
                // the micro sign and the Greek letter mu are the same symbol
                const symbol = text.trim().normalize("NFKC");
                setPhysicalUnit(
                  Object.values(PhysicalUnit).find(
                    (unit) =>
                      physicalUnits[unit].symbol.normalize("NFKC") === symbol,
                  ) ?? (symbol === "" ? undefined : text),
                );
              }}
              mode="none"
              items={Object.values(PhysicalUnit)}
              itemToStringValue={(unit) => physicalUnits[unit].symbol}
              openOnInputClick
            >
              <AutocompleteInputGroup>
                <AutocompleteInput placeholder="Unit" />
                <InputGroupAddon align="inline-end">
                  <AutocompleteTrigger label="Show units" />
                </InputGroupAddon>
              </AutocompleteInputGroup>
              <AutocompletePopup>
                <AutocompleteList>
                  {(unit: PhysicalUnit) => (
                    <AutocompleteItem key={unit} value={unit}>
                      {physicalUnits[unit].symbol}
                    </AutocompleteItem>
                  )}
                </AutocompleteList>
              </AutocompletePopup>
            </Autocomplete>
          </Field>
        </div>
        <FieldDescription className="text-muted-foreground text-xs">
          Physical size of one world unit and its unit, for the scalebar; a
          world unit is an image pixel, unless transformed
        </FieldDescription>
      </Field>
      <Field>
        <FieldLabel>Point size factor</FieldLabel>
        <FieldControl
          render={
            <Input
              type="number"
              inputMode="decimal"
              step={0.1}
              min={0}
              value={globalPointSizeFactor}
              onChange={(event) => {
                const newValue = event.target.valueAsNumber;
                if (!isNaN(newValue)) {
                  setGLOptions({
                    ...glOptions,
                    pointsRenderOptions: {
                      ...glOptions.pointsRenderOptions,
                      globalPointSizeFactor: Math.max(0, newValue),
                    },
                  });
                }
              }}
            />
          }
        />
      </Field>
      <Field>
        <FieldLabel>Shape stroke width</FieldLabel>
        <FieldControl
          render={
            <Input
              type="number"
              inputMode="decimal"
              step={0.1}
              min={0}
              value={glOptions.shapesRenderOptions.strokeWidth}
              onChange={(event) => {
                const newValue = event.target.valueAsNumber;
                if (!isNaN(newValue)) {
                  setGLOptions({
                    ...glOptions,
                    shapesRenderOptions: {
                      ...glOptions.shapesRenderOptions,
                      strokeWidth: Math.max(0, newValue),
                    },
                  });
                }
              }}
            />
          }
        />
        <FieldDescription className="text-muted-foreground text-xs">
          In world coordinates
        </FieldDescription>
      </Field>
      <Collapsible>
        <div className="flex flex-row items-center">
          <CollapsibleTriggerRightDownIcon />
          <CollapsibleTrigger>Advanced rendering</CollapsibleTrigger>
        </div>
        <CollapsiblePanel className="flex flex-col gap-2 p-2 pb-4 pl-6">
          <Field>
            <FieldLabel>Shape edges per scanline</FieldLabel>
            <FieldControl
              render={
                <Input
                  type="number"
                  inputMode="decimal"
                  step={1}
                  min={1}
                  value={glOptions.shapesRenderOptions.edgesPerScanline}
                  onChange={(event) => {
                    const newValue = event.target.valueAsNumber;
                    if (!isNaN(newValue)) {
                      setGLOptions({
                        ...glOptions,
                        shapesRenderOptions: {
                          ...glOptions.shapesRenderOptions,
                          edgesPerScanline: Math.max(1, newValue),
                        },
                      });
                    }
                  }}
                />
              }
            />
            <FieldDescription className="text-muted-foreground text-xs">
              Fewer is faster, but uses more memory
            </FieldDescription>
          </Field>
          <Field>
            <FieldLabel>Shape bin width factor</FieldLabel>
            <FieldControl
              render={
                <Input
                  type="number"
                  inputMode="decimal"
                  step={0.1}
                  min={0.1}
                  value={glOptions.shapesRenderOptions.binWidthFactor}
                  onChange={(event) => {
                    const newValue = event.target.valueAsNumber;
                    if (!isNaN(newValue)) {
                      setGLOptions({
                        ...glOptions,
                        shapesRenderOptions: {
                          ...glOptions.shapesRenderOptions,
                          binWidthFactor: Math.max(0.1, newValue),
                        },
                      });
                    }
                  }}
                />
              }
            />
            <FieldDescription className="text-muted-foreground text-xs">
              Relative to the median shape width; smaller is faster, but uses
              more memory
            </FieldDescription>
          </Field>
          <Field>
            <FieldLabel>Shape padding</FieldLabel>
            <FieldControl
              render={
                <Input
                  type="number"
                  inputMode="decimal"
                  step={0.1}
                  min={0}
                  value={glOptions.shapesRenderOptions.shapePadding}
                  onChange={(event) => {
                    const newValue = event.target.valueAsNumber;
                    if (!isNaN(newValue)) {
                      setGLOptions({
                        ...glOptions,
                        shapesRenderOptions: {
                          ...glOptions.shapesRenderOptions,
                          shapePadding: Math.max(0, newValue),
                        },
                      });
                    }
                  }}
                />
              }
            />
            <FieldDescription className="text-muted-foreground text-xs">
              Relative to the median shape size; larger lets strokes reach
              further, but is slower
            </FieldDescription>
          </Field>
          <Button
            variant="ghost"
            size="sm"
            className="text-muted-foreground self-start"
            onClick={() => {
              const { edgesPerScanline, binWidthFactor, shapePadding } =
                projectDefaults.glOptions.shapesRenderOptions;
              setGLOptions({
                ...glOptions,
                shapesRenderOptions: {
                  ...glOptions.shapesRenderOptions,
                  edgesPerScanline,
                  binWidthFactor,
                  shapePadding,
                },
              });
            }}
          >
            <RotateCcwIcon />
            Reset to defaults
          </Button>
        </CollapsiblePanel>
      </Collapsible>
    </div>
  );
}
