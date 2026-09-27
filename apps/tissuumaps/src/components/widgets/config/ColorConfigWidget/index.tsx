import { RefreshCwIcon, Square } from "lucide-react";

import {
  MathUtils,
  ProjectUtils,
  RandomUtils,
  categoricalColorPalettes,
  continuousColorPalettes,
  defaultRandomSeed,
} from "@tissuumaps/core";

import { Field, FieldLabel } from "@/components/common/field";
import { SimpleColorPicker } from "@/components/common/simple-color-picker";
import { Input } from "@/components/ui/input";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
import { TableColumnInput } from "@/components/widgets/TableColumnInput";
import { GroupValueMapSelect } from "@/components/widgets/config/GroupValueMapSelect";
import { useReferencedMapIds } from "@/hooks/useReferencedMapIds";
import { useProjectStore } from "@/stores/project";

import { ColorPaletteSelect } from "./ColorPaletteSelect";
import type { ColorConfigWidgetAdapter } from "./adapter";

export { ActiveColorConfigValue } from "./ActiveColorConfigValue";
export { ColorConfigSourceToggleGroup } from "./ColorConfigSourceToggleGroup";

export type ColorConfigWidgetProps = {
  adapter: ColorConfigWidgetAdapter;
  className?: string;
};

export function ColorConfigWidget({
  adapter,
  className,
}: ColorConfigWidgetProps) {
  switch (adapter.currentSource) {
    case "constant":
      return (
        <ConstantColorConfigWidget adapter={adapter} className={className} />
      );
    case "from":
      return <FromColorConfigWidget adapter={adapter} className={className} />;
    case "groupBy":
      return (
        <GroupByColorConfigWidget adapter={adapter} className={className} />
      );
    case "random":
      return (
        <RandomColorConfigWidget adapter={adapter} className={className} />
      );
  }
}

type ConstantColorConfigWidgetProps = {
  adapter: ColorConfigWidgetAdapter;
  className?: string;
};

function ConstantColorConfigWidget({
  adapter,
  className,
}: ConstantColorConfigWidgetProps) {
  const { currentConstantValue: color, setCurrentConstantValue: setColor } =
    adapter;

  return (
    <div className={className}>
      <div className="grid grid-cols-4 grid-flow-col gap-x-2 items-center">
        <Field className="contents">
          <FieldLabel>Red</FieldLabel>
          <Input
            type="number"
            min={0}
            max={255}
            value={color.r}
            onChange={(event) => {
              const newValue = event.target.valueAsNumber;
              if (!isNaN(newValue)) {
                setColor({
                  ...color,
                  r: MathUtils.clamp(Math.trunc(newValue), 0, 255),
                });
              }
            }}
          />
        </Field>
        <Field className="contents">
          <FieldLabel>Green</FieldLabel>
          <Input
            type="number"
            min={0}
            max={255}
            value={color.g}
            onChange={(event) => {
              const newValue = event.target.valueAsNumber;
              if (!isNaN(newValue)) {
                setColor({
                  ...color,
                  g: MathUtils.clamp(Math.trunc(newValue), 0, 255),
                });
              }
            }}
          />
        </Field>
        <Field className="contents">
          <FieldLabel>Blue</FieldLabel>
          <Input
            type="number"
            min={0}
            max={255}
            value={color.b}
            onChange={(event) => {
              const newValue = event.target.valueAsNumber;
              if (!isNaN(newValue)) {
                setColor({
                  ...color,
                  b: MathUtils.clamp(Math.trunc(newValue), 0, 255),
                });
              }
            }}
          />
        </Field>
        <SimpleColorPicker
          color={color}
          onColorChange={setColor}
          className="row-start-2 col-start-4"
        >
          <Square fill={`rgb(${color.r}, ${color.g}, ${color.b})`} /> Pick
        </SimpleColorPicker>
      </div>
    </div>
  );
}

type FromColorConfigWidgetProps = {
  adapter: ColorConfigWidgetAdapter;
  className?: string;
};

function FromColorConfigWidget({
  adapter,
  className,
}: FromColorConfigWidgetProps) {
  const {
    tableId,
    fromColumnValueRange: columnValueRange,
    currentFromColumn: column,
    currentFromRangeMin: rangeMin,
    currentFromRangeMax: rangeMax,
    currentFromPalette: palette,
    setCurrentFromColumn: setColumn,
    setCurrentFromRangeMin: setRangeMin,
    setCurrentFromRangeMax: setRangeMax,
    setCurrentFromPalette: setPalette,
  } = adapter;

  return (
    <div className={className}>
      <Field disabled={tableId === null}>
        <FieldLabel>Table column</FieldLabel>
        <TableColumnInput
          tableId={tableId}
          value={column}
          onValueChange={setColumn}
        />
      </Field>
      <Field>
        <FieldLabel>Color palette</FieldLabel>
        <ColorPaletteSelect
          colorPalettes={continuousColorPalettes}
          value={palette}
          onValueChange={setPalette}
        />
      </Field>
      <div className="grid grid-cols-2 gap-x-2">
        <Field>
          <FieldLabel>Min. value</FieldLabel>
          <Input
            type="number"
            inputMode="decimal"
            placeholder={
              columnValueRange !== null
                ? columnValueRange[0].toPrecision(6)
                : undefined
            }
            value={rangeMin ?? ""}
            onChange={(event) => {
              if (event.target.value === "") {
                setRangeMin(null);
              } else {
                const newValue = event.target.valueAsNumber;
                if (!isNaN(newValue)) {
                  setRangeMin(newValue);
                }
              }
            }}
          />
        </Field>
        <Field>
          <FieldLabel>Max. value</FieldLabel>
          <Input
            type="number"
            inputMode="decimal"
            placeholder={
              columnValueRange !== null
                ? columnValueRange[1].toPrecision(6)
                : undefined
            }
            value={rangeMax ?? ""}
            onChange={(event) => {
              if (event.target.value === "") {
                setRangeMax(null);
              } else {
                const newValue = event.target.valueAsNumber;
                if (!isNaN(newValue)) {
                  setRangeMax(newValue);
                }
              }
            }}
          />
        </Field>
      </div>
    </div>
  );
}

type GroupByColorConfigWidgetProps = {
  adapter: ColorConfigWidgetAdapter;
  className?: string;
};

function GroupByColorConfigWidget({
  adapter,
  className,
}: GroupByColorConfigWidgetProps) {
  const {
    tableId,
    currentGroupByColumn: column,
    currentGroupByPalette: palette,
    currentGroupByMap: map,
    setCurrentGroupByColumn: setColumn,
    setCurrentGroupByPalette: setPalette,
    setCurrentGroupByMap: setMap,
  } = adapter;

  const colorMaps = useProjectStore((state) => state.colorMaps);
  const deleteColorMap = useProjectStore((state) => state.deleteColorMap);
  const referencedMapIds = useReferencedMapIds((project) =>
    ProjectUtils.getColorConfigs(project),
  );

  return (
    <div className={className}>
      <Field disabled={tableId === null}>
        <FieldLabel>Table column</FieldLabel>
        <TableColumnInput
          tableId={tableId}
          value={column}
          onValueChange={setColumn}
        />
      </Field>
      <Field disabled={map !== null}>
        <FieldLabel>Color palette</FieldLabel>
        <ColorPaletteSelect
          colorPalettes={categoricalColorPalettes}
          value={palette}
          onValueChange={setPalette}
        />
      </Field>
      <Field>
        <FieldLabel>Color map</FieldLabel>
        <GroupValueMapSelect
          maps={colorMaps}
          isMapDeletable={(map) => !referencedMapIds.has(map.id)}
          value={map}
          onValueChange={setMap}
          onMapDelete={deleteColorMap}
        />
      </Field>
    </div>
  );
}

type RandomColorConfigWidgetProps = {
  adapter: ColorConfigWidgetAdapter;
  className?: string;
};

function RandomColorConfigWidget({
  adapter,
  className,
}: RandomColorConfigWidgetProps) {
  const {
    currentRandomPalette: palette,
    currentRandomSeed: seed,
    setCurrentRandomPalette: setPalette,
    setCurrentRandomSeed: setSeed,
  } = adapter;
  return (
    <div className={className}>
      <Field>
        <FieldLabel>Color palette</FieldLabel>
        <ColorPaletteSelect
          colorPalettes={categoricalColorPalettes}
          value={palette}
          onValueChange={setPalette}
        />
      </Field>
      <Field>
        <FieldLabel>Seed</FieldLabel>
        <InputGroup>
          <InputGroupInput
            type="number"
            inputMode="numeric"
            step={1}
            placeholder={String(defaultRandomSeed)}
            value={seed ?? ""}
            onChange={(event) => {
              if (event.target.value === "") {
                setSeed(null);
              } else {
                const newValue = event.target.valueAsNumber;
                if (!isNaN(newValue)) {
                  setSeed(Math.trunc(newValue));
                }
              }
            }}
          />
          <InputGroupAddon align="inline-end">
            <InputGroupButton
              size="icon-xs"
              aria-label="Shuffle seed"
              title="Shuffle seed"
              onClick={() => setSeed(RandomUtils.seed())}
            >
              <RefreshCwIcon />
            </InputGroupButton>
          </InputGroupAddon>
        </InputGroup>
      </Field>
    </div>
  );
}
