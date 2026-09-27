import { type CoordinateSpace, ProjectUtils } from "@tissuumaps/core";

import { Field, FieldItem, FieldLabel } from "@/components/common/field";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { TableColumnInput } from "@/components/widgets/TableColumnInput";
import { GroupValueMapSelect } from "@/components/widgets/config/GroupValueMapSelect";
import { useReferencedMapIds } from "@/hooks/useReferencedMapIds";
import { useProjectStore } from "@/stores/project";

import type { SizeConfigWidgetAdapter } from "./adapter";

export { ActiveSizeConfigValue } from "./ActiveSizeConfigValue";
export { SizeConfigSourceToggleGroup } from "./SizeConfigSourceToggleGroup";

export type SizeConfigWidgetProps = {
  adapter: SizeConfigWidgetAdapter;
  className?: string;
};

export function SizeConfigWidget({
  adapter,
  className,
}: SizeConfigWidgetProps) {
  switch (adapter.currentSource) {
    case "constant":
      return (
        <ConstantSizeConfigWidget adapter={adapter} className={className} />
      );
    case "from":
      return <FromSizeConfigWidget adapter={adapter} className={className} />;
    case "groupBy":
      return (
        <GroupBySizeConfigWidget adapter={adapter} className={className} />
      );
  }
}

type ConstantSizeConfigWidgetProps = {
  adapter: SizeConfigWidgetAdapter;
  className?: string;
};

function ConstantSizeConfigWidget({
  adapter,
  className,
}: ConstantSizeConfigWidgetProps) {
  const {
    currentConstantValue: value,
    currentConstantUnit: unit,
    setCurrentConstantValue: setValue,
    setCurrentConstantUnit: setUnit,
  } = adapter;

  return (
    <div className={className}>
      <Field>
        <FieldLabel>Size</FieldLabel>
        <Input
          type="number"
          inputMode="decimal"
          min={0}
          value={value}
          onChange={(event) => {
            const newValue = event.target.valueAsNumber;
            if (!isNaN(newValue)) {
              setValue(Math.max(0, newValue));
            }
          }}
        />
      </Field>
      <Field>
        <FieldLabel>Size unit</FieldLabel>
        <RadioGroup
          value={unit}
          onValueChange={(value) => setUnit(value as CoordinateSpace)}
          className="flex gap-x-4"
        >
          <FieldItem className="flex items-center gap-x-2">
            <RadioGroupItem value={"data" satisfies CoordinateSpace} />
            <FieldLabel>Data</FieldLabel>
          </FieldItem>
          <FieldItem className="flex items-center gap-x-2">
            <RadioGroupItem value={"layer" satisfies CoordinateSpace} />
            <FieldLabel>Layer</FieldLabel>
          </FieldItem>
          <FieldItem className="flex items-center gap-x-2">
            <RadioGroupItem value={"world" satisfies CoordinateSpace} />
            <FieldLabel>World</FieldLabel>
          </FieldItem>
        </RadioGroup>
      </Field>
    </div>
  );
}

type FromSizeConfigWidgetProps = {
  adapter: SizeConfigWidgetAdapter;
  className?: string;
};

function FromSizeConfigWidget({
  adapter,
  className,
}: FromSizeConfigWidgetProps) {
  const {
    tableId,
    currentFromColumn: column,
    currentFromUnit: unit,
    setCurrentFromColumn: setColumn,
    setCurrentFromUnit: setUnit,
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
        <FieldLabel>Size unit</FieldLabel>
        <RadioGroup
          value={unit}
          onValueChange={(value) => setUnit(value as CoordinateSpace)}
          className="flex gap-x-4"
        >
          <FieldItem className="flex items-center gap-x-2">
            <RadioGroupItem value={"data" satisfies CoordinateSpace} />
            <FieldLabel>Data</FieldLabel>
          </FieldItem>
          <FieldItem className="flex items-center gap-x-2">
            <RadioGroupItem value={"layer" satisfies CoordinateSpace} />
            <FieldLabel>Layer</FieldLabel>
          </FieldItem>
          <FieldItem className="flex items-center gap-x-2">
            <RadioGroupItem value={"world" satisfies CoordinateSpace} />
            <FieldLabel>World</FieldLabel>
          </FieldItem>
        </RadioGroup>
      </Field>
    </div>
  );
}

type GroupBySizeConfigWidgetProps = {
  adapter: SizeConfigWidgetAdapter;
  className?: string;
};

function GroupBySizeConfigWidget({
  adapter,
  className,
}: GroupBySizeConfigWidgetProps) {
  const {
    tableId,
    currentGroupByColumn: column,
    currentGroupByMap: map,
    currentGroupByUnit: unit,
    setCurrentGroupByColumn: setColumn,
    setCurrentGroupByMap: setMap,
    setCurrentGroupByUnit: setUnit,
  } = adapter;

  const sizeMaps = useProjectStore((state) => state.sizeMaps);
  const deleteSizeMap = useProjectStore((state) => state.deleteSizeMap);
  const referencedMapIds = useReferencedMapIds((project) =>
    ProjectUtils.getSizeConfigs(project),
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
      <Field>
        <FieldLabel>Size map</FieldLabel>
        <GroupValueMapSelect
          maps={sizeMaps}
          isMapDeletable={(map) => !referencedMapIds.has(map.id)}
          value={map}
          onValueChange={setMap}
          onMapDelete={deleteSizeMap}
        />
      </Field>
      <Field>
        <FieldLabel>Size unit</FieldLabel>
        <RadioGroup
          value={unit}
          onValueChange={(value) => setUnit(value as CoordinateSpace)}
          className="flex gap-x-4"
        >
          <FieldItem className="flex items-center gap-x-2">
            <RadioGroupItem value={"data" satisfies CoordinateSpace} />
            <FieldLabel>Data</FieldLabel>
          </FieldItem>
          <FieldItem className="flex items-center gap-x-2">
            <RadioGroupItem value={"layer" satisfies CoordinateSpace} />
            <FieldLabel>Layer</FieldLabel>
          </FieldItem>
          <FieldItem className="flex items-center gap-x-2">
            <RadioGroupItem value={"world" satisfies CoordinateSpace} />
            <FieldLabel>World</FieldLabel>
          </FieldItem>
        </RadioGroup>
      </Field>
    </div>
  );
}
