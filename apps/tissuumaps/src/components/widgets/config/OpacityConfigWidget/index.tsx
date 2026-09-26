import { MathUtils, ProjectUtils } from "@tissuumaps/core";

import { Field, FieldLabel } from "@/components/common/field";
import { Input } from "@/components/ui/input";
import { TableColumnInput } from "@/components/widgets/TableColumnInput";
import { GroupValueMapSelect } from "@/components/widgets/config/GroupValueMapSelect";
import { useReferencedMapIds } from "@/hooks/useReferencedMapIds";
import { useProjectStore } from "@/stores/project";

import type { OpacityConfigWidgetAdapter } from "./adapter";

export { ActiveOpacityConfigValue } from "./ActiveOpacityConfigValue";
export { OpacityConfigSourceToggleGroup } from "./OpacityConfigSourceToggleGroup";

export type OpacityConfigWidgetProps = {
  adapter: OpacityConfigWidgetAdapter;
  className?: string;
};

export function OpacityConfigWidget({
  adapter,
  className,
}: OpacityConfigWidgetProps) {
  switch (adapter.currentSource) {
    case "constant":
      return (
        <ConstantOpacityConfigWidget adapter={adapter} className={className} />
      );
    case "from":
      return (
        <FromOpacityConfigWidget adapter={adapter} className={className} />
      );
    case "groupBy":
      return (
        <GroupByOpacityConfigWidget adapter={adapter} className={className} />
      );
  }
}

type ConstantOpacityConfigWidgetProps = {
  adapter: OpacityConfigWidgetAdapter;
  className?: string;
};

function ConstantOpacityConfigWidget({
  adapter,
  className,
}: ConstantOpacityConfigWidgetProps) {
  const { currentConstantValue: value, setCurrentConstantValue: setValue } =
    adapter;

  return (
    <div className={className}>
      <Field>
        <FieldLabel>Opacity</FieldLabel>
        <Input
          type="number"
          inputMode="decimal"
          step={0.05}
          min={0}
          max={1}
          value={value}
          onChange={(event) => {
            const newValue = event.target.valueAsNumber;
            if (!isNaN(newValue)) {
              setValue(MathUtils.clamp(newValue, 0, 1));
            }
          }}
        />
      </Field>
    </div>
  );
}

type FromOpacityConfigWidgetProps = {
  adapter: OpacityConfigWidgetAdapter;
  className?: string;
};

function FromOpacityConfigWidget({
  adapter,
  className,
}: FromOpacityConfigWidgetProps) {
  const {
    tableId,
    currentFromColumn: column,
    setCurrentFromColumn: setColumn,
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
    </div>
  );
}

type GroupByOpacityConfigWidgetProps = {
  adapter: OpacityConfigWidgetAdapter;
  className?: string;
};

function GroupByOpacityConfigWidget({
  adapter,
  className,
}: GroupByOpacityConfigWidgetProps) {
  const {
    tableId,
    currentGroupByColumn: column,
    currentGroupByMap: map,
    setCurrentGroupByColumn: setColumn,
    setCurrentGroupByMap: setMap,
  } = adapter;

  const opacityMaps = useProjectStore((state) => state.opacityMaps);
  const deleteOpacityMap = useProjectStore((state) => state.deleteOpacityMap);
  const referencedMapIds = useReferencedMapIds((project) =>
    ProjectUtils.getOpacityConfigs(project),
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
        <FieldLabel>Opacity map</FieldLabel>
        <GroupValueMapSelect
          maps={opacityMaps}
          isMapDeletable={(map) => !referencedMapIds.has(map.id)}
          value={map}
          onValueChange={setMap}
          onMapDelete={deleteOpacityMap}
        />
      </Field>
    </div>
  );
}
