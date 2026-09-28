import { ProjectUtils } from "@tissuumaps/core";

import { Field, FieldLabel } from "@/components/common/field";
import { OpacityControl } from "@/components/common/opacity-control";
import { TableColumnField } from "@/components/widgets/TableColumnField";
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
        <OpacityControl
          opacity={value}
          onOpacityCommit={setValue}
          className="self-start"
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
    currentFromTableColumn: from,
    setCurrentFromTableColumn: setFrom,
  } = adapter;

  return (
    <div className={className}>
      <TableColumnField
        label="Column"
        tableId={tableId}
        value={from}
        onValueChange={setFrom}
      />
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
    currentGroupByTableColumn: groupBy,
    currentGroupByMap: map,
    setCurrentGroupByTableColumn: setGroupBy,
    setCurrentGroupByMap: setMap,
  } = adapter;

  const opacityMaps = useProjectStore((state) => state.opacityMaps);
  const deleteOpacityMap = useProjectStore((state) => state.deleteOpacityMap);
  const referencedMapIds = useReferencedMapIds((project) =>
    ProjectUtils.getOpacityConfigs(project),
  );

  return (
    <div className={className}>
      <TableColumnField
        label="Column"
        tableId={tableId}
        value={groupBy}
        onValueChange={setGroupBy}
      />
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
