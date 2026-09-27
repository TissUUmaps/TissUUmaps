import { ProjectUtils } from "@tissuumaps/core";

import { Field, FieldLabel } from "@/components/common/field";
import { Switch } from "@/components/ui/switch";
import { TableColumnInput } from "@/components/widgets/TableColumnInput";
import { GroupValueMapSelect } from "@/components/widgets/config/GroupValueMapSelect";
import { useReferencedMapIds } from "@/hooks/useReferencedMapIds";
import { useProjectStore } from "@/stores/project";

import type { VisibilityConfigWidgetAdapter } from "./adapter";

export { ActiveVisibilityConfigValue } from "./ActiveVisibilityConfigValue";
export { VisibilityConfigSourceToggleGroup } from "./VisibilityConfigSourceToggleGroup";

export type VisibilityConfigWidgetProps = {
  adapter: VisibilityConfigWidgetAdapter;
  className?: string;
};

export function VisibilityConfigWidget({
  adapter,
  className,
}: VisibilityConfigWidgetProps) {
  switch (adapter.currentSource) {
    case "constant":
      return (
        <ConstantVisibilityConfigWidget
          adapter={adapter}
          className={className}
        />
      );
    case "from":
      return (
        <FromVisibilityConfigWidget adapter={adapter} className={className} />
      );
    case "groupBy":
      return (
        <GroupByVisibilityConfigWidget
          adapter={adapter}
          className={className}
        />
      );
  }
}

type ConstantVisibilityConfigWidgetProps = {
  adapter: VisibilityConfigWidgetAdapter;
  className?: string;
};

function ConstantVisibilityConfigWidget({
  adapter,
  className,
}: ConstantVisibilityConfigWidgetProps) {
  const { currentConstantValue: value, setCurrentConstantValue: setValue } =
    adapter;

  return (
    <div className={className}>
      <Field>
        <FieldLabel>Visibility</FieldLabel>
        <div className="flex flex-row items-center gap-x-2">
          <Switch checked={value} onCheckedChange={setValue} />
          {value ? "Visible" : "Hidden"}
        </div>
      </Field>
    </div>
  );
}

type FromVisibilityConfigWidgetProps = {
  adapter: VisibilityConfigWidgetAdapter;
  className?: string;
};

function FromVisibilityConfigWidget({
  adapter,
  className,
}: FromVisibilityConfigWidgetProps) {
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

type GroupByVisibilityConfigWidgetProps = {
  adapter: VisibilityConfigWidgetAdapter;
  className?: string;
};

function GroupByVisibilityConfigWidget({
  adapter,
  className,
}: GroupByVisibilityConfigWidgetProps) {
  const {
    tableId,
    currentGroupByColumn: column,
    currentGroupByMap: map,
    setCurrentGroupByColumn: setColumn,
    setCurrentGroupByMap: setMap,
  } = adapter;

  const visibilityMaps = useProjectStore((state) => state.visibilityMaps);
  const deleteVisibilityMap = useProjectStore(
    (state) => state.deleteVisibilityMap,
  );
  const referencedMapIds = useReferencedMapIds((project) =>
    ProjectUtils.getVisibilityConfigs(project),
  );

  return (
    <div className={className}>
      <Field disabled={tableId === null}>
        <FieldLabel>Source column</FieldLabel>
        <TableColumnInput
          tableId={tableId}
          value={column}
          onValueChange={setColumn}
        />
      </Field>
      <Field>
        <FieldLabel>Visibility map</FieldLabel>
        <GroupValueMapSelect
          maps={visibilityMaps}
          isMapDeletable={(map) => !referencedMapIds.has(map.id)}
          value={map}
          onValueChange={setMap}
          onMapDelete={deleteVisibilityMap}
        />
      </Field>
    </div>
  );
}
