import { ProjectUtils } from "@tissuumaps/core";

import { Field, FieldLabel } from "@/components/common/field";
import { SimpleSelect } from "@/components/common/simple-select";
import { GroupValueMapSelect } from "@/components/controls/GroupValueMapSelect";
import { TableColumnField } from "@/components/controls/TableColumnField";
import { markers } from "@/components/markers";
import { useReferencedMapIds } from "@/hooks/useReferencedMapIds";
import { useProjectStore } from "@/stores/project";

import type { MarkerConfigWidgetAdapter } from "./adapter";

export { ActiveMarkerConfigValue } from "./ActiveMarkerConfigValue";
export { MarkerConfigSourceToggleGroup } from "./MarkerConfigSourceToggleGroup";

export type MarkerConfigWidgetProps = {
  adapter: MarkerConfigWidgetAdapter;
  className?: string;
};

export function MarkerConfigWidget({
  adapter,
  className,
}: MarkerConfigWidgetProps) {
  switch (adapter.currentSource) {
    case "constant":
      return (
        <ConstantMarkerConfigWidget adapter={adapter} className={className} />
      );
    case "from":
      return <FromMarkerConfigWidget adapter={adapter} className={className} />;
    case "groupBy":
      return (
        <GroupByMarkerConfigWidget adapter={adapter} className={className} />
      );
  }
}

type ConstantMarkerConfigWidgetProps = {
  adapter: MarkerConfigWidgetAdapter;
  className?: string;
};

function ConstantMarkerConfigWidget({
  adapter,
  className,
}: ConstantMarkerConfigWidgetProps) {
  const { currentConstantValue: value, setCurrentConstantValue: setValue } =
    adapter;

  return (
    <div className={className}>
      <Field>
        <FieldLabel>Marker</FieldLabel>
        <SimpleSelect
          items={markers}
          itemLabel={(marker) => (
            <>
              {marker.icon} {marker.label}
            </>
          )}
          itemValue={(marker) => marker.value}
          value={value}
          onValueChange={(value) => {
            if (value !== null) {
              setValue(value);
            }
          }}
        />
      </Field>
    </div>
  );
}

type FromMarkerConfigWidgetProps = {
  adapter: MarkerConfigWidgetAdapter;
  className?: string;
};

function FromMarkerConfigWidget({
  adapter,
  className,
}: FromMarkerConfigWidgetProps) {
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

type GroupByMarkerConfigWidgetProps = {
  adapter: MarkerConfigWidgetAdapter;
  className?: string;
};

function GroupByMarkerConfigWidget({
  adapter,
  className,
}: GroupByMarkerConfigWidgetProps) {
  const {
    tableId,
    currentGroupByTableColumn: groupBy,
    currentGroupByMap: map,
    setCurrentGroupByTableColumn: setGroupBy,
    setCurrentGroupByMap: setMap,
  } = adapter;

  const markerMaps = useProjectStore((state) => state.markerMaps);
  const deleteMarkerMap = useProjectStore((state) => state.deleteMarkerMap);
  const referencedMapIds = useReferencedMapIds((project) =>
    ProjectUtils.getMarkerConfigs(project),
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
        <FieldLabel>Marker map</FieldLabel>
        <GroupValueMapSelect
          maps={markerMaps}
          isMapDeletable={(map) => !referencedMapIds.has(map.id)}
          value={map}
          onValueChange={setMap}
          onMapDelete={deleteMarkerMap}
        />
      </Field>
    </div>
  );
}
