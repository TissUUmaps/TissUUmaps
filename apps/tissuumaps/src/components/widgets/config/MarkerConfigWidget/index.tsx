import { ProjectUtils } from "@tissuumaps/core";

import { Field, FieldLabel } from "@/components/common/field";
import { SimpleSelect } from "@/components/common/simple-select";
import { markers } from "@/components/markers";
import { TableColumnInput } from "@/components/widgets/TableColumnInput";
import { GroupValueMapSelect } from "@/components/widgets/config/GroupValueMapSelect";
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
    currentGroupByColumn: column,
    currentGroupByMap: map,
    setCurrentGroupByColumn: setColumn,
    setCurrentGroupByMap: setMap,
  } = adapter;

  const markerMaps = useProjectStore((state) => state.markerMaps);
  const deleteMarkerMap = useProjectStore((state) => state.deleteMarkerMap);
  const referencedMapIds = useReferencedMapIds((project) =>
    ProjectUtils.getMarkerConfigs(project),
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
