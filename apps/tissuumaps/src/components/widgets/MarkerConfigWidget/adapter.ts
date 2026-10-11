import type { Marker, MarkerConfig, TableColumnRef } from "@tissuumaps/core";

export type MarkerConfigSource = Exclude<MarkerConfig["source"], undefined>;

export type MarkerConfigWidgetAdapter = {
  markerConfig: MarkerConfig;
  defaultMarker: Marker;
  tableId: string | null;
  activeSource: MarkerConfigSource;
  currentSource: MarkerConfigSource;
  currentConstantValue: Marker;
  currentFromTableColumn: TableColumnRef | null;
  currentGroupByTableColumn: TableColumnRef | null;
  currentGroupByMap: string | null;
  setCurrentSource: (newCurrentSource: MarkerConfigSource) => void;
  setCurrentConstantValue: (newCurrentValue: Marker) => void;
  setCurrentFromTableColumn: (
    newCurrentFromTableColumn: TableColumnRef | null,
  ) => void;
  setCurrentGroupByTableColumn: (
    newCurrentGroupByTableColumn: TableColumnRef | null,
  ) => void;
  setCurrentGroupByMap: (newCurrentGroupByMap: string | null) => void;
};
