import type { OpacityConfig, TableColumnRef } from "@tissuumaps/core";

export type OpacityConfigSource = Exclude<OpacityConfig["source"], undefined>;

export type OpacityConfigWidgetAdapter = {
  opacityConfig: OpacityConfig;
  defaultOpacity: number;
  tableId: string | null;
  activeSource: OpacityConfigSource;
  fromColumnValueRange: [number, number] | null;
  currentSource: OpacityConfigSource;
  currentConstantValue: number;
  currentFromTableColumn: TableColumnRef | null;
  currentFromRangeMin: number | null;
  currentFromRangeMax: number | null;
  currentGroupByTableColumn: TableColumnRef | null;
  currentGroupByMap: string | null;
  setCurrentSource: (newCurrentSource: OpacityConfigSource) => void;
  setCurrentConstantValue: (newCurrentConstantValue: number) => void;
  setCurrentFromTableColumn: (
    newCurrentFromTableColumn: TableColumnRef | null,
  ) => void;
  setCurrentFromRangeMin: (newCurrentFromRangeMin: number | null) => void;
  setCurrentFromRangeMax: (newCurrentFromRangeMax: number | null) => void;
  setCurrentGroupByTableColumn: (
    newCurrentGroupByTableColumn: TableColumnRef | null,
  ) => void;
  setCurrentGroupByMap: (newCurrentGroupByMap: string | null) => void;
};
