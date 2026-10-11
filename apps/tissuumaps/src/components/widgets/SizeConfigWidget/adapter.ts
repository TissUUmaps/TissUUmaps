import type {
  CoordinateSpace,
  SizeConfig,
  TableColumnRef,
} from "@tissuumaps/core";

export type SizeConfigSource = Exclude<SizeConfig["source"], undefined>;

export type SizeConfigWidgetAdapter = {
  sizeConfig: SizeConfig;
  defaultSize: number;
  defaultSizeUnit: CoordinateSpace;
  tableId: string | null;
  activeSource: SizeConfigSource;
  currentSource: SizeConfigSource;
  currentConstantValue: number;
  currentConstantUnit: CoordinateSpace;
  currentFromTableColumn: TableColumnRef | null;
  currentFromUnit: CoordinateSpace;
  currentGroupByTableColumn: TableColumnRef | null;
  currentGroupByMap: string | null;
  currentGroupByUnit: CoordinateSpace;
  setCurrentSource: (newCurrentSource: SizeConfigSource) => void;
  setCurrentConstantValue: (newCurrentConstantValue: number) => void;
  setCurrentConstantUnit: (newCurrentConstantUnit: CoordinateSpace) => void;
  setCurrentFromTableColumn: (
    newCurrentFromTableColumn: TableColumnRef | null,
  ) => void;
  setCurrentFromUnit: (newCurrentFromUnit: CoordinateSpace) => void;
  setCurrentGroupByTableColumn: (
    newCurrentGroupByTableColumn: TableColumnRef | null,
  ) => void;
  setCurrentGroupByMap: (newCurrentGroupByMap: string | null) => void;
  setCurrentGroupByUnit: (newCurrentGroupByUnit: CoordinateSpace) => void;
};
