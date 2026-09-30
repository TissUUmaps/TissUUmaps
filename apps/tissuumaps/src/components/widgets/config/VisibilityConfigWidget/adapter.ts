import type { TableColumnRef, VisibilityConfig } from "@tissuumaps/core";

export type VisibilityConfigSource = Exclude<
  VisibilityConfig["source"],
  undefined
>;

export type VisibilityConfigWidgetAdapter = {
  visibilityConfig: VisibilityConfig;
  defaultVisibility: boolean;
  tableId: string | null;
  activeSource: VisibilityConfigSource;
  currentSource: VisibilityConfigSource;
  currentConstantValue: boolean;
  currentFromTableColumn: TableColumnRef | null;
  currentGroupByTableColumn: TableColumnRef | null;
  currentGroupByMap: string | null;
  setCurrentSource: (newCurrentSource: VisibilityConfigSource) => void;
  setCurrentConstantValue: (newCurrentConstantValue: boolean) => void;
  setCurrentFromTableColumn: (
    newCurrentFromTableColumn: TableColumnRef | null,
  ) => void;
  setCurrentGroupByTableColumn: (
    newCurrentGroupByTableColumn: TableColumnRef | null,
  ) => void;
  setCurrentGroupByMap: (newCurrentGroupByMap: string | null) => void;
};
