import type { Color, ColorConfig, TableColumnRef } from "@tissuumaps/core";

export type ColorConfigSource = Exclude<ColorConfig["source"], undefined>;

export type ColorConfigWidgetAdapter = {
  colorConfig: ColorConfig;
  defaultColor: Color;
  tableId: string | null;
  activeSource: ColorConfigSource;
  fromColumnValueRange: [number, number] | null;
  currentSource: ColorConfigSource;
  currentConstantValue: Color;
  currentFromTableColumn: TableColumnRef | null;
  currentFromRangeMin: number | null;
  currentFromRangeMax: number | null;
  currentFromPalette: string | null;
  currentGroupByTableColumn: TableColumnRef | null;
  currentGroupByPalette: string | null;
  currentGroupByMap: string | null;
  currentRandomPalette: string | null;
  currentRandomSeed: number | null;
  setCurrentSource: (newCurrentSource: ColorConfigSource) => void;
  setCurrentConstantValue: (newCurrentValue: Color) => void;
  setCurrentFromTableColumn: (
    newCurrentFromTableColumn: TableColumnRef | null,
  ) => void;
  setCurrentFromRangeMin: (newCurrentFromRangeMin: number | null) => void;
  setCurrentFromRangeMax: (newCurrentFromRangeMax: number | null) => void;
  setCurrentFromPalette: (newCurrentFromPalette: string | null) => void;
  setCurrentGroupByTableColumn: (
    newCurrentGroupByTableColumn: TableColumnRef | null,
  ) => void;
  setCurrentGroupByPalette: (newCurrentGroupByPalette: string | null) => void;
  setCurrentGroupByMap: (newCurrentGroupByMap: string | null) => void;
  setCurrentRandomPalette: (newCurrentRandomPalette: string | null) => void;
  setCurrentRandomSeed: (newCurrentRandomSeed: number | null) => void;
};
