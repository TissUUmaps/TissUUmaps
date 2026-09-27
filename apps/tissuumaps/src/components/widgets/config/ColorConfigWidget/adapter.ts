import type { Color, ColorConfig } from "@tissuumaps/core";

export type ColorConfigSource = Exclude<ColorConfig["source"], undefined>;

export type ColorConfigWidgetAdapter = {
  colorConfig: ColorConfig;
  defaultColor: Color;
  tableId: string | null;
  activeSource: ColorConfigSource;
  fromColumnValueRange: [number, number] | null;
  currentSource: ColorConfigSource;
  currentConstantValue: Color;
  currentFromColumn: string | null;
  currentFromRangeMin: number | null;
  currentFromRangeMax: number | null;
  currentFromPalette: string | null;
  currentGroupByColumn: string | null;
  currentGroupByPalette: string | null;
  currentGroupByMap: string | null;
  currentRandomPalette: string | null;
  currentRandomSeed: number | null;
  setCurrentSource: (newCurrentSource: ColorConfigSource) => void;
  setCurrentConstantValue: (newCurrentValue: Color) => void;
  setCurrentFromColumn: (newCurrentFromColumn: string | null) => void;
  setCurrentFromRangeMin: (newCurrentFromRangeMin: number | null) => void;
  setCurrentFromRangeMax: (newCurrentFromRangeMax: number | null) => void;
  setCurrentFromPalette: (newCurrentFromPalette: string | null) => void;
  setCurrentGroupByColumn: (newCurrentGroupByColumn: string | null) => void;
  setCurrentGroupByPalette: (newCurrentGroupByPalette: string | null) => void;
  setCurrentGroupByMap: (newCurrentGroupByMap: string | null) => void;
  setCurrentRandomPalette: (newCurrentRandomPalette: string | null) => void;
  setCurrentRandomSeed: (newCurrentRandomSeed: number | null) => void;
};
