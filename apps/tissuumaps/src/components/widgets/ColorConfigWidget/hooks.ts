import { useEffect, useMemo, useState } from "react";

import {
  type Color,
  type ColorConfig,
  type TableColumnRef,
  TableUtils,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
  isRandomConfig,
} from "@tissuumaps/core";

import { useTableData } from "@/data/hooks/useData";
import { useDraftState } from "@/hooks/useDraftState";

import type { ColorConfigSource, ColorConfigWidgetAdapter } from "./adapter";

type ColorConfigWidgetState = {
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
};

/**
 * Returns the widget state that shows the given color configuration
 *
 * @param config - The color configuration
 * @param defaultColor - The constant color if the configuration has none
 * @returns The widget state
 */
function configToState(
  config: ColorConfig,
  defaultColor: Color,
): ColorConfigWidgetState {
  return {
    currentSource: getActiveConfigSource(config) ?? "constant",
    currentConstantValue: isConstantConfig(config)
      ? config.constant.value
      : defaultColor,
    currentFromTableColumn: isFromConfig(config) ? config.from : null,
    currentFromRangeMin:
      isFromConfig(config) && config.from.range !== undefined
        ? config.from.range[0]
        : null,
    currentFromRangeMax:
      isFromConfig(config) && config.from.range !== undefined
        ? config.from.range[1]
        : null,
    currentFromPalette: isFromConfig(config) ? config.from.palette : null,
    currentGroupByTableColumn: isGroupByConfig(config) ? config.groupBy : null,
    currentGroupByPalette:
      isGroupByConfig(config) && config.groupBy.palette !== undefined
        ? config.groupBy.palette
        : null,
    currentGroupByMap:
      isGroupByConfig(config) && config.groupBy.map !== undefined
        ? config.groupBy.map
        : null,
    currentRandomPalette: isRandomConfig(config) ? config.random.palette : null,
    currentRandomSeed:
      isRandomConfig(config) && config.random.seed !== undefined
        ? config.random.seed
        : null,
  };
}

/**
 * Returns the color configuration that the widget state sets
 *
 * @param state - The widget state
 * @param config - The color configuration to update
 * @returns The updated color configuration, or `null` if the current source
 * is incomplete
 */
function stateToConfig(
  state: ColorConfigWidgetState,
  config: ColorConfig,
): ColorConfig | null {
  switch (state.currentSource) {
    case "constant":
      return {
        ...config,
        source: "constant",
        constant: { value: state.currentConstantValue },
      };
    case "from":
      if (
        state.currentFromTableColumn === null ||
        state.currentFromPalette === null
      ) {
        return null;
      }
      return {
        ...config,
        source: "from",
        from: {
          ...state.currentFromTableColumn,
          range:
            state.currentFromRangeMin !== null &&
            state.currentFromRangeMax !== null
              ? [state.currentFromRangeMin, state.currentFromRangeMax]
              : undefined,
          palette: state.currentFromPalette,
        },
      };
    case "groupBy":
      if (
        state.currentGroupByTableColumn === null ||
        (state.currentGroupByPalette === null &&
          state.currentGroupByMap === null)
      ) {
        return null;
      }
      return {
        ...config,
        source: "groupBy",
        groupBy: {
          ...state.currentGroupByTableColumn,
          palette: state.currentGroupByPalette ?? undefined,
          map: state.currentGroupByMap ?? undefined,
        },
      };
    case "random":
      if (state.currentRandomPalette === null) {
        return null;
      }
      return {
        ...config,
        source: "random",
        random: {
          palette: state.currentRandomPalette,
          seed: state.currentRandomSeed ?? undefined,
        },
      };
  }
}

export function useColorConfigWidget(
  colorConfig: ColorConfig,
  onColorConfigChange: (newColorConfig: ColorConfig) => void,
  defaultColor: Color,
  tableId: string | null,
): ColorConfigWidgetAdapter {
  const activeSource = getActiveConfigSource(colorConfig) ?? "constant";
  const [state, setState] = useDraftState(
    colorConfig,
    onColorConfigChange,
    (config) => configToState(config, defaultColor),
    stateToConfig,
  );

  const tableData = useTableData(
    state.currentFromTableColumn?.table ?? tableId,
  );

  const [fromColumnValueRange, setFromColumnValueRange] = useState<
    [number, number] | null
  >(null);

  useEffect(() => {
    // clear the previous column's range synchronously, so that the placeholder
    // does not show a stale range while the new one is still loading
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setFromColumnValueRange(null);
    if (
      state.currentSource === "from" &&
      state.currentFromTableColumn !== null &&
      tableData !== null
    ) {
      const abortController = new AbortController();
      TableUtils.loadValueRange(
        tableData,
        state.currentFromTableColumn.column,
        { signal: abortController.signal },
      )
        .then((valueRange) => {
          if (!abortController.signal.aborted) {
            setFromColumnValueRange(valueRange ?? null);
          }
        })
        .catch((error) => {
          if (!abortController.signal.aborted) {
            console.error("Error loading table value range", error);
          }
        });
      return () => abortController.abort();
    }
  }, [tableData, state.currentSource, state.currentFromTableColumn]);

  const setters = useMemo(
    () => ({
      setCurrentSource: (currentSource: ColorConfigSource) =>
        setState((state) => ({ ...state, currentSource })),
      setCurrentConstantValue: (currentConstantValue: Color) =>
        setState((state) => ({ ...state, currentConstantValue })),
      setCurrentFromTableColumn: (
        currentFromTableColumn: TableColumnRef | null,
      ) => setState((state) => ({ ...state, currentFromTableColumn })),
      setCurrentFromRangeMin: (currentFromRangeMin: number | null) =>
        setState((state) => ({ ...state, currentFromRangeMin })),
      setCurrentFromRangeMax: (currentFromRangeMax: number | null) =>
        setState((state) => ({ ...state, currentFromRangeMax })),
      setCurrentFromPalette: (currentFromPalette: string | null) =>
        setState((state) => ({ ...state, currentFromPalette })),
      setCurrentGroupByTableColumn: (
        currentGroupByTableColumn: TableColumnRef | null,
      ) => setState((state) => ({ ...state, currentGroupByTableColumn })),
      setCurrentGroupByPalette: (currentGroupByPalette: string | null) =>
        setState((state) => ({ ...state, currentGroupByPalette })),
      setCurrentGroupByMap: (currentGroupByMap: string | null) =>
        setState((state) => ({ ...state, currentGroupByMap })),
      setCurrentRandomPalette: (currentRandomPalette: string | null) =>
        setState((state) => ({ ...state, currentRandomPalette })),
      setCurrentRandomSeed: (currentRandomSeed: number | null) =>
        setState((state) => ({ ...state, currentRandomSeed })),
    }),
    [setState],
  );

  return {
    colorConfig,
    defaultColor,
    tableId,
    activeSource,
    fromColumnValueRange,
    ...state,
    ...setters,
  };
}
