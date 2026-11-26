import { useMemo } from "react";

import {
  type OpacityConfig,
  type TableColumnRef,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
} from "@tissuumaps/core";

import { useTableColumnValueRange } from "@/data/hooks/useTableColumnValueRange";
import { useDraftState } from "@/hooks/useDraftState";

import type {
  OpacityConfigSource,
  OpacityConfigWidgetAdapter,
} from "./adapter";

type OpacityConfigWidgetState = {
  currentSource: OpacityConfigSource;
  currentConstantValue: number;
  currentFromTableColumn: TableColumnRef | null;
  currentFromRangeMin: number | null;
  currentFromRangeMax: number | null;
  currentGroupByTableColumn: TableColumnRef | null;
  currentGroupByMap: string | null;
};

/**
 * Returns the widget state that shows the given opacity configuration
 *
 * @param config - The opacity configuration
 * @param defaultOpacity - The constant opacity if the configuration has none
 * @returns The widget state
 */
function configToState(
  config: OpacityConfig,
  defaultOpacity: number,
): OpacityConfigWidgetState {
  return {
    currentSource: getActiveConfigSource(config) ?? "constant",
    currentConstantValue: isConstantConfig(config)
      ? config.constant.value
      : defaultOpacity,
    currentFromTableColumn: isFromConfig(config) ? config.from : null,
    // starts at 0, so that the lowest values are not transparent by default
    currentFromRangeMin: isFromConfig(config)
      ? (config.from.range?.[0] ?? null)
      : 0,
    currentFromRangeMax: isFromConfig(config)
      ? (config.from.range?.[1] ?? null)
      : null,
    currentGroupByTableColumn: isGroupByConfig(config) ? config.groupBy : null,
    currentGroupByMap:
      isGroupByConfig(config) && config.groupBy.map !== undefined
        ? config.groupBy.map
        : null,
  };
}

/**
 * Returns the opacity configuration that the widget state sets
 *
 * @param state - The widget state
 * @param config - The opacity configuration to update
 * @returns The updated opacity configuration, or `null` if the current source
 * is incomplete
 */
function stateToConfig(
  state: OpacityConfigWidgetState,
  config: OpacityConfig,
): OpacityConfig | null {
  switch (state.currentSource) {
    case "constant":
      return {
        ...config,
        source: "constant",
        constant: {
          value: state.currentConstantValue,
        },
      };
    case "from":
      if (state.currentFromTableColumn === null) {
        return null;
      }
      return {
        ...config,
        source: "from",
        from: {
          ...state.currentFromTableColumn,
          range:
            state.currentFromRangeMin !== null ||
            state.currentFromRangeMax !== null
              ? [state.currentFromRangeMin, state.currentFromRangeMax]
              : undefined,
        },
      };
    case "groupBy":
      if (
        state.currentGroupByTableColumn === null ||
        state.currentGroupByMap === null
      ) {
        return null;
      }
      return {
        ...config,
        source: "groupBy",
        groupBy: {
          ...state.currentGroupByTableColumn,
          map: state.currentGroupByMap,
        },
      };
  }
}

export function useOpacityConfigWidget(
  opacityConfig: OpacityConfig,
  onOpacityConfigChange: (newOpacityConfig: OpacityConfig) => void,
  defaultOpacity: number,
  tableId: string | null,
): OpacityConfigWidgetAdapter {
  const activeSource = getActiveConfigSource(opacityConfig) ?? "constant";
  const [state, setState] = useDraftState(
    opacityConfig,
    onOpacityConfigChange,
    (config) => configToState(config, defaultOpacity),
    stateToConfig,
  );

  const fromColumnValueRange = useTableColumnValueRange(
    state.currentSource === "from" && state.currentFromTableColumn !== null
      ? (state.currentFromTableColumn.table ?? tableId)
      : null,
    state.currentFromTableColumn?.column ?? null,
  );

  const setters = useMemo(
    () => ({
      setCurrentSource: (currentSource: OpacityConfigSource) =>
        setState((state) => ({ ...state, currentSource })),
      setCurrentConstantValue: (currentConstantValue: number) =>
        setState((state) => ({ ...state, currentConstantValue })),
      setCurrentFromTableColumn: (
        currentFromTableColumn: TableColumnRef | null,
      ) => setState((state) => ({ ...state, currentFromTableColumn })),
      setCurrentFromRangeMin: (currentFromRangeMin: number | null) =>
        setState((state) => ({ ...state, currentFromRangeMin })),
      setCurrentFromRangeMax: (currentFromRangeMax: number | null) =>
        setState((state) => ({ ...state, currentFromRangeMax })),
      setCurrentGroupByTableColumn: (
        currentGroupByTableColumn: TableColumnRef | null,
      ) => setState((state) => ({ ...state, currentGroupByTableColumn })),
      setCurrentGroupByMap: (currentGroupByMap: string | null) =>
        setState((state) => ({ ...state, currentGroupByMap })),
    }),
    [setState],
  );

  return {
    opacityConfig,
    defaultOpacity,
    tableId,
    activeSource,
    fromColumnValueRange,
    ...state,
    ...setters,
  };
}
