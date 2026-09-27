import { useMemo } from "react";

import {
  type OpacityConfig,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
} from "@tissuumaps/core";

import { useConfigWidgetState } from "../useConfigWidgetState";
import type {
  OpacityConfigSource,
  OpacityConfigWidgetAdapter,
} from "./adapter";

type OpacityConfigWidgetState = {
  currentSource: OpacityConfigSource;
  currentConstantValue: number;
  currentFromColumn: string | null;
  currentGroupByColumn: string | null;
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
    currentFromColumn: isFromConfig(config) ? config.from.column : null,
    currentGroupByColumn: isGroupByConfig(config)
      ? config.groupBy.column
      : null,
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
      if (state.currentFromColumn === null) {
        return null;
      }
      return {
        ...config,
        source: "from",
        from: {
          column: state.currentFromColumn,
        },
      };
    case "groupBy":
      if (
        state.currentGroupByColumn === null ||
        state.currentGroupByMap === null
      ) {
        return null;
      }
      return {
        ...config,
        source: "groupBy",
        groupBy: {
          column: state.currentGroupByColumn,
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
  const [state, setState] = useConfigWidgetState(
    opacityConfig,
    onOpacityConfigChange,
    (config) => configToState(config, defaultOpacity),
    stateToConfig,
  );

  const setters = useMemo(
    () => ({
      setCurrentSource: (currentSource: OpacityConfigSource) =>
        setState((state) => ({ ...state, currentSource })),
      setCurrentConstantValue: (currentConstantValue: number) =>
        setState((state) => ({ ...state, currentConstantValue })),
      setCurrentFromColumn: (currentFromColumn: string | null) =>
        setState((state) => ({ ...state, currentFromColumn })),
      setCurrentGroupByColumn: (currentGroupByColumn: string | null) =>
        setState((state) => ({ ...state, currentGroupByColumn })),
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
    ...state,
    ...setters,
  };
}
