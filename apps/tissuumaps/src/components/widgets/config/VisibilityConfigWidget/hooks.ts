import { useMemo } from "react";

import {
  type VisibilityConfig,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
} from "@tissuumaps/core";

import { useConfigWidgetState } from "../useConfigWidgetState";
import type {
  VisibilityConfigSource,
  VisibilityConfigWidgetAdapter,
} from "./adapter";

type VisibilityConfigWidgetState = {
  currentSource: VisibilityConfigSource;
  currentConstantValue: boolean;
  currentFromColumn: string | null;
  currentGroupByColumn: string | null;
  currentGroupByMap: string | null;
};

/**
 * Returns the widget state that shows the given visibility configuration
 *
 * @param config - The visibility configuration
 * @param defaultVisibility - The constant visibility if the configuration has none
 * @returns The widget state
 */
function configToState(
  config: VisibilityConfig,
  defaultVisibility: boolean,
): VisibilityConfigWidgetState {
  return {
    currentSource: getActiveConfigSource(config) ?? "constant",
    currentConstantValue: isConstantConfig(config)
      ? config.constant.value
      : defaultVisibility,
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
 * Returns the visibility configuration that the widget state sets
 *
 * @param state - The widget state
 * @param config - The visibility configuration to update
 * @returns The updated visibility configuration, or `null` if the current source
 * is incomplete
 */
function stateToConfig(
  state: VisibilityConfigWidgetState,
  config: VisibilityConfig,
): VisibilityConfig | null {
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

export function useVisibilityConfigWidget(
  visibilityConfig: VisibilityConfig,
  onVisibilityConfigChange: (newVisibilityConfig: VisibilityConfig) => void,
  defaultVisibility: boolean,
  tableId: string | null,
): VisibilityConfigWidgetAdapter {
  const activeSource = getActiveConfigSource(visibilityConfig) ?? "constant";
  const [state, setState] = useConfigWidgetState(
    visibilityConfig,
    onVisibilityConfigChange,
    (config) => configToState(config, defaultVisibility),
    stateToConfig,
  );

  const setters = useMemo(
    () => ({
      setCurrentSource: (currentSource: VisibilityConfigSource) =>
        setState((state) => ({ ...state, currentSource })),
      setCurrentConstantValue: (currentConstantValue: boolean) =>
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
    visibilityConfig,
    defaultVisibility,
    tableId,
    activeSource,
    ...state,
    ...setters,
  };
}
