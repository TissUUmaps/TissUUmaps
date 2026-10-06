import { useMemo } from "react";

import {
  type TableColumnRef,
  type VisibilityConfig,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
} from "@tissuumaps/core";

import { useDraftState } from "@/hooks/useDraftState";

import type {
  VisibilityConfigSource,
  VisibilityConfigWidgetAdapter,
} from "./adapter";

type VisibilityConfigWidgetState = {
  currentSource: VisibilityConfigSource;
  currentConstantValue: boolean;
  currentFromTableColumn: TableColumnRef | null;
  currentGroupByTableColumn: TableColumnRef | null;
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
    currentFromTableColumn: isFromConfig(config) ? config.from : null,
    currentGroupByTableColumn: isGroupByConfig(config) ? config.groupBy : null,
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
      if (state.currentFromTableColumn === null) {
        return null;
      }
      return {
        ...config,
        source: "from",
        from: {
          ...state.currentFromTableColumn,
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

export function useVisibilityConfigWidget(
  visibilityConfig: VisibilityConfig,
  onVisibilityConfigChange: (newVisibilityConfig: VisibilityConfig) => void,
  defaultVisibility: boolean,
  tableId: string | null,
): VisibilityConfigWidgetAdapter {
  const activeSource = getActiveConfigSource(visibilityConfig) ?? "constant";
  const [state, setState] = useDraftState(
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
      setCurrentFromTableColumn: (
        currentFromTableColumn: TableColumnRef | null,
      ) => setState((state) => ({ ...state, currentFromTableColumn })),
      setCurrentGroupByTableColumn: (
        currentGroupByTableColumn: TableColumnRef | null,
      ) => setState((state) => ({ ...state, currentGroupByTableColumn })),
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
