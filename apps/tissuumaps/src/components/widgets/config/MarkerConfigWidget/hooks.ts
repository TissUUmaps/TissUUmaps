import { useMemo } from "react";

import {
  type Marker,
  type MarkerConfig,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
} from "@tissuumaps/core";

import { useConfigWidgetState } from "../useConfigWidgetState";
import type { MarkerConfigSource, MarkerConfigWidgetAdapter } from "./adapter";

type MarkerConfigWidgetState = {
  currentSource: MarkerConfigSource;
  currentConstantValue: Marker;
  currentFromColumn: string | null;
  currentGroupByColumn: string | null;
  currentGroupByMap: string | null;
};

/**
 * Returns the widget state that shows the given marker configuration
 *
 * @param config - The marker configuration
 * @param defaultMarker - The constant marker if the configuration has none
 * @returns The widget state
 */
function configToState(
  config: MarkerConfig,
  defaultMarker: Marker,
): MarkerConfigWidgetState {
  return {
    currentSource: getActiveConfigSource(config) ?? "constant",
    currentConstantValue: isConstantConfig(config)
      ? config.constant.value
      : defaultMarker,
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
 * Returns the marker configuration that the widget state sets
 *
 * @param state - The widget state
 * @param config - The marker configuration to update
 * @returns The updated marker configuration, or `null` if the current source
 * is incomplete
 */
function stateToConfig(
  state: MarkerConfigWidgetState,
  config: MarkerConfig,
): MarkerConfig | null {
  switch (state.currentSource) {
    case "constant":
      return {
        ...config,
        source: "constant",
        constant: { value: state.currentConstantValue },
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
      if (state.currentGroupByColumn === null) {
        return null;
      }
      return {
        ...config,
        source: "groupBy",
        groupBy: {
          column: state.currentGroupByColumn,
          map: state.currentGroupByMap ?? undefined,
        },
      };
  }
}

export function useMarkerConfigWidget(
  markerConfig: MarkerConfig,
  onMarkerConfigChange: (newMarkerConfig: MarkerConfig) => void,
  defaultMarker: Marker,
  tableId: string | null,
): MarkerConfigWidgetAdapter {
  const activeSource = getActiveConfigSource(markerConfig) ?? "constant";
  const [state, setState] = useConfigWidgetState(
    markerConfig,
    onMarkerConfigChange,
    (config) => configToState(config, defaultMarker),
    stateToConfig,
  );

  const setters = useMemo(
    () => ({
      setCurrentSource: (currentSource: MarkerConfigSource) =>
        setState((state) => ({ ...state, currentSource })),
      setCurrentConstantValue: (currentConstantValue: Marker) =>
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
    markerConfig,
    defaultMarker,
    tableId,
    activeSource,
    ...state,
    ...setters,
  };
}
