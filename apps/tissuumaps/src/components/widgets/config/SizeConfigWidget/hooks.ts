import { useMemo } from "react";

import {
  type CoordinateSpace,
  type SizeConfig,
  type TableColumnRef,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
} from "@tissuumaps/core";

import { useConfigWidgetState } from "../useConfigWidgetState";
import type { SizeConfigSource, SizeConfigWidgetAdapter } from "./adapter";

type SizeConfigWidgetState = {
  currentSource: SizeConfigSource;
  currentConstantValue: number;
  currentConstantUnit: CoordinateSpace;
  currentFromTableColumn: TableColumnRef | null;
  currentFromUnit: CoordinateSpace;
  currentGroupByTableColumn: TableColumnRef | null;
  currentGroupByMap: string | null;
  currentGroupByUnit: CoordinateSpace;
};

/**
 * Returns the widget state that shows the given size configuration
 *
 * @param config - The size configuration
 * @param defaultSize - The constant size if the configuration has none
 * @param defaultSizeUnit - The unit of each source that has none
 * @returns The widget state
 */
function configToState(
  config: SizeConfig,
  defaultSize: number,
  defaultSizeUnit: CoordinateSpace,
): SizeConfigWidgetState {
  return {
    currentSource: getActiveConfigSource(config) ?? "constant",
    currentConstantValue: isConstantConfig(config)
      ? config.constant.value
      : defaultSize,
    currentConstantUnit:
      isConstantConfig(config) && config.constant.unit !== undefined
        ? config.constant.unit
        : defaultSizeUnit,
    currentFromTableColumn: isFromConfig(config) ? config.from : null,
    currentFromUnit:
      isFromConfig(config) && config.from.unit !== undefined
        ? config.from.unit
        : defaultSizeUnit,
    currentGroupByTableColumn: isGroupByConfig(config) ? config.groupBy : null,
    currentGroupByMap:
      isGroupByConfig(config) && config.groupBy.map !== undefined
        ? config.groupBy.map
        : null,
    currentGroupByUnit:
      isGroupByConfig(config) && config.groupBy.unit !== undefined
        ? config.groupBy.unit
        : defaultSizeUnit,
  };
}

/**
 * Returns the size configuration that the widget state sets
 *
 * @param state - The widget state
 * @param config - The size configuration to update
 * @returns The updated size configuration, or `null` if the current source is
 * incomplete
 */
function stateToConfig(
  state: SizeConfigWidgetState,
  config: SizeConfig,
): SizeConfig | null {
  switch (state.currentSource) {
    case "constant":
      return {
        ...config,
        source: "constant",
        constant: {
          value: state.currentConstantValue,
          unit: state.currentConstantUnit,
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
          unit: state.currentFromUnit,
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
          unit: state.currentGroupByUnit,
        },
      };
  }
}

export function useSizeConfigWidget(
  sizeConfig: SizeConfig,
  onSizeConfigChange: (newSizeConfig: SizeConfig) => void,
  defaultSize: number,
  defaultSizeUnit: CoordinateSpace,
  tableId: string | null,
): SizeConfigWidgetAdapter {
  const activeSource = getActiveConfigSource(sizeConfig) ?? "constant";
  const [state, setState] = useConfigWidgetState(
    sizeConfig,
    onSizeConfigChange,
    (config) => configToState(config, defaultSize, defaultSizeUnit),
    stateToConfig,
  );

  const setters = useMemo(
    () => ({
      setCurrentSource: (currentSource: SizeConfigSource) =>
        setState((state) => ({ ...state, currentSource })),
      setCurrentConstantValue: (currentConstantValue: number) =>
        setState((state) => ({ ...state, currentConstantValue })),
      setCurrentConstantUnit: (currentConstantUnit: CoordinateSpace) =>
        setState((state) => ({ ...state, currentConstantUnit })),
      setCurrentFromTableColumn: (
        currentFromTableColumn: TableColumnRef | null,
      ) => setState((state) => ({ ...state, currentFromTableColumn })),
      setCurrentFromUnit: (currentFromUnit: CoordinateSpace) =>
        setState((state) => ({ ...state, currentFromUnit })),
      setCurrentGroupByTableColumn: (
        currentGroupByTableColumn: TableColumnRef | null,
      ) => setState((state) => ({ ...state, currentGroupByTableColumn })),
      setCurrentGroupByMap: (currentGroupByMap: string | null) =>
        setState((state) => ({ ...state, currentGroupByMap })),
      setCurrentGroupByUnit: (currentGroupByUnit: CoordinateSpace) =>
        setState((state) => ({ ...state, currentGroupByUnit })),
    }),
    [setState],
  );

  return {
    sizeConfig,
    defaultSize,
    defaultSizeUnit,
    tableId,
    activeSource,
    ...state,
    ...setters,
  };
}
