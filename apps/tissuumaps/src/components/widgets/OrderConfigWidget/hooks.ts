import { useMemo } from "react";

import type { OrderConfig, TableColumnRef } from "@tissuumaps/core";

import { useDraftState } from "@/hooks/useDraftState";

import type { OrderConfigSource, OrderConfigWidgetAdapter } from "./adapter";

type OrderConfigWidgetState = {
  currentSource: OrderConfigSource;
  currentFromTableColumn: TableColumnRef | null;
};

/**
 * Returns the widget state that shows the given order configuration
 *
 * @param config - The order configuration, or `undefined` for item order
 * @returns The widget state
 */
function configToState(
  config: OrderConfig | undefined,
): OrderConfigWidgetState {
  return {
    currentSource: config === undefined ? "none" : "from",
    currentFromTableColumn: config?.from ?? null,
  };
}

/**
 * Returns the order configuration that the widget state sets
 *
 * @param state - The widget state
 * @param config - The order configuration to update
 * @returns The updated order configuration, `undefined` for item order, or
 * `null` if the current source is incomplete
 */
function stateToConfig(
  state: OrderConfigWidgetState,
  config: OrderConfig | undefined,
): OrderConfig | undefined | null {
  switch (state.currentSource) {
    case "none":
      return undefined;
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
  }
}

export function useOrderConfigWidget(
  orderConfig: OrderConfig | undefined,
  onOrderConfigChange: (newOrderConfig: OrderConfig | undefined) => void,
  tableId: string | null,
): OrderConfigWidgetAdapter {
  const activeSource = orderConfig === undefined ? "none" : "from";
  const [state, setState] = useDraftState(
    orderConfig,
    onOrderConfigChange,
    configToState,
    stateToConfig,
  );

  const setters = useMemo(
    () => ({
      setCurrentSource: (currentSource: OrderConfigSource) =>
        setState((state) => ({ ...state, currentSource })),
      setCurrentFromTableColumn: (
        currentFromTableColumn: TableColumnRef | null,
      ) => setState((state) => ({ ...state, currentFromTableColumn })),
    }),
    [setState],
  );

  return {
    orderConfig,
    tableId,
    activeSource,
    ...state,
    ...setters,
  };
}
