import { createStore, useStore } from "zustand";
import { devtools } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";

import type { AppStore, AppStoreApi, AppStoreState } from "@tissuumaps/core";

import "./zustand";

/**
 * The store holding application state that is not part of the project
 *
 * This comprises the open workspace, the current interaction mode, the hovered
 * channel preview, the registered data providers, and the registered plugins.
 * The plugins are written by the plugin registry, which owns their lifecycle,
 * rather than through an action.
 */
export const appStore: AppStoreApi = createStore<AppStore>()(
  devtools(
    immer((set) => ({
      ...createInitialAppStoreState(),
      setWorkspace: (workspace) => set({ workspace }),
      setInteractionMode: (interactionMode) => set({ interactionMode }),
      setImageChannelPreview: (imageChannelPreview) =>
        set({ imageChannelPreview }),
      setHighlightedItemGroup: (highlightedItemGroup) =>
        set({ highlightedItemGroup }),
      registerImageDataProvider: (type, dataProvider) =>
        set((draft) => {
          draft.imageDataProviders.set(type, dataProvider);
        }),
      registerLabelsDataProvider: (type, dataProvider) =>
        set((draft) => {
          draft.labelsDataProviders.set(type, dataProvider);
        }),
      registerPointsDataProvider: (type, dataProvider) =>
        set((draft) => {
          draft.pointsDataProviders.set(type, dataProvider);
        }),
      registerShapesDataProvider: (type, dataProvider) =>
        set((draft) => {
          draft.shapesDataProviders.set(type, dataProvider);
        }),
      registerTableDataProvider: (type, dataProvider) =>
        set((draft) => {
          draft.tableDataProviders.set(type, dataProvider);
        }),
    })),
    { name: "app", enabled: import.meta.env.DEV },
  ),
);

/**
 * Subscribes a component to a part of the {@link appStore}
 *
 * @param selector - Selects the part of the store state to subscribe to
 * @returns The selected value, re-rendering the component whenever it changes
 */
export function useAppStore<T>(selector: (state: AppStore) => T): T {
  return useStore(appStore, selector);
}

/**
 * Creates the initial {@link appStore} state, with nothing registered and no
 * workspace open
 */
function createInitialAppStoreState(): AppStoreState {
  return {
    workspace: null,
    interactionMode: "pan",
    imageChannelPreview: null,
    highlightedItemGroup: null,
    imageDataProviders: new Map(),
    labelsDataProviders: new Map(),
    pointsDataProviders: new Map(),
    shapesDataProviders: new Map(),
    tableDataProviders: new Map(),
    plugins: new Map(),
  };
}
