import type { Draft } from "immer";
import { createStore, useStore } from "zustand";
import { devtools } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";

import {
  type ProjectStore,
  type ProjectStoreApi,
  type ProjectStoreState,
  createLayer,
  projectDefaults,
} from "@tissuumaps/core";

import "./zustand";

/**
 * The store holding the currently open project, and the URL it was loaded from
 *
 * Loading a project into this store, and saving it back out, is handled by
 * `@/data/io/project`. The project's URL is only ever set by loading a project,
 * and is never written back out.
 */
export const projectStore: ProjectStoreApi = createStore<ProjectStore>()(
  devtools(
    immer((set) => ({
      ...createInitialProjectStoreState(),
      setName: (name) => set({ name }),
      addLayer: (layer, index) =>
        set((draft) => {
          addCollectionItem(draft.layers, layer, index);
        }),
      addImage: (image, index) =>
        set((draft) => {
          addCollectionItem(draft.images, image, index);
        }),
      addLabels: (labels, index) =>
        set((draft) => {
          addCollectionItem(draft.labels, labels, index);
        }),
      addPoints: (points, index) =>
        set((draft) => {
          addCollectionItem(draft.points, points, index);
        }),
      addShapes: (shapes, index) =>
        set((draft) => {
          addCollectionItem(draft.shapes, shapes, index);
        }),
      addTable: (table, index) =>
        set((draft) => {
          addCollectionItem(draft.tables, table, index);
        }),
      addMarkerMap: (map) =>
        set((draft) => {
          addCollectionItem(draft.markerMaps, map);
        }),
      addSizeMap: (map) =>
        set((draft) => {
          addCollectionItem(draft.sizeMaps, map);
        }),
      addColorMap: (map) =>
        set((draft) => {
          addCollectionItem(draft.colorMaps, map);
        }),
      addVisibilityMap: (map) =>
        set((draft) => {
          addCollectionItem(draft.visibilityMaps, map);
        }),
      addOpacityMap: (map) =>
        set((draft) => {
          addCollectionItem(draft.opacityMaps, map);
        }),
      updateLayer: (layerId, updates) =>
        set((draft) => {
          updateCollectionItem(draft.layers, layerId, updates);
        }),
      updateImage: (imageId, updates) =>
        set((draft) => {
          updateCollectionItem(draft.images, imageId, updates);
        }),
      updateLabels: (labelsId, updates) =>
        set((draft) => {
          updateCollectionItem(draft.labels, labelsId, updates);
        }),
      updatePoints: (pointsId, updates) =>
        set((draft) => {
          updateCollectionItem(draft.points, pointsId, updates);
        }),
      updateShapes: (shapesId, updates) =>
        set((draft) => {
          updateCollectionItem(draft.shapes, shapesId, updates);
        }),
      updateTable: (tableId, updates) =>
        set((draft) => {
          updateCollectionItem(draft.tables, tableId, updates);
        }),
      updateMarkerMap: (mapId, updates) =>
        set((draft) => {
          updateCollectionItem(draft.markerMaps, mapId, updates);
        }),
      updateSizeMap: (mapId, updates) =>
        set((draft) => {
          updateCollectionItem(draft.sizeMaps, mapId, updates);
        }),
      updateColorMap: (mapId, updates) =>
        set((draft) => {
          updateCollectionItem(draft.colorMaps, mapId, updates);
        }),
      updateVisibilityMap: (mapId, updates) =>
        set((draft) => {
          updateCollectionItem(draft.visibilityMaps, mapId, updates);
        }),
      updateOpacityMap: (mapId, updates) =>
        set((draft) => {
          updateCollectionItem(draft.opacityMaps, mapId, updates);
        }),
      moveLayer: (layerId, newIndex) =>
        set((draft) => {
          moveCollectionItem(draft.layers, layerId, newIndex);
        }),
      moveImage: (imageId, newIndex) =>
        set((draft) => {
          moveCollectionItem(draft.images, imageId, newIndex);
        }),
      moveLabels: (labelsId, newIndex) =>
        set((draft) => {
          moveCollectionItem(draft.labels, labelsId, newIndex);
        }),
      movePoints: (pointsId, newIndex) =>
        set((draft) => {
          moveCollectionItem(draft.points, pointsId, newIndex);
        }),
      moveShapes: (shapesId, newIndex) =>
        set((draft) => {
          moveCollectionItem(draft.shapes, shapesId, newIndex);
        }),
      moveTable: (tableId, newIndex) =>
        set((draft) => {
          moveCollectionItem(draft.tables, tableId, newIndex);
        }),
      deleteLayer: (layerId) =>
        set((draft) => {
          deleteCollectionItem(draft.layers, layerId);
        }),
      deleteImage: (imageId) =>
        set((draft) => {
          deleteCollectionItem(draft.images, imageId);
        }),
      deleteLabels: (labelsId) =>
        set((draft) => {
          deleteCollectionItem(draft.labels, labelsId);
        }),
      deletePoints: (pointsId) =>
        set((draft) => {
          deleteCollectionItem(draft.points, pointsId);
        }),
      deleteShapes: (shapesId) =>
        set((draft) => {
          deleteCollectionItem(draft.shapes, shapesId);
        }),
      deleteTable: (tableId) =>
        set((draft) => {
          deleteCollectionItem(draft.tables, tableId);
        }),
      deleteMarkerMap: (mapId) =>
        set((draft) => {
          deleteCollectionItem(draft.markerMaps, mapId);
        }),
      deleteSizeMap: (mapId) =>
        set((draft) => {
          deleteCollectionItem(draft.sizeMaps, mapId);
        }),
      deleteColorMap: (mapId) =>
        set((draft) => {
          deleteCollectionItem(draft.colorMaps, mapId);
        }),
      deleteVisibilityMap: (mapId) =>
        set((draft) => {
          deleteCollectionItem(draft.visibilityMaps, mapId);
        }),
      deleteOpacityMap: (mapId) =>
        set((draft) => {
          deleteCollectionItem(draft.opacityMaps, mapId);
        }),
      clearLayers: () => set({ layers: [] }),
      clearImages: () => set({ images: [] }),
      clearLabels: () => set({ labels: [] }),
      clearPoints: () => set({ points: [] }),
      clearShapes: () => set({ shapes: [] }),
      clearTables: () => set({ tables: [] }),
      clearMarkerMaps: () => set({ markerMaps: [] }),
      clearSizeMaps: () => set({ sizeMaps: [] }),
      clearColorMaps: () => set({ colorMaps: [] }),
      clearVisibilityMaps: () => set({ visibilityMaps: [] }),
      clearOpacityMaps: () => set({ opacityMaps: [] }),
      setOSOptions: (osOptions) => set({ osOptions }),
      setGLOptions: (glOptions) => set({ glOptions }),
      clear: () => set(createInitialProjectStoreState()),
    })),
    { name: "project", enabled: import.meta.env.DEV },
  ),
);

/**
 * Subscribes a component to a part of the {@link projectStore}
 *
 * @param selector - Selects the part of the store state to subscribe to
 * @returns The selected value, re-rendering the component whenever it changes
 */
export function useProjectStore<T>(selector: (state: ProjectStore) => T): T {
  return useStore(projectStore, selector);
}

/**
 * Creates the state of a new, empty project with a single default layer, which
 * was not loaded from a URL
 */
function createInitialProjectStoreState(): ProjectStoreState {
  return {
    ...structuredClone(projectDefaults),
    name: "New project",
    layers: [createLayer({ id: crypto.randomUUID(), name: "Default" })],
    images: [],
    labels: [],
    points: [],
    shapes: [],
    tables: [],
    source: null,
    instanceId: crypto.randomUUID(),
  };
}

/**
 * Inserts an item into one of the project's collections
 *
 * @param items - The collection to insert into
 * @param item - The item to insert
 * @param index - The index to insert at, defaulting to the end of the collection
 * @throws Error if an item with the same ID already exists, or if `index` is
 * out of bounds
 */
function addCollectionItem<T extends { id: string }>(
  items: Draft<T>[],
  item: T,
  index?: number,
): void {
  if (items.some((existingItem) => existingItem.id === item.id)) {
    throw new Error(`Item with ID ${item.id} already exists.`);
  }
  if (index !== undefined && (index < 0 || index > items.length)) {
    throw new Error(`Index ${index} out of bounds.`);
  }
  items.splice(index ?? items.length, 0, item as Draft<T>);
}

/**
 * Applies partial updates to an item of one of the project's collections
 *
 * @param items - The collection containing the item
 * @param itemId - The ID of the item to update
 * @param updates - The properties to overwrite on the item
 * @throws Error if no item with the given ID exists
 */
function updateCollectionItem<T extends { id: string }>(
  items: Draft<T>[],
  itemId: string,
  updates: Partial<Omit<T, "id">>,
): void {
  const index = items.findIndex((item) => item.id === itemId);
  if (index === -1) {
    throw new Error(`Item with ID ${itemId} not found.`);
  }
  Object.assign(items[index]!, updates);
}

/**
 * Moves an item of one of the project's collections to another index
 *
 * @param items - The collection containing the item
 * @param itemId - The ID of the item to move
 * @param newIndex - The index to move the item to
 * @throws Error if no item with the given ID exists, or if `newIndex` is out of
 * bounds
 */
function moveCollectionItem<T extends { id: string }>(
  items: Draft<T>[],
  itemId: string,
  newIndex: number,
): void {
  if (newIndex < 0 || newIndex >= items.length) {
    throw new Error(`Index ${newIndex} out of bounds.`);
  }
  const oldIndex = items.findIndex((item) => item.id === itemId);
  if (oldIndex === -1) {
    throw new Error(`Item with ID ${itemId} not found.`);
  }
  if (oldIndex !== newIndex) {
    const item = items.splice(oldIndex, 1)[0]!;
    items.splice(newIndex, 0, item);
  }
}

/**
 * Removes an item from one of the project's collections
 *
 * @param items - The collection to remove the item from
 * @param itemId - The ID of the item to remove
 * @throws Error if no item with the given ID exists
 */
function deleteCollectionItem<T extends { id: string }>(
  items: Draft<T>[],
  itemId: string,
): void {
  const index = items.findIndex((item) => item.id === itemId);
  if (index === -1) {
    throw new Error(`Item with ID ${itemId} not found.`);
  }
  items.splice(index, 1);
}
