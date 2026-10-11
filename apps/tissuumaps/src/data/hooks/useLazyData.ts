import { useCallback } from "react";

import type {
  Data,
  DataObject,
  DataSource,
  ImageData,
  LabelsData,
  PointsData,
  ProgressCallback,
  ShapesData,
  TableData,
} from "@tissuumaps/core";

import { useProjectStore } from "@/stores/project";

import {
  useImageDataLoader,
  useLabelsDataLoader,
  usePointsDataLoader,
  useShapesDataLoader,
  useTableDataLoader,
} from "./useDataLoader";

/**
 * Provides a callback for loading the data of one of the project's images
 *
 * @param imageId - The ID of the image whose data to load, if any
 * @returns A callback that takes an optional abort signal and resolves to the
 * image's data, or to `null` if the ID does not identify an image of the
 * current project
 */
export function useLazyImageData(
  imageId: string | null,
): (options?: { signal?: AbortSignal }) => Promise<ImageData | null> {
  const image = useProjectStore(
    (state) => state.images.find((image) => image.id === imageId) ?? null,
  );
  const loadImage = useImageDataLoader();
  return useLazyData(image, loadImage);
}

/**
 * Provides a callback for loading the data of one of the project's labels
 *
 * @param labelsId - The ID of the labels whose data to load, if any
 * @returns A callback that takes an optional abort signal and resolves to the
 * labels' data, or to `null` if the ID does not identify labels of the current
 * project
 */
export function useLazyLabelsData(
  labelsId: string | null,
): (options?: { signal?: AbortSignal }) => Promise<LabelsData | null> {
  const labels = useProjectStore(
    (state) => state.labels.find((labels) => labels.id === labelsId) ?? null,
  );
  const loadLabels = useLabelsDataLoader();
  return useLazyData(labels, loadLabels);
}

/**
 * Provides a callback for loading the data of one of the project's points
 *
 * @param pointsId - The ID of the points whose data to load, if any
 * @returns A callback that takes an optional abort signal and resolves to the
 * points' data, or to `null` if the ID does not identify points of the current
 * project
 */
export function useLazyPointsData(
  pointsId: string | null,
): (options?: { signal?: AbortSignal }) => Promise<PointsData | null> {
  const points = useProjectStore(
    (state) => state.points.find((points) => points.id === pointsId) ?? null,
  );
  const loadPoints = usePointsDataLoader();
  return useLazyData(points, loadPoints);
}

/**
 * Provides a callback for loading the data of one of the project's shapes
 *
 * @param shapesId - The ID of the shapes whose data to load, if any
 * @returns A callback that takes an optional abort signal and resolves to the
 * shapes' data, or to `null` if the ID does not identify shapes of the current
 * project
 */
export function useLazyShapesData(
  shapesId: string | null,
): (options?: { signal?: AbortSignal }) => Promise<ShapesData | null> {
  const shapes = useProjectStore(
    (state) => state.shapes.find((shapes) => shapes.id === shapesId) ?? null,
  );
  const loadShapes = useShapesDataLoader();
  return useLazyData(shapes, loadShapes);
}

/**
 * Provides a callback for loading the data of one of the project's tables
 *
 * @param tableId - The ID of the table whose data to load, if any
 * @returns A callback that takes an optional abort signal and resolves to the
 * table's data, or to `null` if the ID does not identify a table of the
 * current project
 */
export function useLazyTableData(
  tableId: string | null,
): (options?: { signal?: AbortSignal }) => Promise<TableData | null> {
  const table = useProjectStore(
    (state) => state.tables.find((table) => table.id === tableId) ?? null,
  );
  const loadTable = useTableDataLoader();
  return useLazyData(table, loadTable);
}

/**
 * Shared implementation of the lazy data hooks above
 *
 * @param object - The object whose data to load, if any
 * @param loadObject - Loads the object's data through the responsible cache
 * @returns A callback that takes an optional abort signal and resolves to the
 * object's data, or to `null` if there is no object; it changes identity
 * whenever the object or the loader changes
 */
function useLazyData<
  TDataSource extends DataSource,
  TData extends Data,
  TDataObject extends DataObject<TDataSource>,
>(
  object: TDataObject | null,
  loadObject: (
    object: TDataObject,
    options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
  ) => Promise<TData>,
): (options?: { signal?: AbortSignal }) => Promise<TData | null> {
  return useCallback(
    async (options?: { signal?: AbortSignal }) => {
      const { signal } = options ?? {};
      signal?.throwIfAborted();
      return object !== null ? await loadObject(object, { signal }) : null;
    },
    [object, loadObject],
  );
}
