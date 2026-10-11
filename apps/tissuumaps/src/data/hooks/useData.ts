import { useEffect } from "react";

import type {
  Data,
  DataObject,
  DataRef,
  DataSource,
  DataStore,
  ImageData,
  LabelsData,
  PointsData,
  ProgressCallback,
  ShapesData,
  TableData,
} from "@tissuumaps/core";

import { useDataStore } from "@/stores/data";
import { useProjectStore } from "@/stores/project";

import {
  useImageDataLoader,
  useLabelsDataLoader,
  usePointsDataLoader,
  useShapesDataLoader,
  useTableDataLoader,
} from "./useDataLoader";

/**
 * Loads and provides the data of one of the project's images
 *
 * @param imageId - The ID of the image whose data to provide, if any
 * @returns The image's data, or `null` while it is loading, if loading failed,
 * or if the ID does not identify an image of the current project
 */
export function useImageData(imageId: string | null): ImageData | null {
  const image = useProjectStore(
    (state) => state.images.find((image) => image.id === imageId) ?? null,
  );
  const loadImage = useImageDataLoader();
  return useData(image, loadImage, (state) => state.imageDataRefs);
}

/**
 * Loads and provides the data of one of the project's labels
 *
 * @param labelsId - The ID of the labels whose data to provide, if any
 * @returns The labels' data, or `null` while it is loading, if loading failed,
 * or if the ID does not identify labels of the current project
 */
export function useLabelsData(labelsId: string | null): LabelsData | null {
  const labels = useProjectStore(
    (state) => state.labels.find((labels) => labels.id === labelsId) ?? null,
  );
  const loadLabels = useLabelsDataLoader();
  return useData(labels, loadLabels, (state) => state.labelsDataRefs);
}

/**
 * Loads and provides the data of one of the project's points
 *
 * @param pointsId - The ID of the points whose data to provide, if any
 * @returns The points' data, or `null` while it is loading, if loading failed,
 * or if the ID does not identify points of the current project
 */
export function usePointsData(pointsId: string | null): PointsData | null {
  const points = useProjectStore(
    (state) => state.points.find((points) => points.id === pointsId) ?? null,
  );
  const loadPoints = usePointsDataLoader();
  return useData(points, loadPoints, (state) => state.pointsDataRefs);
}

/**
 * Loads and provides the data of one of the project's shapes
 *
 * @param shapesId - The ID of the shapes whose data to provide, if any
 * @returns The shapes' data, or `null` while it is loading, if loading failed,
 * or if the ID does not identify shapes of the current project
 */
export function useShapesData(shapesId: string | null): ShapesData | null {
  const shapes = useProjectStore(
    (state) => state.shapes.find((shapes) => shapes.id === shapesId) ?? null,
  );
  const loadShapes = useShapesDataLoader();
  return useData(shapes, loadShapes, (state) => state.shapesDataRefs);
}

/**
 * Loads and provides the data of one of the project's tables
 *
 * @param tableId - The ID of the table whose data to provide, if any
 * @returns The table's data, or `null` while it is loading, if loading failed,
 * or if the ID does not identify a table of the current project
 */
export function useTableData(tableId: string | null): TableData | null {
  const table = useProjectStore(
    (state) => state.tables.find((table) => table.id === tableId) ?? null,
  );
  const loadTable = useTableDataLoader();
  return useData(table, loadTable, (state) => state.tableDataRefs);
}

/**
 * Shared implementation of the data hooks above
 *
 * Requests the object's data whenever the object or the loader changes,
 * aborting the pending request on cleanup, and subscribes to the data store for
 * the result. Loading errors are logged; the hook keeps returning `null`.
 *
 * @param object - The object whose data to provide, if any
 * @param loadObject - Loads the object's data through the responsible cache
 * @param selectDataRefs - Selects the data references of the object's kind
 * @returns The object's data, or `null` unless it has been loaded
 */
function useData<
  TDataSource extends DataSource,
  TData extends Data,
  TDataObject extends DataObject<TDataSource>,
>(
  object: TDataObject | null,
  loadObject: (
    object: TDataObject,
    options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
  ) => Promise<TData>,
  selectDataRefs: (state: DataStore) => Map<string, DataRef<TData>>,
): TData | null {
  useEffect(() => {
    if (object !== null) {
      const abortController = new AbortController();
      loadObject(object, { signal: abortController.signal }).catch((error) => {
        if (!abortController.signal.aborted) {
          console.error(`Failed to load object with ID '${object.id}'`, error);
        }
      });
      return () => abortController.abort();
    }
  }, [object, loadObject]);

  return useDataStore((state) => {
    if (object !== null) {
      const dataRef = selectDataRefs(state).get(object.id);
      if (dataRef !== undefined && dataRef.status === "loaded") {
        return dataRef.data;
      }
    }
    return null;
  });
}
