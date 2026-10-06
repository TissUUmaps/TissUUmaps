import { useCallback } from "react";

import type {
  Image,
  ImageData,
  Labels,
  LabelsData,
  Points,
  PointsData,
  ProgressCallback,
  Shapes,
  ShapesData,
  Table,
  TableData,
} from "@tissuumaps/core";

import {
  imageDataCache,
  labelsDataCache,
  pointsDataCache,
  shapesDataCache,
  tableDataCache,
} from "@/data/cache";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

/**
 * Provides a callback for loading an image's data through the image data cache
 *
 * The callback is bound to the current workspace, the project source and the
 * registered image data providers, and changes identity whenever any of those
 * change.
 *
 * @returns A callback that takes an image and an optional abort signal and
 * progress callback, and resolves to the image's data
 */
export function useImageDataLoader(): (
  image: Image,
  options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
) => Promise<ImageData> {
  const workspace = useAppStore((state) => state.workspace);
  const projectSource = useProjectStore((state) => state.source);
  const dataProviders = useAppStore((state) => state.imageDataProviders);
  return useCallback(
    async (image, options) => {
      const { signal, onProgress } = options ?? {};
      signal?.throwIfAborted();
      return await imageDataCache.load(
        image,
        { workspace, projectSource, dataProviders },
        { signal, onProgress },
      );
    },
    [workspace, projectSource, dataProviders],
  );
}

/**
 * Provides a callback for loading labels' data through the labels data cache
 *
 * The callback is bound to the current workspace, the project source, the
 * registered labels data providers, and the project's tables and their data
 * providers, and changes identity whenever any of those change.
 *
 * @returns A callback that takes labels and an optional abort signal and
 * progress callback, and resolves to the labels' data
 */
export function useLabelsDataLoader(): (
  labels: Labels,
  options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
) => Promise<LabelsData> {
  const workspace = useAppStore((state) => state.workspace);
  const projectSource = useProjectStore((state) => state.source);
  const dataProviders = useAppStore((state) => state.labelsDataProviders);
  const tableDataProviders = useAppStore((state) => state.tableDataProviders);
  const tables = useProjectStore((state) => state.tables);
  return useCallback(
    async (labels, options) => {
      const { signal, onProgress } = options ?? {};
      signal?.throwIfAborted();
      return await labelsDataCache.load(
        labels,
        { workspace, projectSource, dataProviders, tables, tableDataProviders },
        { signal, onProgress },
      );
    },
    [workspace, projectSource, dataProviders, tables, tableDataProviders],
  );
}

/**
 * Provides a callback for loading points' data through the points data cache
 *
 * The callback is bound to the current workspace, the project source, the
 * registered points data providers, and the project's tables and their data
 * providers, and changes identity whenever any of those change.
 *
 * @returns A callback that takes points and an optional abort signal and
 * progress callback, and resolves to the points' data
 */
export function usePointsDataLoader(): (
  points: Points,
  options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
) => Promise<PointsData> {
  const workspace = useAppStore((state) => state.workspace);
  const projectSource = useProjectStore((state) => state.source);
  const dataProviders = useAppStore((state) => state.pointsDataProviders);
  const tableDataProviders = useAppStore((state) => state.tableDataProviders);
  const tables = useProjectStore((state) => state.tables);
  return useCallback(
    async (points, options) => {
      const { signal, onProgress } = options ?? {};
      signal?.throwIfAborted();
      return await pointsDataCache.load(
        points,
        { workspace, projectSource, dataProviders, tables, tableDataProviders },
        { signal, onProgress },
      );
    },
    [workspace, projectSource, dataProviders, tables, tableDataProviders],
  );
}

/**
 * Provides a callback for loading shapes' data through the shapes data cache
 *
 * The callback is bound to the current workspace, the project source, the
 * registered shapes data providers, and the project's tables and their data
 * providers, and changes identity whenever any of those change.
 *
 * @returns A callback that takes shapes and an optional abort signal and
 * progress callback, and resolves to the shapes' data
 */
export function useShapesDataLoader(): (
  shapes: Shapes,
  options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
) => Promise<ShapesData> {
  const workspace = useAppStore((state) => state.workspace);
  const projectSource = useProjectStore((state) => state.source);
  const dataProviders = useAppStore((state) => state.shapesDataProviders);
  const tableDataProviders = useAppStore((state) => state.tableDataProviders);
  const tables = useProjectStore((state) => state.tables);
  return useCallback(
    async (shapes, options) => {
      const { signal, onProgress } = options ?? {};
      signal?.throwIfAborted();
      return await shapesDataCache.load(
        shapes,
        { workspace, projectSource, dataProviders, tables, tableDataProviders },
        { signal, onProgress },
      );
    },
    [workspace, projectSource, dataProviders, tables, tableDataProviders],
  );
}

/**
 * Provides a callback for loading a table's data through the table data cache
 *
 * The callback is bound to the current workspace, the project source and the
 * registered table data providers, and changes identity whenever any of those
 * change.
 *
 * @returns A callback that takes a table and an optional abort signal and
 * progress callback, and resolves to the table's data
 */
export function useTableDataLoader(): (
  table: Table,
  options?: { signal?: AbortSignal; onProgress?: ProgressCallback },
) => Promise<TableData> {
  const workspace = useAppStore((state) => state.workspace);
  const projectSource = useProjectStore((state) => state.source);
  const dataProviders = useAppStore((state) => state.tableDataProviders);
  return useCallback(
    async (table, options) => {
      const { signal, onProgress } = options ?? {};
      signal?.throwIfAborted();
      return await tableDataCache.load(
        table,
        { workspace, projectSource, dataProviders },
        { signal, onProgress },
      );
    },
    [workspace, projectSource, dataProviders],
  );
}
