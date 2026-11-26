import type {
  AppStoreState,
  DataRef,
  ImageData,
  ImageDataSource,
  LabelsData,
  LabelsDataSource,
  PointsData,
  PointsDataSource,
  ProjectStoreState,
  ShapesData,
  ShapesDataSource,
  TableData,
  TableDataSource,
} from "@tissuumaps/core";

import { appStore } from "@/stores/app";
import { dataStore } from "@/stores/data";
import { projectStore } from "@/stores/project";

import { AnnotatedDataCache, DataCache } from "./DataCache";
import { ImageDataWrapper } from "./wrappers/ImageDataWrapper";
import { LabelsDataWrapper } from "./wrappers/LabelsDataWrapper";
import { PointsDataWrapper } from "./wrappers/PointsDataWrapper";
import { ShapesDataWrapper } from "./wrappers/ShapesDataWrapper";
import { TableDataWrapper } from "./wrappers/TableDataWrapper";

/** Caches the data of the project's tables, publishing it to the data store */
export const tableDataCache = new DataCache<TableDataSource, TableData>(
  (data) => new TableDataWrapper(data),
  {
    onObjectDataRefsChanged: (changedTableDataRefs) =>
      dataStore.setState((draft) => {
        for (const [tableId, newDataRef] of changedTableDataRefs) {
          draft.tableDataRefs.set(tableId, newDataRef);
        }
      }),
    onObjectDataRefsRemoved: (removedTableIds) =>
      dataStore.setState((draft) => {
        for (const tableId of removedTableIds) {
          draft.tableDataRefs.delete(tableId);
        }
      }),
  },
);

/** Caches the data of the project's images, publishing it to the data store */
export const imageDataCache = new DataCache<ImageDataSource, ImageData>(
  (data) => new ImageDataWrapper(data),
  {
    onObjectDataRefsChanged: (changedImageDataRefs) =>
      dataStore.setState((draft) => {
        for (const [imageId, newDataRef] of changedImageDataRefs) {
          draft.imageDataRefs.set(imageId, newDataRef);
        }
      }),
    onObjectDataRefsRemoved: (removedImageIds) =>
      dataStore.setState((draft) => {
        for (const imageId of removedImageIds) {
          draft.imageDataRefs.delete(imageId);
        }
      }),
  },
);

/** Caches the data of the project's labels, publishing it to the data store */
export const labelsDataCache = new AnnotatedDataCache<
  LabelsDataSource,
  LabelsData
>((data) => new LabelsDataWrapper(data), tableDataCache, {
  onObjectDataRefsChanged: (changedLabelsDataRefs) =>
    dataStore.setState((draft) => {
      for (const [labelsId, newDataRef] of changedLabelsDataRefs) {
        draft.labelsDataRefs.set(labelsId, newDataRef);
      }
    }),
  onObjectDataRefsRemoved: (removedLabelsIds) =>
    dataStore.setState((draft) => {
      for (const labelsId of removedLabelsIds) {
        draft.labelsDataRefs.delete(labelsId);
      }
    }),
});

/** Caches the data of the project's points, publishing it to the data store */
export const pointsDataCache = new AnnotatedDataCache<
  PointsDataSource,
  PointsData
>((data) => new PointsDataWrapper(data), tableDataCache, {
  onObjectDataRefsChanged: (changedPointsDataRefs) =>
    dataStore.setState((draft) => {
      for (const [pointsId, newDataRef] of changedPointsDataRefs) {
        draft.pointsDataRefs.set(pointsId, newDataRef);
      }
    }),
  onObjectDataRefsRemoved: (removedPointsIds) =>
    dataStore.setState((draft) => {
      for (const pointsId of removedPointsIds) {
        draft.pointsDataRefs.delete(pointsId);
      }
    }),
});

/** Caches the data of the project's shapes, publishing it to the data store */
export const shapesDataCache = new AnnotatedDataCache<
  ShapesDataSource,
  ShapesData
>((data) => new ShapesDataWrapper(data), tableDataCache, {
  onObjectDataRefsChanged: (changedShapesDataRefs) =>
    dataStore.setState((draft) => {
      for (const [shapesId, newDataRef] of changedShapesDataRefs) {
        draft.shapesDataRefs.set(shapesId, newDataRef);
      }
    }),
  onObjectDataRefsRemoved: (removedShapesIds) =>
    dataStore.setState((draft) => {
      for (const shapesId of removedShapesIds) {
        draft.shapesDataRefs.delete(shapesId);
      }
    }),
});

/**
 * Starts keeping the data caches in sync with the app and the project store
 *
 * Whenever the workspace, the project source, a data provider or one of the
 * project's objects changes, the caches release the data that is no longer
 * referenced or no longer valid, and the data store is updated accordingly.
 *
 * @returns A callback that stops the synchronization and releases all cached
 * data
 */
export function startDataCaches(): () => void {
  sanitizeDataCaches({ init: true });
  const unsubscribeAppStoreListener = appStore.subscribe(
    (appState, prevAppState) => {
      sanitizeDataCaches({ appState, prevAppState });
    },
  );
  const unsubscribeProjectStoreListener = projectStore.subscribe(
    (projectState, prevProjectState) => {
      sanitizeDataCaches({ projectState, prevProjectState });
    },
  );
  return () => {
    unsubscribeAppStoreListener();
    unsubscribeProjectStoreListener();
    sanitizeDataCaches({ cleanup: true });
  };
}

/**
 * Releases the data that the given state does not reference anymore, and
 * updates the data store accordingly
 *
 * Each cache is only sanitized if the state it depends on has changed with
 * respect to the given previous state, when initializing, or when cleaning up;
 * the annotated data caches are additionally sanitized whenever the table data
 * cache was.
 *
 * @param options - Whether the caches are being initialized (`init`) or emptied
 * (`cleanup`), the app and project state to sanitize for, defaulting to the
 * stores' current state, and the previous app and project state, if any
 */
function sanitizeDataCaches(options?: {
  init?: boolean;
  cleanup?: boolean;
  appState?: AppStoreState;
  projectState?: ProjectStoreState;
  prevAppState?: AppStoreState;
  prevProjectState?: ProjectStoreState;
}): void {
  const {
    init = false,
    cleanup = false,
    appState = appStore.getState(),
    projectState = projectStore.getState(),
    prevAppState,
    prevProjectState,
  } = options ?? {};

  const workspaceChanged =
    prevAppState !== undefined && appState.workspace !== prevAppState.workspace;
  const projectSourceChanged =
    prevProjectState !== undefined &&
    projectState.source !== prevProjectState.source;

  let newTableDataRefs: Map<string, DataRef<TableData>> | undefined;
  if (
    init ||
    cleanup ||
    workspaceChanged ||
    projectSourceChanged ||
    (prevProjectState !== undefined &&
      projectState.tables !== prevProjectState.tables) ||
    (prevAppState !== undefined &&
      appState.tableDataProviders !== prevAppState.tableDataProviders)
  ) {
    newTableDataRefs = tableDataCache.retainOnly(
      cleanup ? [] : projectState.tables,
      {
        workspace: appState.workspace,
        projectSource: projectState.source,
        dataProviders: appState.tableDataProviders,
      },
    );
  }

  let newImageDataRefs: Map<string, DataRef<ImageData>> | undefined;
  if (
    init ||
    cleanup ||
    workspaceChanged ||
    projectSourceChanged ||
    (prevProjectState !== undefined &&
      projectState.images !== prevProjectState.images) ||
    (prevAppState !== undefined &&
      appState.imageDataProviders !== prevAppState.imageDataProviders)
  ) {
    newImageDataRefs = imageDataCache.retainOnly(
      cleanup ? [] : projectState.images,
      {
        workspace: appState.workspace,
        projectSource: projectState.source,
        dataProviders: appState.imageDataProviders,
      },
    );
  }

  let newLabelsDataRefs: Map<string, DataRef<LabelsData>> | undefined;
  if (
    init ||
    cleanup ||
    workspaceChanged ||
    projectSourceChanged ||
    newTableDataRefs !== undefined ||
    (prevProjectState !== undefined &&
      projectState.labels !== prevProjectState.labels) ||
    (prevAppState !== undefined &&
      appState.labelsDataProviders !== prevAppState.labelsDataProviders)
  ) {
    newLabelsDataRefs = labelsDataCache.retainOnly(
      cleanup ? [] : projectState.labels,
      {
        workspace: appState.workspace,
        projectSource: projectState.source,
        dataProviders: appState.labelsDataProviders,
        tables: projectState.tables,
        tableDataProviders: appState.tableDataProviders,
      },
    );
  }

  let newPointsDataRefs: Map<string, DataRef<PointsData>> | undefined;
  if (
    init ||
    cleanup ||
    workspaceChanged ||
    projectSourceChanged ||
    newTableDataRefs !== undefined ||
    (prevProjectState !== undefined &&
      projectState.points !== prevProjectState.points) ||
    (prevAppState !== undefined &&
      appState.pointsDataProviders !== prevAppState.pointsDataProviders)
  ) {
    newPointsDataRefs = pointsDataCache.retainOnly(
      cleanup ? [] : projectState.points,
      {
        workspace: appState.workspace,
        projectSource: projectState.source,
        dataProviders: appState.pointsDataProviders,
        tables: projectState.tables,
        tableDataProviders: appState.tableDataProviders,
      },
    );
  }

  let newShapesDataRefs: Map<string, DataRef<ShapesData>> | undefined;
  if (
    init ||
    cleanup ||
    workspaceChanged ||
    projectSourceChanged ||
    newTableDataRefs !== undefined ||
    (prevProjectState !== undefined &&
      projectState.shapes !== prevProjectState.shapes) ||
    (prevAppState !== undefined &&
      appState.shapesDataProviders !== prevAppState.shapesDataProviders)
  ) {
    newShapesDataRefs = shapesDataCache.retainOnly(
      cleanup ? [] : projectState.shapes,
      {
        workspace: appState.workspace,
        projectSource: projectState.source,
        dataProviders: appState.shapesDataProviders,
        tables: projectState.tables,
        tableDataProviders: appState.tableDataProviders,
      },
    );
  }

  dataStore.setState((draft) => {
    if (newTableDataRefs !== undefined) {
      for (const tableId of [...draft.tableDataRefs.keys()]) {
        if (!newTableDataRefs.has(tableId)) {
          draft.tableDataRefs.delete(tableId);
        }
      }
      for (const [tableId, newDataRef] of newTableDataRefs) {
        draft.tableDataRefs.set(tableId, newDataRef);
      }
    }
    if (newImageDataRefs !== undefined) {
      for (const imageId of [...draft.imageDataRefs.keys()]) {
        if (!newImageDataRefs.has(imageId)) {
          draft.imageDataRefs.delete(imageId);
        }
      }
      for (const [imageId, newDataRef] of newImageDataRefs) {
        draft.imageDataRefs.set(imageId, newDataRef);
      }
    }
    if (newLabelsDataRefs !== undefined) {
      for (const labelsId of [...draft.labelsDataRefs.keys()]) {
        if (!newLabelsDataRefs.has(labelsId)) {
          draft.labelsDataRefs.delete(labelsId);
        }
      }
      for (const [labelsId, newDataRef] of newLabelsDataRefs) {
        draft.labelsDataRefs.set(labelsId, newDataRef);
      }
    }
    if (newPointsDataRefs !== undefined) {
      for (const pointsId of [...draft.pointsDataRefs.keys()]) {
        if (!newPointsDataRefs.has(pointsId)) {
          draft.pointsDataRefs.delete(pointsId);
        }
      }
      for (const [pointsId, newDataRef] of newPointsDataRefs) {
        draft.pointsDataRefs.set(pointsId, newDataRef);
      }
    }
    if (newShapesDataRefs !== undefined) {
      for (const shapesId of [...draft.shapesDataRefs.keys()]) {
        if (!newShapesDataRefs.has(shapesId)) {
          draft.shapesDataRefs.delete(shapesId);
        }
      }
      for (const [shapesId, newDataRef] of newShapesDataRefs) {
        draft.shapesDataRefs.set(shapesId, newDataRef);
      }
    }
  });
}
