import { useEffect, useRef, useState } from "react";

import {
  type Data,
  type DataProvider,
  type DataSource,
  type ImageDataSource,
  type LabelsDataSource,
  type PointsDataSource,
  type ShapesDataSource,
  SourceUtils,
  type TableDataSource,
} from "@tissuumaps/core";

import {
  addImageDataObject,
  addLabelsDataObject,
  addPointsDataObject,
  addShapesDataObject,
  addTableDataObject,
} from "@/data/io/dataObject";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

import type { AddDataObjectDialogParams } from ".";
import { useDialogContext } from "../DialogContext";

/** Returns the `addDataObject` function of the nearest `DialogProvider`. */
export function useAddDataObjectDialog() {
  return useDialogContext().addDataObject;
}

/** Returns the params of the dialog that adds images */
export function useAddImageDialogParams(): AddDataObjectDialogParams<ImageDataSource> {
  const dataProviders = useAppStore((state) => state.imageDataProviders);
  return {
    title: "Add image",
    dataProviders,
    withLayer: true,
    onAdd: addImageDataObject,
  };
}

/** Returns the params of the dialog that adds labels */
export function useAddLabelsDialogParams(): AddDataObjectDialogParams<LabelsDataSource> {
  const dataProviders = useAppStore((state) => state.labelsDataProviders);
  return {
    title: "Add labels",
    dataProviders,
    withLayer: true,
    onAdd: addLabelsDataObject,
  };
}

/** Returns the params of the dialog that adds points */
export function useAddPointsDialogParams(): AddDataObjectDialogParams<PointsDataSource> {
  const dataProviders = useAppStore((state) => state.pointsDataProviders);
  return {
    title: "Add points",
    dataProviders,
    withLayer: true,
    onAdd: addPointsDataObject,
  };
}

/** Returns the params of the dialog that adds shapes */
export function useAddShapesDialogParams(): AddDataObjectDialogParams<ShapesDataSource> {
  const dataProviders = useAppStore((state) => state.shapesDataProviders);
  return {
    title: "Add shapes",
    dataProviders,
    withLayer: true,
    onAdd: addShapesDataObject,
  };
}

/** Returns the params of the dialog that adds tables */
export function useAddTableDialogParams(): AddDataObjectDialogParams<TableDataSource> {
  const dataProviders = useAppStore((state) => state.tableDataProviders);
  return {
    title: "Add table",
    dataProviders,
    onAdd: (name, _layerId, origSource, preparedDataSource) =>
      addTableDataObject(name, origSource, preparedDataSource),
  };
}

/**
 * Inspects the sources entered in an add data object dialog
 *
 * `inspect` asks the data providers whether they support a source (see
 * `DataProvider.supports`), then lets `selectType` pick a type from the first
 * supporting one, and reads the name with that type's data provider (see
 * `DataProvider.readName`), falling back to the source's file name. It resolves
 * to that type and name, or to `undefined` if the source is missing or invalid,
 * or if the inspection was cancelled.
 *
 * One source is inspected at a time: a new inspection, `cancel` and unmounting
 * abort the pending one, and `cancel` also forgets the last support results.
 *
 * @param dataProviders - The data providers to ask, by data source type
 * @returns Whether each data provider with a `supports` method supports the
 * last inspected source (`null` if none was inspected), whether a source is
 * being inspected, `inspect` and `cancel`
 */
export function useSourceInspection<TDataSource extends DataSource>(
  dataProviders: Map<string, DataProvider<TDataSource, Data>>,
) {
  const workspace = useAppStore((state) => state.workspace);
  const projectSource = useProjectStore((state) => state.source);

  const [supportByType, setSupportByType] = useState<Map<
    string,
    boolean
  > | null>(null);
  const [isInspecting, setInspecting] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  useEffect(() => () => abortControllerRef.current?.abort(), []);

  const cancel = () => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setInspecting(false);
    setSupportByType(null);
  };

  const inspectSource = async (
    source: string,
    selectType: (firstSupportedType: string | undefined) => string,
    signal: AbortSignal,
  ) => {
    let normalizedSource: string;
    try {
      normalizedSource = SourceUtils.normalizeSource(
        source,
        workspace,
        projectSource,
      );
    } catch {
      setSupportByType(new Map());
      return undefined;
    }
    const support = await Promise.all(
      Array.from(dataProviders)
        .filter(([, dataProvider]) => dataProvider.supports !== undefined)
        .map(async ([type, dataProvider]) => {
          let isSupported = false;
          try {
            isSupported =
              (await dataProvider.supports?.(normalizedSource, workspace, {
                signal,
              })) ?? false;
          } catch {
            // treated as unsupported
          }
          return [type, isSupported] as const;
        }),
    );
    if (signal.aborted) {
      return undefined;
    }
    const supportByType = new Map(support);
    setSupportByType(supportByType);
    const type = selectType(
      Array.from(dataProviders.keys()).find(
        (type) => supportByType.get(type) === true,
      ),
    );
    let name: string | undefined;
    try {
      name = await dataProviders
        .get(type)
        ?.readName?.(normalizedSource, workspace, { signal });
    } catch {
      // the file name is used instead, unless the inspection was cancelled
    }
    if (signal.aborted) {
      return undefined;
    }
    return { type, name: name ?? SourceUtils.getStem(normalizedSource) };
  };

  const inspect = async (
    source: string | undefined,
    selectType: (firstSupportedType: string | undefined) => string,
  ) => {
    cancel();
    if (source === undefined) {
      return undefined;
    }
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    setInspecting(true);
    try {
      return await inspectSource(source, selectType, abortController.signal);
    } finally {
      // a cancelled inspection must not end a newer one
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null;
        setInspecting(false);
      }
    }
  };

  return { supportByType, isInspecting, inspect, cancel };
}
