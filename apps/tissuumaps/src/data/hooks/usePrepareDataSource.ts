import { useEffect, useRef, useState } from "react";

import type { Data, DataProvider, DataSource } from "@tissuumaps/core";

import { useAlertDialog } from "@/components/dialogs/AlertDialog/hooks";
import { appStore } from "@/stores/app";
import { projectStore } from "@/stores/project";

/**
 * Prepares data sources before they are added or saved (see
 * `DataProvider.prepareDataSource`)
 *
 * One data source is prepared at a time, against the open workspace and the
 * project source. Preparing is aborted by `cancel` and when the component
 * unmounts, so that the data provider creates nothing more for it. A failure
 * is shown in an alert.
 *
 * `prepare` resolves to the prepared data source (the data source itself if
 * its data provider has nothing to prepare), or to `undefined` if preparing
 * failed or was aborted.
 *
 * @returns Whether a data source is being prepared, `prepare` and `cancel`
 */
export function usePrepareDataSource() {
  const [isPreparing, setPreparing] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  useEffect(() => () => abortControllerRef.current?.abort(), []);

  const alert = useAlertDialog();

  const cancel = () => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setPreparing(false);
  };

  const prepare = async <TDataSource extends DataSource>(
    dataSource: TDataSource,
    dataProvider: DataProvider<DataSource, Data> | undefined,
  ): Promise<TDataSource | undefined> => {
    if (dataProvider?.prepareDataSource === undefined) {
      return dataSource;
    }
    abortControllerRef.current?.abort(); // one preparation at a time
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    const { signal } = abortController;
    setPreparing(true);
    let preparedDataSource: TDataSource;
    try {
      preparedDataSource = (await dataProvider.prepareDataSource(
        dataSource,
        appStore.getState().workspace,
        projectStore.getState().source,
        { signal },
      )) as TDataSource;
    } catch (error) {
      if (!signal.aborted) {
        console.error("Failed to prepare the data source", error);
        void alert({
          title: "Cannot prepare the data source",
          body: error instanceof Error ? error.message : String(error),
        });
      }
      return undefined;
    } finally {
      // a cancelled preparation must not end a newer one
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null;
        setPreparing(false);
      }
    }
    return preparedDataSource;
  };

  return { isPreparing, prepare, cancel };
}
