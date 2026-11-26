import { useEffect, useRef, useState } from "react";

import { MathUtils, type TableColumnType, TableUtils } from "@tissuumaps/core";

import { useAlertDialog } from "@/components/dialogs/AlertDialog/hooks";
import { useConfirmDialog } from "@/components/dialogs/ConfirmDialog/hooks";
import { useLazyTableData } from "@/data/hooks/useLazyData";

import type { StyleBy } from "./styleBy";

/**
 * Distinct values beyond which using a numerical column as categorical asks
 * for confirmation: a guard against grouping a continuous column by accident,
 * not a decision between categorical and numerical
 */
const maxCategoryCount = 1000;

/** The state of styling by a column of the object's own table */
export type StyleByState = {
  /** Whether the column is categorical, once it is known */
  isCategorical: boolean | null;

  /** Whether a column is being loaded for a pick or a switch */
  isPending: boolean;

  /**
   * Styles by a column, or by none; categorical or numerical follows from
   * the type of the column's values. Resolves to whether it applied, `false`
   * when cancelled, failed or aborted.
   */
  pick: (column: string | null) => Promise<boolean>;

  /** Switches the column between categorical and numerical */
  setColumnType: (columnType: TableColumnType) => void;
};

/**
 * Picks the column to style by and switches between categorical and
 * numerical, loading what that takes from the object's own table
 *
 * Categorical columns start as categorical, all others as numerical (see
 * `TableUtils.loadColumnType`). Using a column as categorical,
 * whether picked or switched to, counts its distinct values first, and asks
 * for confirmation beyond {@link maxCategoryCount}. A failed pick or switch
 * is shown in an alert. Nothing is written until the user picks or switches;
 * the write happens once the column is loaded, so the caller must apply it to
 * the object as it is then (see `PointsSettingsWidget`).
 *
 * @param tableId - The ID of the object's own table, if any
 * @param styleBy - What the object is styled by
 * @param onStyleByChange - Writes what the object is styled by
 * @returns The state of styling by a column
 */
export function useStyleBy(
  tableId: string | null,
  styleBy: StyleBy,
  onStyleByChange: (styleBy: StyleBy) => void,
): StyleByState {
  const loadTableData = useLazyTableData(tableId);
  const confirm = useConfirmDialog();
  const alert = useAlertDialog();
  const [isCategorical, setCategorical] = useState<boolean | null>(null);
  const [isPending, setPending] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const column = styleBy.kind === "column" ? styleBy.column : null;
  const columnType = styleBy.kind === "column" ? styleBy.columnType : null;

  // a new table or column is unknown until loaded, so that nothing stale is
  // shown meanwhile
  // https://react.dev/reference/react/useState#storing-information-from-previous-renders
  const [prevTableId, setPrevTableId] = useState(tableId);
  const [prevColumn, setPrevColumn] = useState(column);
  if (tableId !== prevTableId || column !== prevColumn) {
    setPrevTableId(tableId);
    setPrevColumn(column);
    setCategorical(null);
  }

  // follows the configuration, e.g. when the column is edited in the
  // advanced settings; the column is loaded anyway to render it
  useEffect(() => {
    if (column === null) {
      return;
    }
    const abortController = new AbortController();
    const { signal } = abortController;
    (async () => {
      const tableData = await loadTableData({ signal });
      if (tableData === null) {
        return;
      }
      setCategorical(
        (await TableUtils.loadColumnType(tableData, column, { signal })) ===
          "categorical",
      );
    })().catch((error: unknown) => {
      if (!signal.aborted) {
        console.error("Failed to load the column to style by", error);
      }
    });
    return () => abortController.abort();
  }, [loadTableData, column]);

  // a pick or switch still loading is for the previous table
  useEffect(() => () => abortControllerRef.current?.abort(), [tableId]);

  // an action resolving to null changes nothing, e.g. when the user cancels;
  // resolves to whether it applied
  const run = (
    action: (options: { signal: AbortSignal }) => Promise<StyleBy | null>,
  ): Promise<boolean> => {
    abortControllerRef.current?.abort();
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    const { signal } = abortController;
    setPending(true);
    return action({ signal })
      .then((result) => {
        if (signal.aborted || result === null) {
          return false;
        }
        onStyleByChange(result);
        return true;
      })
      .catch(async (error: unknown) => {
        if (!signal.aborted) {
          console.error("Failed to style by the column", error);
          await alert({
            title: "Cannot style by the column",
            body: error instanceof Error ? error.message : String(error),
          });
        }
        return false;
      })
      .finally(() => {
        if (abortControllerRef.current === abortController) {
          abortControllerRef.current = null;
          setPending(false);
        }
      });
  };

  const loadTable = async (options: { signal: AbortSignal }) => {
    const tableData = await loadTableData(options);
    if (tableData === null) {
      throw new Error(`Table with ID '${tableId}' not found`);
    }
    return tableData;
  };

  /**
   * Asks whether to use a column with very many distinct values as
   * categorical
   *
   * @param column - The column to use as categorical
   * @param options - Aborts the count
   * @returns Whether to go on
   */
  const confirmCategorical = async (
    column: string,
    options: { signal: AbortSignal },
  ) => {
    const tableData = await loadTable(options);
    const values = await tableData.loadValues(column, options);
    const count = await MathUtils.countUniqueValues(
      values,
      maxCategoryCount,
      options,
    );
    return (
      count <= maxCategoryCount ||
      (await confirm({
        title: "Use as categorical?",
        body: `This column has more than ${maxCategoryCount.toLocaleString()} distinct values. Using it as categorical may be slow. Use it anyway?`,
        cancelLabel: "Cancel",
        actionLabel: "Use as categorical",
      }))
    );
  };

  const pick = (newColumn: string | null) => {
    if (newColumn === null) {
      abortControllerRef.current?.abort();
      onStyleByChange({ kind: "none" });
      return Promise.resolve(true);
    }
    return run(async (options) => {
      const tableData = await loadTable(options);
      const columnType = await TableUtils.loadColumnType(
        tableData,
        newColumn,
        options,
      );
      if (
        columnType === "categorical" &&
        !(await confirmCategorical(newColumn, options))
      ) {
        return null;
      }
      return { kind: "column", column: newColumn, columnType };
    });
  };

  const setColumnType = (newColumnType: TableColumnType) => {
    if (column === null || newColumnType === columnType) {
      return;
    }
    void run(async (options) => {
      if (
        newColumnType === "categorical" &&
        !(await confirmCategorical(column, options))
      ) {
        return null;
      }
      return { kind: "column", column, columnType: newColumnType };
    });
  };

  return { isCategorical, isPending, pick, setColumnType };
}
