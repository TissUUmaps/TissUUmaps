import { useEffect, useRef, useState } from "react";

import {
  MathUtils,
  type TableColumnDataType,
  TableUtils,
} from "@tissuumaps/core";

import { useConfirmDialog } from "@/components/dialogs/ConfirmDialog/hooks";
import { useLazyTableData } from "@/data/hooks/useLazyData";

import type { ColorBy, ColorByEncoding } from "./colorBy";

/**
 * Distinct values beyond which switching a numeric column to categories asks
 * for confirmation: a guard against grouping a continuous column by accident,
 * not a decision between categories and a gradient
 */
const maxCategoryCount = 1000;

/** The state of coloring by a column of the object's own table */
export type ColorByState = {
  /** The type of the colored column's values, once it is known */
  dataType: TableColumnDataType | null;

  /** Whether a column is being loaded for a pick or a switch */
  isPending: boolean;

  /** Why the last pick or switch failed, if it did */
  error: string | null;

  /**
   * Colors by a column, or by none; categories or a gradient follow from the
   * type of the column's values
   */
  pick: (column: string | null) => void;

  /** Switches the colored column between categories and a gradient */
  setEncoding: (encoding: ColorByEncoding) => void;
};

/**
 * Picks the column to color by and switches between categories and a
 * gradient, loading what that takes from the object's own table
 *
 * Numeric columns are shown as a gradient, all other columns (and those the
 * table declares as categories or booleans) as categories. Showing a column
 * as categories, whether picked or switched to, counts its distinct values
 * first, and asks for confirmation beyond {@link maxCategoryCount}. Nothing is written until the user picks or
 * switches; the write happens once the column is loaded, so the caller must
 * apply it to the object as it is then (see `PointsStyleWidget`).
 *
 * @param tableId - The ID of the object's own table, if any
 * @param colorBy - What the object is colored by
 * @param onColorByChange - Writes what the object is colored by
 * @returns The state of coloring by a column
 */
export function useColorBy(
  tableId: string | null,
  colorBy: ColorBy,
  onColorByChange: (colorBy: ColorBy) => void,
): ColorByState {
  const loadTableData = useLazyTableData(tableId);
  const confirm = useConfirmDialog();
  const [dataType, setDataType] = useState<TableColumnDataType | null>(null);
  const [isPending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const column = colorBy.kind === "column" ? colorBy.column : null;
  const encoding = colorBy.kind === "column" ? colorBy.encoding : null;

  // the column's type follows the configuration, e.g. when it is
  // edited in the advanced settings; the column is loaded anyway to render it
  useEffect(() => {
    /* eslint-disable react-hooks/set-state-in-effect -- reset synchronously,
       so that nothing stale is shown while the new column loads */
    setDataType(null);
    /* eslint-enable react-hooks/set-state-in-effect */
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
      const newDataType = await TableUtils.loadColumnDataType(
        tableData,
        column,
        { signal },
      );
      setDataType(newDataType);
    })().catch((error: unknown) => {
      if (!signal.aborted) {
        console.error("Failed to load the column to color by", error);
      }
    });
    return () => abortController.abort();
  }, [loadTableData, column]);

  useEffect(() => () => abortControllerRef.current?.abort(), []);

  // an action resolving to null changes nothing, e.g. when the user cancels
  const run = (action: (signal: AbortSignal) => Promise<ColorBy | null>) => {
    abortControllerRef.current?.abort();
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    const { signal } = abortController;
    setPending(true);
    setError(null);
    action(signal)
      .then((result) => {
        if (!signal.aborted && result !== null) {
          onColorByChange(result);
        }
      })
      .catch((error: unknown) => {
        if (!signal.aborted) {
          console.error("Failed to color by the column", error);
          setError(error instanceof Error ? error.message : String(error));
        }
      })
      .finally(() => {
        if (abortControllerRef.current === abortController) {
          abortControllerRef.current = null;
          setPending(false);
        }
      });
  };

  const loadTable = async (signal: AbortSignal) => {
    const tableData = await loadTableData({ signal });
    if (tableData === null) {
      throw new Error("The table is not available.");
    }
    return tableData;
  };

  /**
   * Asks whether to show a column with very many distinct values as
   * categories
   *
   * @param column - The column to show as categories
   * @param signal - Aborts the count
   * @returns Whether to go on
   */
  const confirmCategories = async (column: string, signal: AbortSignal) => {
    const tableData = await loadTable(signal);
    const values = await tableData.loadValues(column, { signal });
    const count = await MathUtils.countDistinct(values, maxCategoryCount, {
      signal,
    });
    return (
      count <= maxCategoryCount ||
      (await confirm({
        title: "Show as categories?",
        body: `This column has more than ${maxCategoryCount.toLocaleString()} distinct values. Showing it as categories may be slow. Show anyway?`,
        cancelLabel: "Cancel",
        actionLabel: "Show as categories",
      }))
    );
  };

  const pick = (newColumn: string | null) => {
    if (newColumn === null) {
      abortControllerRef.current?.abort();
      setError(null);
      onColorByChange({ kind: "none" });
      return;
    }
    run(async (signal) => {
      const tableData = await loadTable(signal);
      const newDataType = await TableUtils.loadColumnDataType(
        tableData,
        newColumn,
        { signal },
      );
      if (
        newDataType !== "number" &&
        !(await confirmCategories(newColumn, signal))
      ) {
        return null;
      }
      return {
        kind: "column",
        column: newColumn,
        encoding: newDataType === "number" ? "gradient" : "categories",
      };
    });
  };

  const setEncoding = (newEncoding: ColorByEncoding) => {
    if (column === null || newEncoding === encoding) {
      return;
    }
    run(async (signal) => {
      if (
        newEncoding === "categories" &&
        !(await confirmCategories(column, signal))
      ) {
        return null;
      }
      return { kind: "column", column, encoding: newEncoding };
    });
  };

  return { dataType, isPending, error, pick, setEncoding };
}
