import {
  type IDArray,
  MathUtils,
  NumberUtils,
  type OrderConfig,
  type TableData,
  TableUtils,
  getActiveConfigSource,
  isFromConfig,
} from "@tissuumaps/core";

/**
 * Resolves the order to draw items in
 */
export class OrderResolver {
  /**
   * Resolves the order to draw items in, so that high values of the column
   * are drawn on top of low ones
   *
   * Items without a value come first. The values are sorted into bins over
   * the value range of the column (see
   * {@link MathUtils.computeBinnedSortOrder}).
   *
   * @param ids - Ordered list of item IDs
   * @param config - Order configuration specifying the column, or
   * `undefined` for item order
   * @param loadTable - Async function that loads a {@link TableData} by ID, or
   * the object's own table for `undefined`; it rejects for a missing table
   * @param options - Optional abort signal
   * @returns The positions of the IDs in `ids`, in draw order, or `null` for
   * item order, i.e. if there is no `from` source or its table values cannot
   * be loaded
   */
  static async resolveOrder(
    ids: IDArray,
    config: OrderConfig | undefined,
    loadTable: (
      tableId: string | undefined,
      options?: { signal?: AbortSignal },
    ) => Promise<TableData>,
    options?: { signal?: AbortSignal },
  ): Promise<Uint32Array | null> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    if (
      config === undefined ||
      getActiveConfigSource(config) !== "from" ||
      !isFromConfig(config)
    ) {
      return null;
    }
    try {
      const data = await loadTable(config.from.table, { signal });
      const [tableValues, valueRange] = await Promise.all([
        data.loadValues(config.from.column, { signal }),
        TableUtils.loadValueRange(data, config.from.column, { signal }),
      ]);
      const values = new Float64Array(ids.length).fill(NaN);
      await TableUtils.forEachRow(
        ids,
        data,
        (rowIndex, i) => {
          if (rowIndex !== undefined) {
            values[i] =
              NumberUtils.tryParseFinite(tableValues[rowIndex], {
                requireSafeBigInt: true,
              }) ?? NaN;
          }
        },
        { signal },
      );
      return await MathUtils.computeBinnedSortOrder(
        values,
        valueRange ?? [0, 1],
        { signal },
      );
    } catch (error) {
      signal?.throwIfAborted();
      console.warn(
        "Failed to resolve the order from the table, drawing in item order",
        error,
      );
      return null;
    }
  }
}
