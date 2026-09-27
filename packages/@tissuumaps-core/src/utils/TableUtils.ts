import type { TableData } from "../storage/table";
import type { IDArray, TypedArray } from "../types/arrays";
import { AsyncUtils } from "./AsyncUtils";

/**
 * Lookups into tabular data (`TableData`) by item ID
 *
 * A table annotates the items of another data object: each of its rows holds
 * the item ID it belongs to, and consumers address rows by that ID rather than
 * by position (see `ItemsData`). Looking rows up by ID needs a map from ID to
 * row index, which is linear in the size of the table to build and costly to
 * hold; {@link forEachRow} therefore builds it at most once per table, and not
 * at all in the common case that the IDs to look up are the table's own, in
 * row order. Once built, the map is retained for as long as the table's data
 * is: it is keyed by the identity of the table's ID array, so it is freed
 * together with the data, and a table that is looked up by other IDs - shapes
 * annotated by a separate table, or the items of one layer of an object -
 * keeps one entry per row alive until then. That is the price of sharing the
 * map between every consumer of the table.
 *
 * Item IDs are expected to be unique within a table (see `ItemsData`). For a
 * duplicated ID, the row-order fast path addresses each row by its own
 * position, whereas the lookup maps the ID to its last row; a table with
 * duplicated IDs is reported once, when its map is built.
 *
 * On top of the row lookup, {@link fillFromTableValues} and
 * {@link fillFromTableGroups} fill a typed array with one packed value per
 * item from a table column, one by parsing the cell values and one by mapping
 * their distinct values as groups. IDs that the table does not contain fall
 * back to a default value, with a warning.
 */
export class TableUtils {
  /**
   * Row indices by item ID, by the table's ID array
   *
   * Keyed by the array that `getIds()` returns, which stays the very same array
   * for as long as the data is unchanged (see `ItemsData`), so an entry lives
   * exactly as long as the data it was built from. Holds the pending build,
   * so that concurrent lookups share it.
   */
  private static readonly _rowIndicesCache = new WeakMap<
    IDArray,
    Promise<ReadonlyMap<number | string, number>>
  >();

  /**
   * Fills `packedValues` by loading values from the given table column
   *
   * For each ID in `ids`, the corresponding row is looked up in the table by ID
   * (see {@link forEachRow}). The raw cell value is parsed by
   * `parseTableValue`; if that fails, or if the table does not contain the ID,
   * `defaultValue` is used instead.
   *
   * @param packedValues - Output typed array to fill, in the order of `ids`
   * @param tableData - The table to look up values in
   * @param ids - Ordered list of item IDs
   * @param column - Name of the table column to load values from
   * @param defaultValue - Value used when the ID is missing or parsing fails
   * @param parseTableValue - Converts a raw cell value to `TValue`, or to
   * `undefined` on failure
   * @param packValue - Converts `TValue` to the numeric representation stored
   * in `packedValues`
   * @param options - Optional abort signal
   */
  static async fillFromTableValues<TValue>(
    packedValues: TypedArray,
    tableData: TableData,
    ids: IDArray,
    column: string,
    defaultValue: TValue,
    parseTableValue: (value: unknown) => TValue | undefined,
    packValue: (value: TValue) => number,
    options?: { signal?: AbortSignal },
  ): Promise<void> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const tableValues = await tableData.loadValues(column, { signal });
    let numMissingIds = 0;
    let numUnparsedValues = 0;
    await TableUtils.forEachRow(
      ids,
      tableData,
      (rowIndex, i) => {
        if (rowIndex !== undefined) {
          const parsedValue = parseTableValue(tableValues[rowIndex]);
          if (parsedValue === undefined) {
            numUnparsedValues++;
          }
          packedValues[i] = packValue(parsedValue ?? defaultValue);
        } else {
          numMissingIds++;
          packedValues[i] = packValue(defaultValue);
        }
      },
      { signal },
    );
    if (numMissingIds > 0) {
      console.warn(
        `${numMissingIds} IDs missing in column ${column}, using default value`,
      );
    }
    if (numUnparsedValues > 0) {
      console.warn(
        `Failed to parse ${numUnparsedValues} values from column ${column}, using default value`,
      );
    }
  }

  /**
   * Fills `packedValues` by grouping IDs by the given table column and mapping the groups to values
   *
   * For each ID in `ids`, the corresponding row is looked up in the table by ID
   * (see {@link forEachRow}). The raw cell value is converted to a string to
   * form the group key, which `mapGroupToValue` maps to a value; if the table
   * does not contain the ID, `defaultValue` is used instead. Groups are
   * resolved once per distinct cell value, i.e. `mapGroupToValue` is not
   * called per item.
   *
   * @param packedValues - Output typed array to fill, in the order of `ids`
   * @param tableData - The table to look up group keys in
   * @param ids - Ordered list of item IDs
   * @param column - Name of the table column to load group keys from
   * @param defaultValue - Value used when the ID is missing
   * @param mapGroupToValue - Maps a group key (the cell value as a string) to
   * `TValue`
   * @param packValue - Converts `TValue` to the numeric representation stored
   * in `packedValues`
   * @param options - Optional abort signal
   */
  static async fillFromTableGroups<TValue>(
    packedValues: TypedArray,
    tableData: TableData,
    ids: IDArray,
    column: string,
    defaultValue: TValue,
    mapGroupToValue: (group: string) => TValue,
    packValue: (value: TValue) => number,
    options?: { signal?: AbortSignal },
  ): Promise<void> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const tableGroups = await tableData.loadValues(column, { signal });
    let numMissingIds = 0;
    const packedValueByTableGroup = new Map<unknown, number>();
    await TableUtils.forEachRow(
      ids,
      tableData,
      (rowIndex, i) => {
        if (rowIndex !== undefined) {
          const tableGroup = tableGroups[rowIndex];
          let packedValue = packedValueByTableGroup.get(tableGroup);
          if (packedValue === undefined) {
            packedValue = packValue(mapGroupToValue(String(tableGroup)));
            packedValueByTableGroup.set(tableGroup, packedValue);
          }
          packedValues[i] = packedValue;
        } else {
          numMissingIds++;
          packedValues[i] = packValue(defaultValue);
        }
      },
      { signal },
    );
    if (numMissingIds > 0) {
      console.warn(
        `${numMissingIds} IDs missing in column ${column}, using default value`,
      );
    }
  }

  /**
   * Calls `callback` for every ID with the index of the table row holding that ID
   *
   * If `ids` is the table's own ID array, the ID at position `i` is held by
   * row `i` and no lookup is needed. Otherwise the rows are looked up in the
   * table's row indices (see {@link getRowIndices}); IDs the table does not
   * contain are reported with an undefined row index.
   *
   * @param ids - Ordered list of item IDs
   * @param tableData - The table to look the IDs up in
   * @param callback - Called with the row index of each ID, or `undefined` if
   * the table does not contain it, and the position of the ID in `ids`
   * @param options - Optional abort signal
   */
  static async forEachRow(
    ids: IDArray,
    tableData: TableData,
    callback: (rowIndex: number | undefined, i: number) => void,
    options?: { signal?: AbortSignal },
  ): Promise<void> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    if (ids === tableData.getIds()) {
      await AsyncUtils.forEach<number | string>(ids, (_, i) => callback(i, i), {
        signal,
      });
      return;
    }
    const rowIndices = await TableUtils.getRowIndices(tableData, { signal });
    await AsyncUtils.forEach<number | string>(
      ids,
      (id, i) => callback(rowIndices.get(id), i),
      { signal },
    );
  }

  /**
   * Returns the row indices of a table by item ID
   *
   * Built at most once per table, cooperatively yielding, and shared by all
   * callers, which is why a caller's abort signal only detaches that caller:
   * the build itself keeps going for the others, and for the next lookup.
   *
   * @param tableData - The table to return the row indices of
   * @param options - Optional abort signal
   * @returns The row index of every item ID of the table; for an ID occurring
   * more than once, which is reported with a warning, the index of its last row
   */
  static getRowIndices(
    tableData: TableData,
    options?: { signal?: AbortSignal },
  ): Promise<ReadonlyMap<number | string, number>> {
    const tableIds = tableData.getIds();
    let rowIndicesPromise = TableUtils._rowIndicesCache.get(tableIds);
    if (rowIndicesPromise === undefined) {
      rowIndicesPromise = TableUtils._buildRowIndices(tableIds);
      TableUtils._rowIndicesCache.set(tableIds, rowIndicesPromise);
    }
    return AsyncUtils.raceSignal(rowIndicesPromise, options);
  }

  /**
   * Builds the row indices by item ID for the given table IDs
   */
  private static async _buildRowIndices(
    tableIds: IDArray,
  ): Promise<ReadonlyMap<number | string, number>> {
    const rowIndices = new Map<number | string, number>();
    await AsyncUtils.forEach<number | string>(tableIds, (id, i) => {
      rowIndices.set(id, i);
    });
    if (rowIndices.size !== tableIds.length) {
      console.warn(
        `${tableIds.length - rowIndices.size} duplicated item IDs in table, using the last row of each`,
      );
    }
    return rowIndices;
  }
}
