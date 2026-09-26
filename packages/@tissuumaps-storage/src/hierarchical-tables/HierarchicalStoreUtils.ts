import { NumberUtils, type TypedArrayOrArray } from "@tissuumaps/core";

import type { HierarchicalStoreValues } from "./HierarchicalStore";

/** Helpers over the nodes of a hierarchical store, for the reader and the profiles */
export class HierarchicalStoreUtils {
  /**
   * Converts 64-bit integers, which read as `BigInt64Array`, to numbers
   *
   * @param values - The values to convert
   * @returns The values, as numbers if they were 64-bit integers
   * @throws Error if a 64-bit integer is outside the safe integer range
   */
  static toNumbersIfInt64(
    values: HierarchicalStoreValues,
  ): TypedArrayOrArray<unknown> {
    if (values instanceof BigInt64Array || values instanceof BigUint64Array) {
      return Float64Array.from(values, (v) => NumberUtils.parseSafeInt(v));
    }
    return values;
  }

  /**
   * @param prefix - The path of the parent, empty for the root
   * @param name - The path below the parent
   * @returns The joined path
   */
  static joinPath(prefix: string, name: string): string {
    return prefix !== "" ? `${prefix}/${name}` : name;
  }
}
