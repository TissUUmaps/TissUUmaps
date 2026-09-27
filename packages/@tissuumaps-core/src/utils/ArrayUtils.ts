import type { IDArray, IntOrUintArray } from "../types/arrays";
import { NumberUtils } from "./NumberUtils";

/**
 * Utility methods for converting arrays to the types the storage API holds
 */
export class ArrayUtils {
  /**
   * Converts item IDs to an `IDArray`
   *
   * Integer typed arrays, string arrays and 64-bit float arrays holding safe
   * integers are returned as they are. Any other array has to hold either
   * strings only, which are copied into a string array, or safe integers only,
   * which are copied into 64-bit floats. Numeric strings are not parsed:
   * whether `"12"` is an ID or a number is up to the data provider. An empty
   * array yields an empty unsigned 32-bit integer array.
   *
   * @param values - The IDs to convert
   * @returns The IDs as an integer typed array, a 64-bit float array or a
   * string array
   * @throws Error if a value is neither a string nor a safe integer, or if
   * strings and integers are mixed
   */
  static toIDArray(values: ArrayLike<unknown>): IDArray {
    if (ArrayUtils._isIntOrUintArray(values)) {
      return values;
    }
    if (values.length === 0) {
      return new Uint32Array(0);
    }
    if (typeof values[0] === "string") {
      const ids = Array.isArray(values) ? values : Array.from(values);
      for (const id of ids) {
        if (typeof id !== "string") {
          throw new Error(`ID is not a string: ${String(id)}`);
        }
      }
      return ids as string[];
    }
    const ids =
      values instanceof Float64Array ? values : new Float64Array(values.length);
    for (let i = 0; i < values.length; i++) {
      const id = values[i];
      if (typeof id !== "number" || !Number.isSafeInteger(id)) {
        throw new Error(`ID is not a safe integer: ${String(id)}`);
      }
      ids[i] = id;
    }
    return ids;
  }

  /**
   * Tells whether an array is an integer typed array of at most 32 bits
   */
  private static _isIntOrUintArray(
    values: ArrayLike<unknown>,
  ): values is IntOrUintArray {
    return (
      values instanceof Int8Array ||
      values instanceof Int16Array ||
      values instanceof Int32Array ||
      values instanceof Uint8Array ||
      values instanceof Uint8ClampedArray ||
      values instanceof Uint16Array ||
      values instanceof Uint32Array
    );
  }

  /**
   * Converts 64-bit integers to 64-bit floats
   *
   * @param values - The integers, as read from a file
   * @returns The integers as floats
   * @throws Error if an integer is outside the safe integer range
   */
  static parseSafeInts(values: BigInt64Array | BigUint64Array): Float64Array {
    return Float64Array.from(values, (v) => NumberUtils.parseSafeInt(v));
  }
}
