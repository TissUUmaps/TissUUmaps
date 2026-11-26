/**
 * Utility methods for coercing untrusted values to numbers
 *
 * Numbers, bigints and non-blank strings are accepted; everything else, and
 * everything that does not coerce to a finite number, is rejected. The
 * `tryParse*` methods report rejection by returning `undefined`, the `parse*`
 * methods by throwing.
 */
export class NumberUtils {
  private static readonly _minSafeBigInt = BigInt(Number.MIN_SAFE_INTEGER);
  private static readonly _maxSafeBigInt = BigInt(Number.MAX_SAFE_INTEGER);

  /**
   * Parses a value as a finite number
   *
   * @param value - The value to parse
   * @returns The parsed number
   * @throws Error if the value is not a finite number
   */
  static parseFinite(value: unknown): number {
    const v = NumberUtils.tryParseFinite(value);
    if (v === undefined) {
      throw new Error(
        `Value ${typeof value === "string" ? `'${value}'` : String(value)} is not a finite number`,
      );
    }
    return v;
  }

  /**
   * Parses a value as a safe integer
   *
   * @param value - The value to parse
   * @returns The parsed integer
   * @throws Error if the value is not a safe integer
   */
  static parseSafeInt(value: unknown): number {
    const v = NumberUtils.tryParseSafeInt(value);
    if (v === undefined) {
      throw new Error(
        `Value ${typeof value === "string" ? `'${value}'` : String(value)} is not a safe integer`,
      );
    }
    return v;
  }

  /**
   * Parses a value as a finite number, without throwing
   *
   * A bigint outside of the safe integer range loses precision when converted:
   * by default this is only warned about, `requireSafeBigInt` rejects it.
   *
   * @param value - The value to parse
   * @param options - Set `requireSafeBigInt` to reject bigints that are outside
   * of the safe integer range, rather than converting them with a warning
   * @returns The parsed number, or `undefined` if the value is not a finite
   * number
   */
  static tryParseFinite(
    value: unknown,
    options?: { requireSafeBigInt?: boolean },
  ): number | undefined {
    let v: number;
    if (typeof value === "number") {
      v = value;
    } else if (typeof value === "bigint") {
      if (
        value < NumberUtils._minSafeBigInt ||
        value > NumberUtils._maxSafeBigInt
      ) {
        if (options?.requireSafeBigInt) {
          return undefined;
        }
        console.warn(`Value ${value} is outside the safe integer range`);
      }
      v = Number(value);
    } else if (typeof value === "string" && value.trim() !== "") {
      v = Number(value);
    } else {
      return undefined;
    }
    return Number.isFinite(v) ? v : undefined;
  }

  /**
   * Parses a value as a safe integer, without throwing
   *
   * @param value - The value to parse
   * @returns The parsed integer, or `undefined` if the value is not a safe
   * integer
   */
  static tryParseSafeInt(value: unknown): number | undefined {
    const v = NumberUtils.tryParseFinite(value, { requireSafeBigInt: true });
    if (v === undefined) {
      return undefined;
    }
    return Number.isSafeInteger(v) ? v : undefined;
  }
}
