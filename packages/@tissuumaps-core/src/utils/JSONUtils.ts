/**
 * Utility methods for JSON serialization with support for non-finite numbers
 *
 * `NaN`, `Infinity` and `-Infinity` are encoded as the sentinel strings
 * `"__NaN__"`, `"__PositiveInfinity__"` and `"__NegativeInfinity__"`, which
 * {@link JSONUtils.parse} decodes back into numbers. Strings that happen to
 * equal a sentinel are therefore not round-trip safe.
 */
export class JSONUtils {
  /**
   * Serializes a value to JSON, encoding non-finite numbers as sentinel strings
   *
   * Behaves like `JSON.stringify` without indentation: `toJSON` methods are
   * honored, and object properties that cannot be represented (`undefined`,
   * functions, symbols) are omitted.
   *
   * It differs from `JSON.stringify` where the latter is lossy: such a value
   * throws instead of yielding `undefined` when it is the top-level value, and
   * throws instead of being written as `null` when it is an array element. It
   * also does not detect circular references, which overflow the stack rather
   * than throwing a `TypeError`.
   *
   * @param value - The value to serialize
   * @param options - Optional `stable` flag, sorting object keys alphabetically
   *   so that the output is independent of property insertion order
   *   (default `false`)
   * @throws If the value itself, or any array element it contains, cannot be
   *   represented in JSON
   */
  static stringify(value: unknown, options?: { stable?: boolean }): string {
    const { stable = false } = options || {};
    if (!JSONUtils._isStringifiable(value)) {
      throw new Error(`Value of type '${typeof value}' is not stringifiable`);
    }
    if (typeof value !== "object" || value === null) {
      return JSONUtils._stringifyPrimitive(value);
    }
    const obj = value as Record<string, unknown> & { toJSON?: () => unknown };
    if (typeof obj.toJSON === "function") {
      return JSONUtils.stringify(obj.toJSON(), options);
    }
    if (Array.isArray(value)) {
      return `[${value.map((v) => JSONUtils.stringify(v, options)).join(",")}]`;
    }
    const keys = Object.keys(obj).filter((key) =>
      JSONUtils._isStringifiable(obj[key]),
    );
    if (stable) {
      keys.sort();
    }
    return `{${keys
      .map(
        (k) =>
          `${JSONUtils._stringifyPrimitive(k)}:${JSONUtils.stringify(obj[k], options)}`,
      )
      .join(",")}}`;
  }

  /**
   * Parses JSON, decoding the sentinel strings written by
   * {@link JSONUtils.stringify} back into `NaN`, `Infinity` and `-Infinity`
   *
   * @param text - The JSON text to parse
   * @throws If the text is not valid JSON
   */
  static parse(text: string): unknown {
    return JSON.parse(text, function (_key, value) {
      if (typeof value === "string") {
        if (value === "__NaN__") {
          return Number.NaN;
        }
        if (value === "__PositiveInfinity__") {
          return Number.POSITIVE_INFINITY;
        }
        if (value === "__NegativeInfinity__") {
          return Number.NEGATIVE_INFINITY;
        }
      }
      return value as unknown;
    });
  }

  /** Whether a value can be represented in JSON */
  private static _isStringifiable(value: unknown): boolean {
    return (
      value !== undefined &&
      typeof value !== "function" &&
      typeof value !== "symbol"
    );
  }

  /**
   * Serializes a single primitive value (or object key) using
   * `JSON.stringify`, replacing non-finite numbers with sentinel strings
   *
   * The replacer reads the raw value from the holder object, because
   * `JSON.stringify` would otherwise pass the already-converted value (`null`
   * for non-finite numbers) to the replacer.
   */
  private static _stringifyPrimitive(value: unknown): string {
    return JSON.stringify(value, function (key) {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
      const rawValue: unknown = this[key];
      if (typeof rawValue === "number") {
        if (Number.isNaN(rawValue)) {
          return "__NaN__";
        }
        if (rawValue === Infinity) {
          return "__PositiveInfinity__";
        }
        if (rawValue === -Infinity) {
          return "__NegativeInfinity__";
        }
      }
      return rawValue;
    });
  }
}
