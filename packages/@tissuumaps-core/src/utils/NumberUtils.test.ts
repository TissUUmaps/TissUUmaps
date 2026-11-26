import { afterEach, describe, expect, it, vi } from "vitest";

import { NumberUtils } from "./NumberUtils";

describe("NumberUtils", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("tryParseFinite", () => {
    it("returns finite numbers as-is", () => {
      expect(NumberUtils.tryParseFinite(0)).toBe(0);
      expect(NumberUtils.tryParseFinite(42)).toBe(42);
      expect(NumberUtils.tryParseFinite(-3.14)).toBe(-3.14);
    });

    it("returns undefined for non-finite numbers", () => {
      expect(NumberUtils.tryParseFinite(NaN)).toBeUndefined();
      expect(NumberUtils.tryParseFinite(Infinity)).toBeUndefined();
      expect(NumberUtils.tryParseFinite(-Infinity)).toBeUndefined();
    });

    it("parses numeric strings", () => {
      expect(NumberUtils.tryParseFinite("42")).toBe(42);
      expect(NumberUtils.tryParseFinite("-3.14")).toBe(-3.14);
      expect(NumberUtils.tryParseFinite("  7  ")).toBe(7);
      expect(NumberUtils.tryParseFinite("1e3")).toBe(1000);
    });

    it("returns undefined for empty or whitespace-only strings", () => {
      expect(NumberUtils.tryParseFinite("")).toBeUndefined();
      expect(NumberUtils.tryParseFinite("   ")).toBeUndefined();
    });

    it("returns undefined for non-numeric strings", () => {
      expect(NumberUtils.tryParseFinite("abc")).toBeUndefined();
      expect(NumberUtils.tryParseFinite("12px")).toBeUndefined();
    });

    it("returns undefined for non-number, non-string, non-bigint values", () => {
      expect(NumberUtils.tryParseFinite(undefined)).toBeUndefined();
      expect(NumberUtils.tryParseFinite(null)).toBeUndefined();
      expect(NumberUtils.tryParseFinite(true)).toBeUndefined();
      expect(NumberUtils.tryParseFinite({})).toBeUndefined();
      expect(NumberUtils.tryParseFinite([])).toBeUndefined();
    });

    it("parses bigints within the safe integer range", () => {
      expect(NumberUtils.tryParseFinite(42n)).toBe(42);
      expect(NumberUtils.tryParseFinite(-7n)).toBe(-7);
      expect(NumberUtils.tryParseFinite(BigInt(Number.MAX_SAFE_INTEGER))).toBe(
        Number.MAX_SAFE_INTEGER,
      );
    });

    it("warns but still parses out-of-range bigints by default", () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const big = BigInt(Number.MAX_SAFE_INTEGER) + 10n;
      expect(NumberUtils.tryParseFinite(big)).toBe(Number(big));
      expect(warn).toHaveBeenCalledOnce();
    });

    it("returns undefined for out-of-range bigints when requireSafeBigInt is set", () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const big = BigInt(Number.MAX_SAFE_INTEGER) + 10n;
      expect(
        NumberUtils.tryParseFinite(big, { requireSafeBigInt: true }),
      ).toBeUndefined();
      expect(warn).not.toHaveBeenCalled();
    });

    it("does not warn for in-range bigints", () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      NumberUtils.tryParseFinite(42n);
      expect(warn).not.toHaveBeenCalled();
    });
  });

  describe("parseFinite", () => {
    it("returns the parsed finite number", () => {
      expect(NumberUtils.parseFinite("3.5")).toBe(3.5);
      expect(NumberUtils.parseFinite(10)).toBe(10);
    });

    it("throws for non-finite values", () => {
      expect(() => NumberUtils.parseFinite("abc")).toThrow(
        "Value 'abc' is not a finite number",
      );
      expect(() => NumberUtils.parseFinite(NaN)).toThrow(
        "Value NaN is not a finite number",
      );
      expect(() => NumberUtils.parseFinite(undefined)).toThrow(
        "Value undefined is not a finite number",
      );
    });
  });

  describe("tryParseSafeInt", () => {
    it("returns safe integers", () => {
      expect(NumberUtils.tryParseSafeInt(0)).toBe(0);
      expect(NumberUtils.tryParseSafeInt(42)).toBe(42);
      expect(NumberUtils.tryParseSafeInt("-7")).toBe(-7);
      expect(NumberUtils.tryParseSafeInt(42n)).toBe(42);
    });

    it("returns undefined for non-integer finite numbers", () => {
      expect(NumberUtils.tryParseSafeInt(3.14)).toBeUndefined();
      expect(NumberUtils.tryParseSafeInt("2.5")).toBeUndefined();
    });

    it("returns undefined for unsafe integers", () => {
      expect(
        NumberUtils.tryParseSafeInt(Number.MAX_SAFE_INTEGER + 1),
      ).toBeUndefined();
    });

    it("returns undefined for out-of-range bigints without warning", () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const big = BigInt(Number.MAX_SAFE_INTEGER) + 10n;
      expect(NumberUtils.tryParseSafeInt(big)).toBeUndefined();
      expect(warn).not.toHaveBeenCalled();
    });

    it("returns undefined for non-parsable values", () => {
      expect(NumberUtils.tryParseSafeInt("abc")).toBeUndefined();
      expect(NumberUtils.tryParseSafeInt(null)).toBeUndefined();
      expect(NumberUtils.tryParseSafeInt(NaN)).toBeUndefined();
    });
  });

  describe("parseSafeInt", () => {
    it("returns the parsed safe integer", () => {
      expect(NumberUtils.parseSafeInt("42")).toBe(42);
      expect(NumberUtils.parseSafeInt(-7)).toBe(-7);
    });

    it("throws for non-safe-integer values", () => {
      expect(() => NumberUtils.parseSafeInt(3.14)).toThrow(
        "Value 3.14 is not a safe integer",
      );
      expect(() => NumberUtils.parseSafeInt("abc")).toThrow(
        "Value 'abc' is not a safe integer",
      );
    });
  });
});
