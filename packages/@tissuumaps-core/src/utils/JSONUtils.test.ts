import { describe, expect, it } from "vitest";

import { JSONUtils } from "./JSONUtils";

describe("JSONUtils", () => {
  describe("stringify", () => {
    it("stringifies primitives", () => {
      expect(JSONUtils.stringify(42)).toBe("42");
      expect(JSONUtils.stringify(-1.5)).toBe("-1.5");
      expect(JSONUtils.stringify("abc")).toBe('"abc"');
      expect(JSONUtils.stringify(true)).toBe("true");
      expect(JSONUtils.stringify(null)).toBe("null");
    });

    it("stringifies objects and arrays without whitespace", () => {
      expect(JSONUtils.stringify({ a: 1, b: "x" })).toBe('{"a":1,"b":"x"}');
      expect(JSONUtils.stringify([1, "x", null])).toBe('[1,"x",null]');
      expect(JSONUtils.stringify({})).toBe("{}");
      expect(JSONUtils.stringify([])).toBe("[]");
    });

    it("stringifies nested structures", () => {
      expect(JSONUtils.stringify({ a: { b: [1, { c: 2 }] } })).toBe(
        '{"a":{"b":[1,{"c":2}]}}',
      );
    });

    it("encodes non-finite numbers as sentinel strings", () => {
      expect(JSONUtils.stringify(Number.NaN)).toBe('"__NaN__"');
      expect(JSONUtils.stringify(Number.POSITIVE_INFINITY)).toBe(
        '"__PositiveInfinity__"',
      );
      expect(JSONUtils.stringify(Number.NEGATIVE_INFINITY)).toBe(
        '"__NegativeInfinity__"',
      );
      expect(
        JSONUtils.stringify({ a: Number.NaN, b: [Number.POSITIVE_INFINITY] }),
      ).toBe('{"a":"__NaN__","b":["__PositiveInfinity__"]}');
    });

    it("preserves key order by default", () => {
      expect(JSONUtils.stringify({ b: 1, a: 2 })).toBe('{"b":1,"a":2}');
    });

    it("sorts keys at every level when stable", () => {
      expect(
        JSONUtils.stringify(
          { b: 1, a: { d: 2, c: 3 } },
          {
            stable: true,
          },
        ),
      ).toBe('{"a":{"c":3,"d":2},"b":1}');
    });

    it("produces equal output for equal objects with different key order when stable", () => {
      const options = { stable: true };
      expect(JSONUtils.stringify({ a: 1, b: 2 }, options)).toBe(
        JSONUtils.stringify({ b: 2, a: 1 }, options),
      );
    });

    it("omits properties that cannot be represented", () => {
      expect(
        JSONUtils.stringify({
          a: 1,
          b: undefined,
          c: () => 0,
          d: Symbol("d"),
        }),
      ).toBe('{"a":1}');
    });

    it("ignores symbol keys", () => {
      expect(JSONUtils.stringify({ a: 1, [Symbol("b")]: 2 })).toBe('{"a":1}');
    });

    it("honors toJSON", () => {
      expect(JSONUtils.stringify(new Date(0))).toBe(
        '"1970-01-01T00:00:00.000Z"',
      );
      expect(JSONUtils.stringify({ a: { toJSON: () => [1, 2] } })).toBe(
        '{"a":[1,2]}',
      );
    });

    it("throws for values that cannot be represented", () => {
      expect(() => JSONUtils.stringify(undefined)).toThrow(
        "Value of type 'undefined' is not stringifiable",
      );
      expect(() => JSONUtils.stringify(() => 0)).toThrow(
        "Value of type 'function' is not stringifiable",
      );
      expect(() => JSONUtils.stringify(Symbol("a"))).toThrow(
        "Value of type 'symbol' is not stringifiable",
      );
    });

    it("throws for array elements that cannot be represented", () => {
      expect(() => JSONUtils.stringify([1, undefined])).toThrow(
        "Value of type 'undefined' is not stringifiable",
      );
      expect(() => JSONUtils.stringify([() => 0])).toThrow(
        "Value of type 'function' is not stringifiable",
      );
    });

    it("throws when toJSON returns a value that cannot be represented", () => {
      expect(() => JSONUtils.stringify({ toJSON: () => undefined })).toThrow(
        "Value of type 'undefined' is not stringifiable",
      );
    });
  });

  describe("parse", () => {
    it("parses JSON", () => {
      expect(JSONUtils.parse('{"a":1,"b":[true,null,"x"]}')).toEqual({
        a: 1,
        b: [true, null, "x"],
      });
    });

    it("decodes sentinel strings into non-finite numbers", () => {
      expect(JSONUtils.parse('"__NaN__"')).toBeNaN();
      expect(JSONUtils.parse('"__PositiveInfinity__"')).toBe(
        Number.POSITIVE_INFINITY,
      );
      expect(JSONUtils.parse('"__NegativeInfinity__"')).toBe(
        Number.NEGATIVE_INFINITY,
      );
      expect(
        JSONUtils.parse('{"a":"__NaN__","b":["__NegativeInfinity__"]}'),
      ).toEqual({
        a: Number.NaN,
        b: [Number.NEGATIVE_INFINITY],
      });
    });

    it("decodes strings that happen to equal a sentinel", () => {
      expect(JSONUtils.parse(JSONUtils.stringify("__NaN__"))).toBeNaN();
    });

    it("throws for invalid JSON", () => {
      expect(() => JSONUtils.parse("{")).toThrow();
      expect(() => JSONUtils.parse("NaN")).toThrow();
    });
  });

  describe("round-trip", () => {
    it("preserves values", () => {
      const value = {
        n: 1,
        s: "x",
        b: false,
        nil: null,
        arr: [1, Number.NaN, Number.POSITIVE_INFINITY],
        nested: { deep: { nan: Number.NaN, inf: Number.NEGATIVE_INFINITY } },
      };
      expect(JSONUtils.parse(JSONUtils.stringify(value))).toEqual(value);
      expect(
        JSONUtils.parse(JSONUtils.stringify(value, { stable: true })),
      ).toEqual(value);
    });
  });
});
