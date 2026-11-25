import { describe, expect, it } from "vitest";

import { ColorUtils } from "./ColorUtils";

describe("ColorUtils", () => {
  describe("parsePalette", () => {
    it("parses a simple color palette with default separator and maxValue", () => {
      const str = "0 0 0\n1 1 1";
      const result = ColorUtils.parsePalette(str);
      expect(result).toEqual([
        { r: 0, g: 0, b: 0 },
        { r: 255, g: 255, b: 255 },
      ]);
    });

    it("parses a color palette with custom separator and maxValue", () => {
      const str = "0,0,0\n0.5,0.5,0.5\n1,1,1";
      const result = ColorUtils.parsePalette(str, {
        sep: ",",
        maxValue: 1,
      });
      expect(result).toEqual([
        { r: 0, g: 0, b: 0 },
        { r: 127.5, g: 127.5, b: 127.5 },
        { r: 255, g: 255, b: 255 },
      ]);
    });

    it("parses an 8-bit palette with maxValue 255", () => {
      const str = "0 0 0\n128 64 32\n255 255 255";
      const result = ColorUtils.parsePalette(str, {
        sep: " ",
        maxValue: 255,
      });
      expect(result).toEqual([
        { r: 0, g: 0, b: 0 },
        { r: 128, g: 64, b: 32 },
        { r: 255, g: 255, b: 255 },
      ]);
    });

    it("throws on invalid line", () => {
      const str = "0 0\n1 1 1";
      expect(() => ColorUtils.parsePalette(str)).toThrow(
        /Invalid color palette line '.*' at index 0/,
      );
    });

    it("ignores empty lines", () => {
      const str = "\n0 0 0\n\n1 1 1\n";
      const result = ColorUtils.parsePalette(str);
      expect(result).toHaveLength(2);
    });

    it("parses a single color line", () => {
      const result = ColorUtils.parsePalette("0.5 0.5 0.5");
      expect(result).toEqual([{ r: 127.5, g: 127.5, b: 127.5 }]);
    });
  });

  describe("packVisibility", () => {
    it("packs booleans as 1 or 0", () => {
      expect(ColorUtils.packVisibility(true)).toBe(1);
      expect(ColorUtils.packVisibility(false)).toBe(0);
    });
  });

  describe("packOpacity", () => {
    it("scales opacity (0–1) into the 0–255 range", () => {
      expect(ColorUtils.packOpacity(0)).toBe(0);
      expect(ColorUtils.packOpacity(1)).toBe(255);
      expect(ColorUtils.packOpacity(0.5)).toBe(128); // round(127.5)
    });

    it("clamps the packed value to [0, 255]", () => {
      expect(ColorUtils.packOpacity(2)).toBe(255);
      expect(ColorUtils.packOpacity(-1)).toBe(0);
    });
  });

  describe("sampleColorScheme", () => {
    const grayscale = (t: number) => {
      const value = Math.round(t * 255);
      return `rgb(${value}, ${value}, ${value})`;
    };

    it("samples both ends of the scheme", () => {
      const result = ColorUtils.sampleColorScheme(grayscale, 2);
      expect(result).toEqual([
        { r: 0, g: 0, b: 0 },
        { r: 255, g: 255, b: 255 },
      ]);
    });

    it("samples evenly spaced positions", () => {
      const result = ColorUtils.sampleColorScheme(grayscale, 3);
      expect(result[1]).toEqual({ r: 128, g: 128, b: 128 });
    });

    it("parses hex color strings", () => {
      const result = ColorUtils.sampleColorScheme(() => "#010203", 2);
      expect(result).toEqual([
        { r: 1, g: 2, b: 3 },
        { r: 1, g: 2, b: 3 },
      ]);
    });

    it("throws on a size smaller than two", () => {
      expect(() => ColorUtils.sampleColorScheme(grayscale, 1)).toThrow(
        /Invalid color scheme size/,
      );
    });

    it("throws on a fractional size", () => {
      expect(() => ColorUtils.sampleColorScheme(grayscale, 2.5)).toThrow(
        /Invalid color scheme size/,
      );
    });

    it("throws on a color string it cannot parse", () => {
      expect(() =>
        ColorUtils.sampleColorScheme(() => "oklch(0.7 0.1 200)", 2),
      ).toThrow(/Invalid color scheme color/);
    });
  });

  describe("withAlpha", () => {
    it("adds the opacity as alpha for a visible item", () => {
      expect(ColorUtils.withAlpha(0x030201, 1, 0xff)).toBe(0xff030201);
    });

    it("yields zero alpha for an invisible item", () => {
      expect(ColorUtils.withAlpha(0x030201, 0, 0xff)).toBe(0x00030201);
    });

    it("discards any existing alpha", () => {
      expect(ColorUtils.withAlpha(0xaa030201, 1, 0x80)).toBe(0x80030201);
      expect(ColorUtils.withAlpha(0xaa030201, 0, 0x80)).toBe(0x00030201);
    });

    it("returns an unsigned 32-bit integer", () => {
      expect(ColorUtils.withAlpha(0xffffff, 1, 0xff)).toBe(0xffffffff);
    });
  });

  describe("packColor", () => {
    it("packs a color to a 24-bit integer", () => {
      expect(ColorUtils.packColor({ r: 1, g: 2, b: 3 })).toBe(
        (3 << 16) | (2 << 8) | 1,
      );
    });

    it("packs white", () => {
      expect(ColorUtils.packColor({ r: 255, g: 255, b: 255 })).toBe(0xffffff);
    });

    it("packs black", () => {
      expect(ColorUtils.packColor({ r: 0, g: 0, b: 0 })).toBe(0x000000);
    });
  });

  describe("fromHex", () => {
    it("converts hex to color", () => {
      expect(ColorUtils.fromHex("#010203")).toEqual({ r: 1, g: 2, b: 3 });
    });

    it("handles mixed case", () => {
      expect(ColorUtils.fromHex("#aaBBcc")).toEqual({ r: 170, g: 187, b: 204 });
    });

    it("throws on missing hash prefix", () => {
      expect(() => ColorUtils.fromHex("010203")).toThrow(/Invalid hex color/);
    });

    it("throws on non-hex characters", () => {
      expect(() => ColorUtils.fromHex("#GGHHII")).toThrow(/Invalid hex color/);
    });

    it("throws on wrong length", () => {
      expect(() => ColorUtils.fromHex("#FFF")).toThrow(/Invalid hex color/);
    });
  });

  describe("toHex", () => {
    it("converts color to hex", () => {
      expect(ColorUtils.toHex({ r: 1, g: 2, b: 3 })).toBe("#010203");
    });

    it("rounds fractional components", () => {
      expect(ColorUtils.toHex({ r: 1.4, g: 2.6, b: 3.5 })).toBe("#010304");
    });

    it("converts white to #ffffff", () => {
      expect(ColorUtils.toHex({ r: 255, g: 255, b: 255 })).toBe("#ffffff");
    });

    it("converts black to #000000", () => {
      expect(ColorUtils.toHex({ r: 0, g: 0, b: 0 })).toBe("#000000");
    });
  });

  describe("equals", () => {
    it("returns true for identical colors", () => {
      expect(
        ColorUtils.equals({ r: 1, g: 2, b: 3 }, { r: 1, g: 2, b: 3 }),
      ).toBe(true);
    });

    it("returns false when a component differs", () => {
      expect(
        ColorUtils.equals({ r: 1, g: 2, b: 3 }, { r: 1, g: 2, b: 4 }),
      ).toBe(false);
      expect(
        ColorUtils.equals({ r: 1, g: 2, b: 3 }, { r: 1, g: 9, b: 3 }),
      ).toBe(false);
      expect(
        ColorUtils.equals({ r: 1, g: 2, b: 3 }, { r: 9, g: 2, b: 3 }),
      ).toBe(false);
    });

    it("compares fractional components exactly", () => {
      expect(
        ColorUtils.equals({ r: 127.5, g: 0, b: 0 }, { r: 127.5, g: 0, b: 0 }),
      ).toBe(true);
      expect(
        ColorUtils.equals({ r: 127.5, g: 0, b: 0 }, { r: 128, g: 0, b: 0 }),
      ).toBe(false);
    });

    it("returns true for the same object", () => {
      const color = { r: 10, g: 20, b: 30 };
      expect(ColorUtils.equals(color, color)).toBe(true);
    });
  });

  describe("fromHSB", () => {
    it("returns gray for zero saturation", () => {
      expect(ColorUtils.fromHSB(0.3, 0, 0.5)).toEqual({
        r: 128,
        g: 128,
        b: 128,
      });
    });

    it.each([
      [0, { r: 255, g: 0, b: 0 }],
      [1 / 6, { r: 255, g: 255, b: 0 }],
      [2 / 6, { r: 0, g: 255, b: 0 }],
      [3 / 6, { r: 0, g: 255, b: 255 }],
      [4 / 6, { r: 0, g: 0, b: 255 }],
      [5 / 6, { r: 255, g: 0, b: 255 }],
    ])("maps hue %f at full saturation and brightness", (hue, color) => {
      expect(ColorUtils.fromHSB(hue, 1, 1)).toEqual(color);
    });

    it("wraps hues outside [0, 1)", () => {
      expect(ColorUtils.fromHSB(1.5, 1, 1)).toEqual(
        ColorUtils.fromHSB(0.5, 1, 1),
      );
      expect(ColorUtils.fromHSB(-0.25, 1, 1)).toEqual(
        ColorUtils.fromHSB(0.75, 1, 1),
      );
    });

    it("scales by brightness", () => {
      expect(ColorUtils.fromHSB(0, 1, 0.5)).toEqual({ r: 128, g: 0, b: 0 });
    });
  });

  describe("fromHex / toHex roundtrip", () => {
    it.each(["#000000", "#ffffff", "#1a2b3c", "#ff8800"])(
      "roundtrips %s",
      (hex) => {
        expect(ColorUtils.toHex(ColorUtils.fromHex(hex))).toBe(hex);
      },
    );
  });
});
