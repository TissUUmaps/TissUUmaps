import { describe, expect, it, vi } from "vitest";

import {
  type Color,
  type ColorConfig,
  type ColorPalette,
  ColorUtils,
  type GroupValueMap,
  HashUtils,
  type IDArray,
  type TableData,
  colorPalettes,
  defaultRandomSeed,
} from "@tissuumaps/core";

import { ColorResolver } from "./ColorResolver";

function createMockTableData(
  ids: IDArray,
  values: unknown[],
  valueRange?: [number, number],
): TableData {
  return {
    getIds: () => ids,
    getSize: () => ids.length,
    getNames: () => undefined,
    close: vi.fn(),
    loadValues: vi.fn().mockResolvedValue(values),
    loadValueRange: vi.fn().mockResolvedValue(valueRange),
    loadUniqueValueCounts: vi.fn(),
    suggestColumnQueries: vi.fn(),
    resolveColumnQuery: vi.fn(),
  };
}

const red: Color = { r: 255, g: 0, b: 0 };
const green: Color = { r: 0, g: 255, b: 0 };
const blue: Color = { r: 0, g: 0, b: 255 };
const black: Color = { r: 0, g: 0, b: 0 };

const testPalette: ColorPalette = {
  id: "test-palette",
  name: "Test",
  colors: [red, green, blue],
};

describe("ColorResolver", () => {
  describe("packColor", () => {
    it("packs the color via ColorUtils.packColor", () => {
      expect(ColorResolver.packColor(red)).toBe(ColorUtils.packColor(red));
      expect(ColorResolver.packColor(black)).toBe(0);
    });
  });

  describe("parseColor", () => {
    it("maps the minimum of the range to the first color", () => {
      expect(ColorResolver.parseColor(0, [0, 3], testPalette)).toEqual(red);
    });

    it("maps the middle of the range to the middle color", () => {
      expect(ColorResolver.parseColor(1.5, [0, 3], testPalette)).toEqual(green);
    });

    it("maps the maximum of the range to the last color", () => {
      expect(ColorResolver.parseColor(3, [0, 3], testPalette)).toEqual(blue);
    });

    it("interpolates between the two colors a value falls between", () => {
      expect(ColorResolver.parseColor(0.75, [0, 3], testPalette)).toEqual({
        r: 127.5,
        g: 127.5,
        b: 0,
      });
    });

    it("maps every value onto the first color for an empty range", () => {
      expect(ColorResolver.parseColor(5, [5, 5], testPalette)).toEqual(red);
    });

    it("normalizes within the given range", () => {
      // 5/10 = 0.5, the middle of the palette → green
      expect(ColorResolver.parseColor(5, [0, 10], testPalette)).toEqual(green);
    });

    it("normalizes within a wide range", () => {
      // 50/100 = 0.5, the middle of the palette → green
      expect(ColorResolver.parseColor(50, [0, 100], testPalette)).toEqual(
        green,
      );
    });

    it("defaults to [0, 1] when the range is undefined", () => {
      // 0.5/1 = 0.5, the middle of the palette → green
      expect(ColorResolver.parseColor(0.5, undefined, testPalette)).toEqual(
        green,
      );
    });

    it("clamps out-of-range values to the palette ends", () => {
      expect(ColorResolver.parseColor(-10, [0, 1], testPalette)).toEqual(red);
      expect(ColorResolver.parseColor(100, [0, 1], testPalette)).toEqual(blue);
    });

    it("returns undefined for non-finite values", () => {
      expect(
        ColorResolver.parseColor(NaN, [0, 1], testPalette),
      ).toBeUndefined();
      expect(
        ColorResolver.parseColor(Infinity, [0, 1], testPalette),
      ).toBeUndefined();
    });

    it("returns undefined for non-number values", () => {
      expect(
        ColorResolver.parseColor("abc", [0, 1], testPalette),
      ).toBeUndefined();
    });
  });

  describe("createColorBuffer", () => {
    it("creates a zeroed Uint32Array of the requested size", () => {
      const packedColors = ColorResolver.createColorBuffer(3);
      expect(packedColors).toBeInstanceOf(Uint32Array);
      expect(packedColors.length).toBe(3);
      expect(Array.from(packedColors)).toEqual([0, 0, 0]);
    });

    it("aligns the buffer length to the given boundary", () => {
      expect(ColorResolver.createColorBuffer(3, { align: 4 }).length).toBe(4);
    });
  });

  describe("createUniformColors", () => {
    it("fills the buffer with the packed color", () => {
      const packedColors = ColorResolver.createUniformColors(3, red);
      const packedColor = ColorResolver.packColor(red);
      expect(packedColors.length).toBe(3);
      expect(Array.from(packedColors)).toEqual([
        packedColor,
        packedColor,
        packedColor,
      ]);
    });

    it("respects alignment while filling only the requested count", () => {
      const packedColors = ColorResolver.createUniformColors(3, red, {
        align: 4,
      });
      const packedColor = ColorResolver.packColor(red);
      expect(packedColors.length).toBe(4);
      expect(packedColors[0]).toBe(packedColor);
      expect(packedColors[2]).toBe(packedColor);
      expect(packedColors[3]).toBe(0); // padding element is left zeroed
    });

    it("returns an empty buffer for n = 0", () => {
      expect(ColorResolver.createUniformColors(0, red).length).toBe(0);
    });
  });

  describe("resolveUniformColors", () => {
    it("fills the buffer with the constant color", () => {
      const config = { constant: { value: green } } satisfies ColorConfig;
      const packedColors = ColorResolver.resolveUniformColors(
        new Uint32Array([1, 2, 3]),
        config,
      );
      const packedColor = ColorResolver.packColor(green);
      expect(Array.from(packedColors)).toEqual([
        packedColor,
        packedColor,
        packedColor,
      ]);
    });
  });

  describe("resolveColorsFromTableValues", () => {
    it("maps table values through the color palette", async () => {
      const palette = colorPalettes[0]!;
      const ids = new Uint32Array([1, 2]);
      const data = createMockTableData(ids, [0, 1], [0, 1]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = {
        from: { column: "col1", palette: palette.id },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveColorsFromTableValues(
        ids,
        config,
        black,
        loadTable,
      );

      // value 0 → normalized 0 → first color; value 1 → clamped to last color
      expect(packedColors[0]).toBe(ColorResolver.packColor(palette.colors[0]!));
      expect(packedColors[1]).toBe(
        ColorResolver.packColor(palette.colors[palette.colors.length - 1]!),
      );
    });

    it("normalizes within the loaded value range when no range is configured", async () => {
      const palette = colorPalettes[0]!;
      const ids = new Uint32Array([1, 2]);
      const loadValueRange = vi.fn().mockResolvedValue([-10, 30]);
      const data = { ...createMockTableData(ids, [-10, 30]), loadValueRange };
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = {
        from: { column: "col1", palette: palette.id },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveColorsFromTableValues(
        ids,
        config,
        black,
        loadTable,
      );

      expect(loadValueRange).toHaveBeenCalledWith("col1", {
        signal: undefined,
      });
      // the loaded range maps -10 to the first and 30 to the last color
      expect(packedColors[0]).toBe(ColorResolver.packColor(palette.colors[0]!));
      expect(packedColors[1]).toBe(
        ColorResolver.packColor(palette.colors[palette.colors.length - 1]!),
      );
    });

    it("normalizes within the configured range without loading the value range", async () => {
      const palette = colorPalettes[0]!;
      const ids = new Uint32Array([1, 2]);
      const loadValueRange = vi.fn().mockResolvedValue([0, 100]);
      const data = { ...createMockTableData(ids, [0, 100]), loadValueRange };
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = {
        from: { column: "col1", palette: palette.id, range: [100, 200] },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveColorsFromTableValues(
        ids,
        config,
        black,
        loadTable,
      );

      expect(loadValueRange).not.toHaveBeenCalled();
      // both values fall below the configured range and clamp to the first color
      const firstColor = ColorResolver.packColor(palette.colors[0]!);
      expect(Array.from(packedColors)).toEqual([firstColor, firstColor]);
    });

    it("returns uniform default color when the palette is not found", async () => {
      const loadTable = vi.fn();
      const config = {
        from: { column: "col1", palette: "nonexistent" },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveColorsFromTableValues(
        new Uint32Array([1, 2]),
        config,
        red,
        loadTable,
      );

      const packedColor = ColorResolver.packColor(red);
      expect(Array.from(packedColors)).toEqual([packedColor, packedColor]);
      expect(loadTable).not.toHaveBeenCalled();
    });

    it("forwards the signal to loadTable", async () => {
      const controller = new AbortController();
      const palette = colorPalettes[0]!;
      const data = createMockTableData(new Uint32Array([1]), [0], [0, 1]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = {
        from: { column: "col1", palette: palette.id },
      } satisfies ColorConfig;

      await ColorResolver.resolveColorsFromTableValues(
        new Uint32Array([1]),
        config,
        black,
        loadTable,
        { signal: controller.signal },
      );

      expect(loadTable).toHaveBeenCalledWith({ signal: controller.signal });
    });
  });

  describe("resolveColorsFromTableGroups", () => {
    it("uses the color map when a map is specified and found", async () => {
      const ids = new Uint32Array([1, 2]);
      const data = createMockTableData(ids, ["cat-a", "cat-b"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const colorMap: GroupValueMap<Color> = {
        id: "cm1",
        name: "Color Map 1",
        values: {
          "cat-a": red,
          "cat-b": green,
        },
      };
      const config = {
        groupBy: { column: "col1", map: "cm1" },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveColorsFromTableGroups(
        ids,
        config,
        [colorMap],
        black,
        loadTable,
      );

      expect(packedColors[0]).toBe(ColorResolver.packColor(red));
      expect(packedColors[1]).toBe(ColorResolver.packColor(green));
    });

    it("uses the color map's default for unmapped groups", async () => {
      const ids = new Uint32Array([1]);
      const data = createMockTableData(ids, ["missing"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const colorMap: GroupValueMap<Color> = {
        id: "cm1",
        name: "Color Map 1",
        values: {},
        default: blue,
      };
      const config = {
        groupBy: { column: "col1", map: "cm1" },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveColorsFromTableGroups(
        ids,
        config,
        [colorMap],
        black,
        loadTable,
      );

      expect(packedColors[0]).toBe(ColorResolver.packColor(blue));
    });

    it("returns uniform default color when a map is specified but not found", async () => {
      const loadTable = vi.fn();
      const config = {
        groupBy: { column: "col1", map: "nonexistent" },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveColorsFromTableGroups(
        new Uint32Array([1]),
        config,
        [],
        red,
        loadTable,
      );

      expect(packedColors[0]).toBe(ColorResolver.packColor(red));
      expect(loadTable).not.toHaveBeenCalled();
    });

    it("hashes group names through the palette when only a palette is given", async () => {
      // resolveColorsFromTableGroups looks palettes up in the built-in
      // colorPalettes, rather than in a parameter
      const builtInPalette = colorPalettes[0]!;
      const ids = new Uint32Array([1, 2]);
      const data = createMockTableData(ids, ["groupA", "groupB"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = {
        groupBy: { column: "col1", map: undefined, palette: builtInPalette.id },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveColorsFromTableGroups(
        ids,
        config,
        [],
        black,
        loadTable,
      );

      // Colors are deterministically hash-picked from the palette
      expect(packedColors[0]).toBe(
        ColorResolver.packColor(
          builtInPalette.colors[
            HashUtils.hash("groupA") % builtInPalette.colors.length
          ]!,
        ),
      );
      expect(packedColors[1]).toBe(
        ColorResolver.packColor(
          builtInPalette.colors[
            HashUtils.hash("groupB") % builtInPalette.colors.length
          ]!,
        ),
      );
    });

    it("returns uniform default color when a palette is specified but not found", async () => {
      const loadTable = vi.fn();
      const config = {
        groupBy: { column: "col1", map: undefined, palette: "nonexistent" },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveColorsFromTableGroups(
        new Uint32Array([1]),
        config,
        [],
        red,
        loadTable,
      );

      expect(packedColors[0]).toBe(ColorResolver.packColor(red));
      expect(loadTable).not.toHaveBeenCalled();
    });

    it("returns uniform default color when neither map nor palette is given", async () => {
      const loadTable = vi.fn();
      const config = {
        groupBy: { column: "col1", map: undefined },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveColorsFromTableGroups(
        new Uint32Array([1]),
        config,
        [],
        green,
        loadTable,
      );

      expect(packedColors[0]).toBe(ColorResolver.packColor(green));
      expect(loadTable).not.toHaveBeenCalled();
    });
  });

  describe("resolveRandomColors", () => {
    // resolveRandomColors samples from the built-in colorPalettes, not a parameter
    const builtInPalette = colorPalettes[0]!;

    it("assigns colors drawn from the palette", async () => {
      const config = {
        random: { seed: 0, palette: builtInPalette.id },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveRandomColors(
        new Uint32Array([1, 2, 3]),
        config,
        black,
        { signal: undefined, align: 1 },
      );

      expect(packedColors.length).toBe(3);
      const valid = builtInPalette.colors.map((color) =>
        ColorResolver.packColor(color),
      );
      for (const value of packedColors) {
        expect(valid).toContain(value);
      }
    });

    it("draws the color of each ID via pickRandomColor", async () => {
      const config = {
        random: { seed: 7, palette: builtInPalette.id },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveRandomColors(
        new Uint32Array([5, 42, 1000]),
        config,
        black,
      );

      expect(Array.from(packedColors)).toEqual(
        [5, 42, 1000].map((id) =>
          ColorResolver.packColor(
            ColorResolver.pickRandomColor(id, 7, builtInPalette),
          ),
        ),
      );
    });

    it("draws the colors of string IDs from their hashes", async () => {
      const config = {
        random: { seed: 7, palette: builtInPalette.id },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveRandomColors(
        ["a", "b", "cell_42"],
        config,
        black,
      );

      expect(Array.from(packedColors)).toEqual(
        ["a", "b", "cell_42"].map((id) =>
          ColorResolver.packColor(
            builtInPalette.colors[
              HashUtils.hash(id, 7) % builtInPalette.colors.length
            ]!,
          ),
        ),
      );
    });

    it("is deterministic across calls and independent of the ID order", async () => {
      const config = {
        random: { seed: 3, palette: builtInPalette.id },
      } satisfies ColorConfig;

      const first = await ColorResolver.resolveRandomColors(
        new Uint32Array([1, 2, 3]),
        config,
        black,
      );
      const second = await ColorResolver.resolveRandomColors(
        new Uint32Array([3, 2, 1]),
        config,
        black,
      );

      expect(Array.from(second)).toEqual(Array.from(first).reverse());
    });

    it("returns uniform default color when the palette is not found", async () => {
      const config = {
        random: { seed: 0, palette: "nonexistent" },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveRandomColors(
        new Uint32Array([1, 2]),
        config,
        red,
      );

      const packedColor = ColorResolver.packColor(red);
      expect(Array.from(packedColors)).toEqual([packedColor, packedColor]);
    });
  });

  describe("resolveColors", () => {
    it("dispatches to constant, leaving the alpha channel to the caller", async () => {
      const ids = new Uint32Array([1, 2]);
      const config = { constant: { value: red } } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveColors(
        ids,
        config,
        [],
        black,
      );

      const packedRed = ColorResolver.packColor(red);
      expect(Array.from(packedColors)).toEqual([packedRed, packedRed]);
    });

    it("dispatches to from config when loadTable is given", async () => {
      const palette = colorPalettes[0]!;
      const data = createMockTableData(new Uint32Array([1]), [0], [0, 1]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = {
        from: { column: "col1", palette: palette.id },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveColors(
        new Uint32Array([1]),
        config,
        [],
        black,
        {
          loadTable,
        },
      );

      expect(loadTable).toHaveBeenCalledOnce();
      const expectedColor = palette.colors[0]!;
      expect(packedColors[0]).toBe(ColorResolver.packColor(expectedColor));
    });

    it("dispatches to groupBy config when loadTable is given", async () => {
      const ids = new Uint32Array([1]);
      const data = createMockTableData(ids, ["cat-a"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const colorMap: GroupValueMap<Color> = {
        id: "cm1",
        name: "CM",
        values: { "cat-a": red },
      };
      const config = {
        groupBy: { column: "col1", map: "cm1" },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveColors(
        ids,
        config,
        [colorMap],
        black,
        { loadTable },
      );

      expect(loadTable).toHaveBeenCalledOnce();
      expect(packedColors[0]).toBe(ColorResolver.packColor(red));
    });

    it("dispatches to random config", async () => {
      // resolveRandomColors samples from the built-in colorPalettes, not a
      // parameter, so a palette that is not among them falls back to the default
      const builtInPalette = colorPalettes[0]!;
      const config = {
        random: { seed: 0, palette: builtInPalette.id },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveColors(
        new Uint32Array([1]),
        config,
        [],
        black,
      );

      expect(packedColors.length).toBe(1);
      expect(
        builtInPalette.colors.map((color) => ColorResolver.packColor(color)),
      ).toContain(packedColors[0]);
    });

    it("falls back to the default color when the config has no active source", async () => {
      const config = {} as ColorConfig;

      const packedColors = await ColorResolver.resolveColors(
        new Uint32Array([1]),
        config,
        [],
        red,
      );

      expect(packedColors[0]).toBe(ColorResolver.packColor(red));
    });

    it("falls back to the default color for a from config without loadTable", async () => {
      const config = {
        from: { column: "col1", palette: colorPalettes[0]!.id },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveColors(
        new Uint32Array([1]),
        config,
        [],
        red,
      );

      expect(packedColors[0]).toBe(ColorResolver.packColor(red));
    });

    it("falls back to the default color for a groupBy config without loadTable", async () => {
      const colorMap: GroupValueMap<Color> = {
        id: "cm1",
        name: "CM",
        values: { "cat-a": green },
      };
      const config = {
        groupBy: { column: "col1", map: "cm1" },
      } satisfies ColorConfig;

      const packedColors = await ColorResolver.resolveColors(
        new Uint32Array([1]),
        config,
        [colorMap],
        red,
        {},
      );

      expect(packedColors[0]).toBe(ColorResolver.packColor(red));
    });

    it("throws when the signal is already aborted", async () => {
      const controller = new AbortController();
      controller.abort();
      const config = { constant: { value: red } } satisfies ColorConfig;

      await expect(
        ColorResolver.resolveColors(new Uint32Array([1]), config, [], black, {
          signal: controller.signal,
        }),
      ).rejects.toThrow();
    });
  });

  describe("resolveConstantColor", () => {
    it("returns the packed color for a constant config", () => {
      const config = { constant: { value: red } } satisfies ColorConfig;
      expect(ColorResolver.resolveConstantColor(config)).toBe(
        ColorUtils.packColor(red),
      );
    });

    it("returns undefined for table-backed and random configs", () => {
      const fromConfig = {
        from: { column: "col1", palette: colorPalettes[0]!.id },
      } satisfies ColorConfig;
      const randomConfig = {
        random: { palette: colorPalettes[0]!.id },
      } satisfies ColorConfig;
      expect(ColorResolver.resolveConstantColor(fromConfig)).toBeUndefined();
      expect(ColorResolver.resolveConstantColor(randomConfig)).toBeUndefined();
    });
  });

  describe("resolveColorWithoutTable", () => {
    it("returns the packed constant color for a constant config", () => {
      const config = { constant: { value: red } } satisfies ColorConfig;
      expect(ColorResolver.resolveColorWithoutTable(1, config, black)).toBe(
        ColorUtils.packColor(red),
      );
    });

    it("draws the color via pickRandomColor for a random config", () => {
      const builtInPalette = colorPalettes[0]!;
      const config = {
        random: { palette: builtInPalette.id, seed: 7 },
      } satisfies ColorConfig;
      expect(ColorResolver.resolveColorWithoutTable(42, config, black)).toBe(
        ColorUtils.packColor(
          ColorResolver.pickRandomColor(42, 7, builtInPalette),
        ),
      );
    });

    it("uses the default seed for a random config without a seed", () => {
      const builtInPalette = colorPalettes[0]!;
      const config = {
        random: { palette: builtInPalette.id },
      } satisfies ColorConfig;
      expect(ColorResolver.resolveColorWithoutTable(42, config, black)).toBe(
        ColorUtils.packColor(
          ColorResolver.pickRandomColor(42, defaultRandomSeed, builtInPalette),
        ),
      );
    });

    it("falls back to the default color for an unknown random palette", () => {
      const config = {
        random: { palette: "nonexistent", seed: 7 },
      } satisfies ColorConfig;
      expect(ColorResolver.resolveColorWithoutTable(42, config, blue)).toBe(
        ColorUtils.packColor(blue),
      );
    });

    it("falls back to the default color for table-backed configs", () => {
      const fromConfig = {
        from: { column: "col1", palette: colorPalettes[0]!.id },
      } satisfies ColorConfig;
      const groupByConfig = {
        groupBy: { column: "col1", map: "cm1" },
      } satisfies ColorConfig;
      expect(ColorResolver.resolveColorWithoutTable(1, fromConfig, green)).toBe(
        ColorUtils.packColor(green),
      );
      expect(
        ColorResolver.resolveColorWithoutTable(1, groupByConfig, green),
      ).toBe(ColorUtils.packColor(green));
    });
  });

  describe("pickRandomColor", () => {
    it("picks the color selected by the seeded hash of the ID", () => {
      expect(ColorResolver.pickRandomColor(42, 3, testPalette)).toBe(
        testPalette.colors[HashUtils.mix(42, 3) % testPalette.colors.length],
      );
    });

    it("is deterministic", () => {
      expect(ColorResolver.pickRandomColor(42, 3, testPalette)).toBe(
        ColorResolver.pickRandomColor(42, 3, testPalette),
      );
    });

    it("picks colors from the palette", () => {
      for (let id = 1; id <= 20; id++) {
        expect(testPalette.colors).toContain(
          ColorResolver.pickRandomColor(id, 0, testPalette),
        );
      }
    });

    it("depends on the seed", () => {
      const picks = new Set(
        Array.from({ length: 16 }, (_, seed) =>
          ColorResolver.pickRandomColor(42, seed, testPalette),
        ),
      );
      expect(picks.size).toBeGreaterThan(1);
    });
  });
});
