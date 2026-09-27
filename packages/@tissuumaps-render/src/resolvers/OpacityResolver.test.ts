import { describe, expect, it, vi } from "vitest";

import type {
  GroupValueMap,
  IDArray,
  OpacityConfig,
  TableData,
} from "@tissuumaps/core";

import { OpacityResolver } from "./OpacityResolver";

function createMockTableData(ids: IDArray, values: unknown[]): TableData {
  return {
    getIds: () => ids,
    getSize: () => ids.length,
    getNames: () => undefined,
    close: vi.fn(),
    loadValues: vi.fn().mockResolvedValue(values),
    loadValueRange: vi.fn().mockResolvedValue(undefined),
    loadUniqueValueCounts: vi.fn(),
    suggestColumnQueries: vi.fn(),
    resolveColumnQuery: vi.fn(),
  };
}

describe("OpacityResolver", () => {
  describe("parseOpacity", () => {
    it("returns numeric values within [0, 1] unchanged", () => {
      expect(OpacityResolver.parseOpacity(0.5)).toBe(0.5);
      expect(OpacityResolver.parseOpacity(0)).toBe(0);
      expect(OpacityResolver.parseOpacity(1)).toBe(1);
    });

    it("clamps out-of-range numbers to [0, 1]", () => {
      expect(OpacityResolver.parseOpacity(2)).toBe(1);
      expect(OpacityResolver.parseOpacity(-1)).toBe(0);
    });

    it("parses numeric strings", () => {
      expect(OpacityResolver.parseOpacity("0.5")).toBe(0.5);
    });

    it("returns undefined for values that are not finite numbers", () => {
      expect(OpacityResolver.parseOpacity("half")).toBeUndefined();
      expect(OpacityResolver.parseOpacity(NaN)).toBeUndefined();
      expect(OpacityResolver.parseOpacity(null)).toBeUndefined();
    });
  });

  describe("packOpacity", () => {
    it("scales opacity (0–1) into the 0–255 range", () => {
      expect(OpacityResolver.packOpacity(0)).toBe(0);
      expect(OpacityResolver.packOpacity(1)).toBe(255);
      expect(OpacityResolver.packOpacity(0.5)).toBe(128); // round(127.5)
    });

    it("clamps the packed value to [0, 255]", () => {
      expect(OpacityResolver.packOpacity(2)).toBe(255);
      expect(OpacityResolver.packOpacity(-1)).toBe(0);
    });
  });

  describe("createOpacityBuffer", () => {
    it("creates a zeroed Uint8Array of the requested size", () => {
      const packedOpacities = OpacityResolver.createOpacityBuffer(3);
      expect(packedOpacities).toBeInstanceOf(Uint8Array);
      expect(Array.from(packedOpacities)).toEqual([0, 0, 0]);
    });

    it("aligns the buffer length to the given boundary", () => {
      expect(OpacityResolver.createOpacityBuffer(3, { align: 4 }).length).toBe(
        4,
      );
    });
  });

  describe("createUniformOpacities", () => {
    it("fills the buffer with the packed opacity", () => {
      const packedOpacities = OpacityResolver.createUniformOpacities(3, 1);
      expect(Array.from(packedOpacities)).toEqual([255, 255, 255]);
    });
  });

  describe("resolveUniformOpacities", () => {
    it("fills the buffer with the constant opacity", () => {
      const config = { constant: { value: 1 } } satisfies OpacityConfig;
      const packedOpacities = OpacityResolver.resolveUniformOpacities(
        new Uint32Array([1, 2]),
        config,
      );
      expect(Array.from(packedOpacities)).toEqual([255, 255]);
    });
  });

  describe("resolveOpacitiesFromTableValues", () => {
    it("reads opacities from the table column", async () => {
      const ids = new Uint32Array([1, 2]);
      const data = createMockTableData(ids, [0, 1]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies OpacityConfig;

      const packedOpacities =
        await OpacityResolver.resolveOpacitiesFromTableValues(
          ids,
          config,
          1,
          loadTable,
        );

      expect(Array.from(packedOpacities)).toEqual([0, 255]);
    });

    it("uses the default opacity for invalid values", async () => {
      const ids = new Uint32Array([1, 2]);
      const data = createMockTableData(ids, ["bad", 1]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies OpacityConfig;

      const packedOpacities =
        await OpacityResolver.resolveOpacitiesFromTableValues(
          ids,
          config,
          0,
          loadTable,
        );

      expect(packedOpacities[0]).toBe(0); // default 0 → packed 0
      expect(packedOpacities[1]).toBe(255);
    });

    it("forwards the signal to loadTable", async () => {
      const controller = new AbortController();
      const data = createMockTableData(new Uint32Array([1]), [1]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies OpacityConfig;

      await OpacityResolver.resolveOpacitiesFromTableValues(
        new Uint32Array([1]),
        config,
        1,
        loadTable,
        { signal: controller.signal },
      );

      expect(loadTable).toHaveBeenCalledWith({ signal: controller.signal });
    });
  });

  describe("resolveOpacitiesFromTableGroups", () => {
    it("maps groups to opacities using the opacity map", async () => {
      const ids = new Uint32Array([1, 2]);
      const data = createMockTableData(ids, ["A", "B"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const opacityMap: GroupValueMap<number> = {
        id: "om1",
        name: "Opacity Map",
        values: {
          A: 1,
          B: 0.5,
        },
      };
      const config = {
        groupBy: { column: "col1", map: "om1" },
      } satisfies OpacityConfig;

      const packedOpacities =
        await OpacityResolver.resolveOpacitiesFromTableGroups(
          ids,
          config,
          [opacityMap],
          0,
          loadTable,
        );

      expect(Array.from(packedOpacities)).toEqual([255, 128]);
    });

    it("uses the opacity map default for unmapped groups", async () => {
      const ids = new Uint32Array([1]);
      const data = createMockTableData(ids, ["missing"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const opacityMap: GroupValueMap<number> = {
        id: "om1",
        name: "Opacity Map",
        values: {},
        default: 1,
      };
      const config = {
        groupBy: { column: "col1", map: "om1" },
      } satisfies OpacityConfig;

      const packedOpacities =
        await OpacityResolver.resolveOpacitiesFromTableGroups(
          ids,
          config,
          [opacityMap],
          0,
          loadTable,
        );

      expect(packedOpacities[0]).toBe(255);
    });

    it("returns uniform default opacity when the map is not found", async () => {
      const loadTable = vi.fn();
      const config = {
        groupBy: { column: "col1", map: "nonexistent" },
      } satisfies OpacityConfig;

      const packedOpacities =
        await OpacityResolver.resolveOpacitiesFromTableGroups(
          new Uint32Array([1, 2]),
          config,
          [],
          1,
          loadTable,
        );

      expect(Array.from(packedOpacities)).toEqual([255, 255]);
      expect(loadTable).not.toHaveBeenCalled();
    });
  });

  describe("resolveConstantOpacity", () => {
    it("returns the packed opacity for a constant config", () => {
      const config = { constant: { value: 1 } } satisfies OpacityConfig;
      expect(OpacityResolver.resolveConstantOpacity(config)).toBe(255);
    });

    it("returns undefined for table-backed configs", () => {
      const config = { from: { column: "col1" } } satisfies OpacityConfig;
      expect(OpacityResolver.resolveConstantOpacity(config)).toBeUndefined();
    });
  });

  describe("resolveOpacityWithoutTable", () => {
    it("returns the packed constant opacity for a constant config", () => {
      const config = { constant: { value: 1 } } satisfies OpacityConfig;
      expect(OpacityResolver.resolveOpacityWithoutTable(1, config, 0)).toBe(
        255,
      );
    });

    it("falls back to the default opacity for table-backed configs", () => {
      const fromConfig = { from: { column: "col1" } } satisfies OpacityConfig;
      const groupByConfig = {
        groupBy: { column: "col1", map: "om1" },
      } satisfies OpacityConfig;
      expect(
        OpacityResolver.resolveOpacityWithoutTable(1, fromConfig, 0.5),
      ).toBe(OpacityResolver.packOpacity(0.5));
      expect(
        OpacityResolver.resolveOpacityWithoutTable(1, groupByConfig, 0.5),
      ).toBe(OpacityResolver.packOpacity(0.5));
    });
  });

  describe("resolveOpacities", () => {
    it("dispatches to constant", async () => {
      const config = { constant: { value: 1 } } satisfies OpacityConfig;
      const packedOpacities = await OpacityResolver.resolveOpacities(
        new Uint32Array([1, 2]),
        config,
        [],
        0,
      );
      expect(Array.from(packedOpacities)).toEqual([255, 255]);
    });

    it("dispatches to from config when loadTable is given", async () => {
      const data = createMockTableData(new Uint32Array([1]), [0.5]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies OpacityConfig;

      const packedOpacities = await OpacityResolver.resolveOpacities(
        new Uint32Array([1]),
        config,
        [],
        0,
        { loadTable },
      );

      expect(loadTable).toHaveBeenCalledOnce();
      expect(packedOpacities[0]).toBe(128);
    });

    it("dispatches to groupBy config when loadTable is given", async () => {
      const data = createMockTableData(new Uint32Array([1]), ["A"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const opacityMap: GroupValueMap<number> = {
        id: "om1",
        name: "Opacity Map",
        values: { A: 1 },
      };
      const config = {
        groupBy: { column: "col1", map: "om1" },
      } satisfies OpacityConfig;

      const packedOpacities = await OpacityResolver.resolveOpacities(
        new Uint32Array([1]),
        config,
        [opacityMap],
        0,
        { loadTable },
      );

      expect(loadTable).toHaveBeenCalledOnce();
      expect(packedOpacities[0]).toBe(255);
    });

    it("falls back to the default opacity when the config has no active source", async () => {
      const config = {} as OpacityConfig;
      const packedOpacities = await OpacityResolver.resolveOpacities(
        new Uint32Array([1, 2]),
        config,
        [],
        1,
      );
      expect(Array.from(packedOpacities)).toEqual([255, 255]);
    });

    it("falls back to the default opacity for a from config without loadTable", async () => {
      const config = { from: { column: "col1" } } satisfies OpacityConfig;

      const packedOpacities = await OpacityResolver.resolveOpacities(
        new Uint32Array([1]),
        config,
        [],
        1,
      );

      expect(packedOpacities[0]).toBe(255);
    });

    it("falls back to the default opacity for a groupBy config without loadTable", async () => {
      const opacityMap: GroupValueMap<number> = {
        id: "om1",
        name: "Opacity Map",
        values: { A: 0.5 },
      };
      const config = {
        groupBy: { column: "col1", map: "om1" },
      } satisfies OpacityConfig;

      const packedOpacities = await OpacityResolver.resolveOpacities(
        new Uint32Array([1]),
        config,
        [opacityMap],
        1,
        {},
      );

      expect(packedOpacities[0]).toBe(255);
    });

    it("throws when the signal is already aborted", async () => {
      const controller = new AbortController();
      controller.abort();
      const config = { constant: { value: 1 } } satisfies OpacityConfig;

      await expect(
        OpacityResolver.resolveOpacities(new Uint32Array([1]), config, [], 0, {
          signal: controller.signal,
        }),
      ).rejects.toThrow();
    });
  });
});
