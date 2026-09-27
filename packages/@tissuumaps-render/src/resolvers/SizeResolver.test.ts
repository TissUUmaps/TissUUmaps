import { describe, expect, it, vi } from "vitest";

import type {
  GroupValueMap,
  IDArray,
  SizeConfig,
  TableData,
} from "@tissuumaps/core";

import { SizeResolver } from "./SizeResolver";

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

describe("SizeResolver", () => {
  describe("parseSize", () => {
    it("returns numeric values unchanged", () => {
      expect(SizeResolver.parseSize(5)).toBe(5);
      expect(SizeResolver.parseSize(0)).toBe(0);
      expect(SizeResolver.parseSize(-2.5)).toBe(-2.5);
    });

    it("parses numeric strings", () => {
      expect(SizeResolver.parseSize("5")).toBe(5);
      expect(SizeResolver.parseSize("-2.5")).toBe(-2.5);
    });

    it("returns undefined for values that are not finite numbers", () => {
      expect(SizeResolver.parseSize("five")).toBeUndefined();
      expect(SizeResolver.parseSize(NaN)).toBeUndefined();
      expect(SizeResolver.parseSize(Infinity)).toBeUndefined();
      expect(SizeResolver.parseSize(null)).toBeUndefined();
      expect(SizeResolver.parseSize(undefined)).toBeUndefined();
    });
  });

  describe("packSize", () => {
    it("returns the size unchanged", () => {
      expect(SizeResolver.packSize(10)).toBe(10);
    });
  });

  describe("createSizeBuffer", () => {
    it("creates a zeroed Float32Array of the requested size", () => {
      const packedSizes = SizeResolver.createSizeBuffer(3);
      expect(packedSizes).toBeInstanceOf(Float32Array);
      expect(Array.from(packedSizes)).toEqual([0, 0, 0]);
    });

    it("aligns the buffer length to the given boundary", () => {
      expect(SizeResolver.createSizeBuffer(3, { align: 4 }).length).toBe(4);
    });
  });

  describe("createUniformSizes", () => {
    it("fills the buffer with the packed size", () => {
      const packedSizes = SizeResolver.createUniformSizes(3, 7);
      expect(Array.from(packedSizes)).toEqual([7, 7, 7]);
    });
  });

  describe("resolveUniformSizes", () => {
    it("fills the buffer with the constant size", () => {
      const config = { constant: { value: 9 } } satisfies SizeConfig;
      const packedSizes = SizeResolver.resolveUniformSizes(
        new Uint32Array([1, 2]),
        config,
      );
      expect(Array.from(packedSizes)).toEqual([9, 9]);
    });
  });

  describe("resolveSizesFromTableValues", () => {
    it("reads sizes from the table column", async () => {
      const ids = new Uint32Array([1, 2]);
      const data = createMockTableData(ids, [3, 4]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies SizeConfig;

      const packedSizes = await SizeResolver.resolveSizesFromTableValues(
        ids,
        config,
        1,
        loadTable,
      );

      expect(Array.from(packedSizes)).toEqual([3, 4]);
    });

    it("uses the default size for invalid values", async () => {
      const ids = new Uint32Array([1, 2]);
      const data = createMockTableData(ids, ["bad", 4]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies SizeConfig;

      const packedSizes = await SizeResolver.resolveSizesFromTableValues(
        ids,
        config,
        99,
        loadTable,
      );

      expect(Array.from(packedSizes)).toEqual([99, 4]);
    });

    it("forwards the signal to loadTable", async () => {
      const controller = new AbortController();
      const data = createMockTableData(new Uint32Array([1]), [3]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies SizeConfig;

      await SizeResolver.resolveSizesFromTableValues(
        new Uint32Array([1]),
        config,
        1,
        loadTable,
        { signal: controller.signal },
      );

      expect(loadTable).toHaveBeenCalledWith({ signal: controller.signal });
    });
  });

  describe("resolveSizesFromTableGroups", () => {
    it("maps groups to sizes using the size map", async () => {
      const ids = new Uint32Array([1, 2]);
      const data = createMockTableData(ids, ["A", "B"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const sizeMap: GroupValueMap<number> = {
        id: "sm1",
        name: "Size Map",
        values: {
          A: 2,
          B: 4,
        },
      };
      const config = {
        groupBy: { column: "col1", map: "sm1" },
      } satisfies SizeConfig;

      const packedSizes = await SizeResolver.resolveSizesFromTableGroups(
        ids,
        config,
        [sizeMap],
        1,
        loadTable,
      );

      expect(Array.from(packedSizes)).toEqual([2, 4]);
    });

    it("uses the size map default for unmapped groups", async () => {
      const ids = new Uint32Array([1]);
      const data = createMockTableData(ids, ["missing"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const sizeMap: GroupValueMap<number> = {
        id: "sm1",
        name: "Size Map",
        values: {},
        default: 7,
      };
      const config = {
        groupBy: { column: "col1", map: "sm1" },
      } satisfies SizeConfig;

      const packedSizes = await SizeResolver.resolveSizesFromTableGroups(
        ids,
        config,
        [sizeMap],
        1,
        loadTable,
      );

      expect(packedSizes[0]).toBe(7);
    });

    it("returns uniform default size when the map is not found", async () => {
      const loadTable = vi.fn();
      const config = {
        groupBy: { column: "col1", map: "nonexistent" },
      } satisfies SizeConfig;

      const packedSizes = await SizeResolver.resolveSizesFromTableGroups(
        new Uint32Array([1, 2]),
        config,
        [],
        5,
        loadTable,
      );

      expect(Array.from(packedSizes)).toEqual([5, 5]);
      expect(loadTable).not.toHaveBeenCalled();
    });
  });

  describe("resolveConstantSize", () => {
    it("returns the packed size for a constant config", () => {
      const config = { constant: { value: 9 } } satisfies SizeConfig;
      expect(SizeResolver.resolveConstantSize(config)).toBe(9);
    });

    it("returns undefined for table-backed configs", () => {
      const config = { from: { column: "col1" } } satisfies SizeConfig;
      expect(SizeResolver.resolveConstantSize(config)).toBeUndefined();
    });
  });

  describe("resolveSizeWithoutTable", () => {
    it("returns the packed constant size for a constant config", () => {
      const config = { constant: { value: 9 } } satisfies SizeConfig;
      expect(SizeResolver.resolveSizeWithoutTable(1, config, 3)).toBe(9);
    });

    it("falls back to the default size for table-backed configs", () => {
      const fromConfig = { from: { column: "col1" } } satisfies SizeConfig;
      const groupByConfig = {
        groupBy: { column: "col1", map: "sm1" },
      } satisfies SizeConfig;
      expect(SizeResolver.resolveSizeWithoutTable(1, fromConfig, 3)).toBe(3);
      expect(SizeResolver.resolveSizeWithoutTable(1, groupByConfig, 3)).toBe(3);
    });
  });

  describe("resolveSizes", () => {
    it("dispatches to constant", async () => {
      const config = { constant: { value: 8 } } satisfies SizeConfig;
      const packedSizes = await SizeResolver.resolveSizes(
        new Uint32Array([1, 2]),
        config,
        [],
        1,
      );
      expect(Array.from(packedSizes)).toEqual([8, 8]);
    });

    it("dispatches to from config when loadTable is given", async () => {
      const data = createMockTableData(new Uint32Array([1]), [3]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies SizeConfig;

      const packedSizes = await SizeResolver.resolveSizes(
        new Uint32Array([1]),
        config,
        [],
        1,
        {
          loadTable,
        },
      );

      expect(loadTable).toHaveBeenCalledOnce();
      expect(packedSizes[0]).toBe(3);
    });

    it("dispatches to groupBy config when loadTable is given", async () => {
      const data = createMockTableData(new Uint32Array([1]), ["A"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const sizeMap: GroupValueMap<number> = {
        id: "sm1",
        name: "Size Map",
        values: { A: 5 },
      };
      const config = {
        groupBy: { column: "col1", map: "sm1" },
      } satisfies SizeConfig;

      const packedSizes = await SizeResolver.resolveSizes(
        new Uint32Array([1]),
        config,
        [sizeMap],
        1,
        { loadTable },
      );

      expect(loadTable).toHaveBeenCalledOnce();
      expect(packedSizes[0]).toBe(5);
    });

    it("falls back to the default size when the config has no active source", async () => {
      const config = {} as SizeConfig;
      const packedSizes = await SizeResolver.resolveSizes(
        new Uint32Array([1, 2]),
        config,
        [],
        3,
      );
      expect(Array.from(packedSizes)).toEqual([3, 3]);
    });

    it("falls back to the default size for a from config without loadTable", async () => {
      const config = { from: { column: "col1" } } satisfies SizeConfig;

      const packedSizes = await SizeResolver.resolveSizes(
        new Uint32Array([1]),
        config,
        [],
        3,
      );

      expect(packedSizes[0]).toBe(3);
    });

    it("falls back to the default size for a groupBy config without loadTable", async () => {
      const sizeMap: GroupValueMap<number> = {
        id: "sm1",
        name: "Size Map",
        values: { A: 5 },
      };
      const config = {
        groupBy: { column: "col1", map: "sm1" },
      } satisfies SizeConfig;

      const packedSizes = await SizeResolver.resolveSizes(
        new Uint32Array([1]),
        config,
        [sizeMap],
        3,
        {},
      );

      expect(packedSizes[0]).toBe(3);
    });

    it("throws when the signal is already aborted", async () => {
      const controller = new AbortController();
      controller.abort();
      const config = { constant: { value: 8 } } satisfies SizeConfig;

      await expect(
        SizeResolver.resolveSizes(new Uint32Array([1]), config, [], 1, {
          signal: controller.signal,
        }),
      ).rejects.toThrow();
    });
  });
});
