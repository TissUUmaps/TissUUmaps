import { describe, expect, it, vi } from "vitest";

import type {
  GroupValueMap,
  IDArray,
  TableData,
  VisibilityConfig,
} from "@tissuumaps/core";

import { VisibilityResolver } from "./VisibilityResolver";

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

describe("VisibilityResolver", () => {
  describe("parseVisibility", () => {
    it("treats positive numbers as visible", () => {
      expect(VisibilityResolver.parseVisibility(1)).toBe(true);
      expect(VisibilityResolver.parseVisibility(5)).toBe(true);
    });

    it("treats zero and negatives as not visible", () => {
      expect(VisibilityResolver.parseVisibility(0)).toBe(false);
      expect(VisibilityResolver.parseVisibility(-1)).toBe(false);
    });

    it("passes booleans through", () => {
      expect(VisibilityResolver.parseVisibility(true)).toBe(true);
      expect(VisibilityResolver.parseVisibility(false)).toBe(false);
    });

    it("parses numeric strings", () => {
      expect(VisibilityResolver.parseVisibility("1")).toBe(true);
      expect(VisibilityResolver.parseVisibility("0")).toBe(false);
    });

    it("returns undefined for values that are not finite numbers", () => {
      expect(VisibilityResolver.parseVisibility("yes")).toBeUndefined();
      expect(VisibilityResolver.parseVisibility(NaN)).toBeUndefined();
      expect(VisibilityResolver.parseVisibility(null)).toBeUndefined();
    });
  });

  describe("packVisibility", () => {
    it("packs booleans as 1 or 0", () => {
      expect(VisibilityResolver.packVisibility(true)).toBe(1);
      expect(VisibilityResolver.packVisibility(false)).toBe(0);
    });
  });

  describe("createVisibilityBuffer", () => {
    it("creates a zeroed Uint8Array of the requested size", () => {
      const packedVisibilities = VisibilityResolver.createVisibilityBuffer(3);
      expect(packedVisibilities).toBeInstanceOf(Uint8Array);
      expect(Array.from(packedVisibilities)).toEqual([0, 0, 0]);
    });

    it("aligns the buffer length to the given boundary", () => {
      expect(
        VisibilityResolver.createVisibilityBuffer(3, { align: 4 }).length,
      ).toBe(4);
    });
  });

  describe("createUniformVisibilities", () => {
    it("fills the buffer with the packed visibility", () => {
      expect(
        Array.from(VisibilityResolver.createUniformVisibilities(3, true)),
      ).toEqual([1, 1, 1]);
      expect(
        Array.from(VisibilityResolver.createUniformVisibilities(2, false)),
      ).toEqual([0, 0]);
    });
  });

  describe("resolveUniformVisibilities", () => {
    it("fills the buffer with the constant visibility", () => {
      const config = { constant: { value: true } } satisfies VisibilityConfig;
      const packedVisibilities = VisibilityResolver.resolveUniformVisibilities(
        new Uint32Array([1, 2]),
        config,
      );
      expect(Array.from(packedVisibilities)).toEqual([1, 1]);
    });
  });

  describe("resolveVisibilitiesFromTableValues", () => {
    it("reads visibilities from the table column", async () => {
      const ids = new Uint32Array([1, 2, 3]);
      const data = createMockTableData(ids, [1, 0, 5]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies VisibilityConfig;

      const packedVisibilities =
        await VisibilityResolver.resolveVisibilitiesFromTableValues(
          ids,
          config,
          false,
          loadTable,
        );

      expect(Array.from(packedVisibilities)).toEqual([1, 0, 1]);
    });

    it("uses the default visibility for invalid values", async () => {
      const ids = new Uint32Array([1, 2]);
      const data = createMockTableData(ids, ["bad", 0]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies VisibilityConfig;

      const packedVisibilities =
        await VisibilityResolver.resolveVisibilitiesFromTableValues(
          ids,
          config,
          true,
          loadTable,
        );

      expect(packedVisibilities[0]).toBe(1); // "bad" → default true
      expect(packedVisibilities[1]).toBe(0);
    });

    it("forwards the signal to loadTable", async () => {
      const controller = new AbortController();
      const data = createMockTableData(new Uint32Array([1]), [1]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies VisibilityConfig;

      await VisibilityResolver.resolveVisibilitiesFromTableValues(
        new Uint32Array([1]),
        config,
        false,
        loadTable,
        { signal: controller.signal },
      );

      expect(loadTable).toHaveBeenCalledWith({ signal: controller.signal });
    });
  });

  describe("resolveVisibilitiesFromTableGroups", () => {
    it("maps groups to visibilities using the visibility map", async () => {
      const ids = new Uint32Array([1, 2]);
      const data = createMockTableData(ids, ["A", "B"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const visibilityMap: GroupValueMap<boolean> = {
        id: "vm1",
        name: "Visibility Map",
        values: {
          A: true,
          B: false,
        },
      };
      const config = {
        groupBy: { column: "col1", map: "vm1" },
      } satisfies VisibilityConfig;

      const packedVisibilities =
        await VisibilityResolver.resolveVisibilitiesFromTableGroups(
          ids,
          config,
          [visibilityMap],
          false,
          loadTable,
        );

      expect(Array.from(packedVisibilities)).toEqual([1, 0]);
    });

    it("uses the visibility map default for unmapped groups", async () => {
      const ids = new Uint32Array([1]);
      const data = createMockTableData(ids, ["missing"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const visibilityMap: GroupValueMap<boolean> = {
        id: "vm1",
        name: "Visibility Map",
        values: {},
        default: true,
      };
      const config = {
        groupBy: { column: "col1", map: "vm1" },
      } satisfies VisibilityConfig;

      const packedVisibilities =
        await VisibilityResolver.resolveVisibilitiesFromTableGroups(
          ids,
          config,
          [visibilityMap],
          false,
          loadTable,
        );

      expect(packedVisibilities[0]).toBe(1);
    });

    it("returns uniform default visibility when the map is not found", async () => {
      const loadTable = vi.fn();
      const config = {
        groupBy: { column: "col1", map: "nonexistent" },
      } satisfies VisibilityConfig;

      const packedVisibilities =
        await VisibilityResolver.resolveVisibilitiesFromTableGroups(
          new Uint32Array([1, 2]),
          config,
          [],
          true,
          loadTable,
        );

      expect(Array.from(packedVisibilities)).toEqual([1, 1]);
      expect(loadTable).not.toHaveBeenCalled();
    });
  });

  describe("resolveConstantVisibility", () => {
    it("returns the packed visibility for a constant config", () => {
      const config = { constant: { value: false } } satisfies VisibilityConfig;
      expect(VisibilityResolver.resolveConstantVisibility(config)).toBe(0);
    });

    it("returns undefined for table-backed configs", () => {
      const config = { from: { column: "col1" } } satisfies VisibilityConfig;
      expect(
        VisibilityResolver.resolveConstantVisibility(config),
      ).toBeUndefined();
    });
  });

  describe("resolveVisibilityWithoutTable", () => {
    it("returns the packed constant visibility for a constant config", () => {
      const config = { constant: { value: false } } satisfies VisibilityConfig;
      expect(
        VisibilityResolver.resolveVisibilityWithoutTable(1, config, true),
      ).toBe(0);
    });

    it("falls back to the default visibility for table-backed configs", () => {
      const fromConfig = {
        from: { column: "col1" },
      } satisfies VisibilityConfig;
      const groupByConfig = {
        groupBy: { column: "col1", map: "vm1" },
      } satisfies VisibilityConfig;
      expect(
        VisibilityResolver.resolveVisibilityWithoutTable(1, fromConfig, true),
      ).toBe(1);
      expect(
        VisibilityResolver.resolveVisibilityWithoutTable(
          1,
          groupByConfig,
          false,
        ),
      ).toBe(0);
    });
  });

  describe("resolveVisibilities", () => {
    it("dispatches to constant", async () => {
      const config = { constant: { value: false } } satisfies VisibilityConfig;
      const packedVisibilities = await VisibilityResolver.resolveVisibilities(
        new Uint32Array([1, 2]),
        config,
        [],
        true,
      );
      expect(Array.from(packedVisibilities)).toEqual([0, 0]);
    });

    it("dispatches to from config when loadTable is given", async () => {
      const data = createMockTableData(new Uint32Array([1]), [1]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies VisibilityConfig;

      const packedVisibilities = await VisibilityResolver.resolveVisibilities(
        new Uint32Array([1]),
        config,
        [],
        false,
        { loadTable },
      );

      expect(loadTable).toHaveBeenCalledOnce();
      expect(packedVisibilities[0]).toBe(1);
    });

    it("dispatches to groupBy config when loadTable is given", async () => {
      const data = createMockTableData(new Uint32Array([1]), ["A"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const visibilityMap: GroupValueMap<boolean> = {
        id: "vm1",
        name: "Visibility Map",
        values: { A: true },
      };
      const config = {
        groupBy: { column: "col1", map: "vm1" },
      } satisfies VisibilityConfig;

      const packedVisibilities = await VisibilityResolver.resolveVisibilities(
        new Uint32Array([1]),
        config,
        [visibilityMap],
        false,
        { loadTable },
      );

      expect(loadTable).toHaveBeenCalledOnce();
      expect(packedVisibilities[0]).toBe(1);
    });

    it("falls back to the default visibility when the config has no active source", async () => {
      const config = {} as VisibilityConfig;
      const packedVisibilities = await VisibilityResolver.resolveVisibilities(
        new Uint32Array([1, 2]),
        config,
        [],
        true,
      );
      expect(Array.from(packedVisibilities)).toEqual([1, 1]);
    });

    it("falls back to the default visibility for a from config without loadTable", async () => {
      const config = { from: { column: "col1" } } satisfies VisibilityConfig;

      const packedVisibilities = await VisibilityResolver.resolveVisibilities(
        new Uint32Array([1]),
        config,
        [],
        true,
      );

      expect(packedVisibilities[0]).toBe(1);
    });

    it("falls back to the default visibility for a groupBy config without loadTable", async () => {
      const visibilityMap: GroupValueMap<boolean> = {
        id: "vm1",
        name: "Visibility Map",
        values: { A: false },
      };
      const config = {
        groupBy: { column: "col1", map: "vm1" },
      } satisfies VisibilityConfig;

      const packedVisibilities = await VisibilityResolver.resolveVisibilities(
        new Uint32Array([1]),
        config,
        [visibilityMap],
        true,
        {},
      );

      expect(packedVisibilities[0]).toBe(1);
    });

    it("throws when the signal is already aborted", async () => {
      const controller = new AbortController();
      controller.abort();
      const config = { constant: { value: true } } satisfies VisibilityConfig;

      await expect(
        VisibilityResolver.resolveVisibilities(
          new Uint32Array([1]),
          config,
          [],
          false,
          {
            signal: controller.signal,
          },
        ),
      ).rejects.toThrow();
    });
  });
});
