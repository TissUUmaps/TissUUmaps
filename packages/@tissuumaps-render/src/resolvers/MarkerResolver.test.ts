import { describe, expect, it, vi } from "vitest";

import {
  type GroupValueMap,
  HashUtils,
  type IDArray,
  Marker,
  type MarkerConfig,
  type TableData,
  markerPalette,
} from "@tissuumaps/core";

import { MarkerResolver } from "./MarkerResolver";

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

describe("MarkerResolver", () => {
  describe("parseMarker", () => {
    it("returns numeric marker indices unchanged", () => {
      expect(MarkerResolver.parseMarker(Marker.Disc)).toBe(Marker.Disc);
      expect(MarkerResolver.parseMarker(0)).toBe(0);
    });

    it("parses integer strings", () => {
      expect(MarkerResolver.parseMarker("2")).toBe(Marker.Square);
    });

    it("returns undefined for values that are not safe integers", () => {
      expect(MarkerResolver.parseMarker("disc")).toBeUndefined();
      expect(MarkerResolver.parseMarker(1.5)).toBeUndefined();
      expect(MarkerResolver.parseMarker(NaN)).toBeUndefined();
      expect(MarkerResolver.parseMarker(null)).toBeUndefined();
    });
  });

  describe("packMarker", () => {
    it("returns the marker index as-is", () => {
      expect(MarkerResolver.packMarker(Marker.Star)).toBe(Marker.Star);
      expect(MarkerResolver.packMarker(Marker.Cross)).toBe(0);
    });
  });

  describe("createMarkerBuffer", () => {
    it("creates a zeroed Uint8Array of the requested size", () => {
      const packedMarkers = MarkerResolver.createMarkerBuffer(3);
      expect(packedMarkers).toBeInstanceOf(Uint8Array);
      expect(Array.from(packedMarkers)).toEqual([0, 0, 0]);
    });

    it("aligns the buffer length to the given boundary", () => {
      expect(MarkerResolver.createMarkerBuffer(3, { align: 4 }).length).toBe(4);
    });
  });

  describe("createUniformMarkers", () => {
    it("fills the buffer with the packed marker", () => {
      const packedMarkers = MarkerResolver.createUniformMarkers(
        3,
        Marker.Square,
      );
      expect(Array.from(packedMarkers)).toEqual([
        Marker.Square,
        Marker.Square,
        Marker.Square,
      ]);
    });
  });

  describe("resolveUniformMarkers", () => {
    it("fills the buffer with the constant marker", () => {
      const config = {
        constant: { value: Marker.Diamond },
      } satisfies MarkerConfig;
      const packedMarkers = MarkerResolver.resolveUniformMarkers(
        new Uint32Array([1, 2]),
        config,
      );
      expect(Array.from(packedMarkers)).toEqual([
        Marker.Diamond,
        Marker.Diamond,
      ]);
    });
  });

  describe("resolveMarkersFromTableValues", () => {
    it("reads markers from the table column", async () => {
      const ids = new Uint32Array([1, 2]);
      const data = createMockTableData(ids, [Marker.Disc, Marker.Star]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies MarkerConfig;

      const packedMarkers = await MarkerResolver.resolveMarkersFromTableValues(
        ids,
        config,
        Marker.Cross,
        loadTable,
      );

      expect(Array.from(packedMarkers)).toEqual([Marker.Disc, Marker.Star]);
    });

    it("uses the default marker for invalid values", async () => {
      const ids = new Uint32Array([1, 2]);
      const data = createMockTableData(ids, ["bad", Marker.Star]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies MarkerConfig;

      const packedMarkers = await MarkerResolver.resolveMarkersFromTableValues(
        ids,
        config,
        Marker.Ring,
        loadTable,
      );

      expect(packedMarkers[0]).toBe(Marker.Ring);
      expect(packedMarkers[1]).toBe(Marker.Star);
    });

    it("forwards the signal to loadTable", async () => {
      const controller = new AbortController();
      const data = createMockTableData(new Uint32Array([1]), [Marker.Star]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies MarkerConfig;

      await MarkerResolver.resolveMarkersFromTableValues(
        new Uint32Array([1]),
        config,
        Marker.Cross,
        loadTable,
        { signal: controller.signal },
      );

      expect(loadTable).toHaveBeenCalledWith({ signal: controller.signal });
    });
  });

  describe("resolveMarkersFromTableGroups", () => {
    it("maps groups to markers using the marker map", async () => {
      const ids = new Uint32Array([1, 2]);
      const data = createMockTableData(ids, ["A", "B"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const markerMap: GroupValueMap<Marker> = {
        id: "mm1",
        name: "Marker Map",
        values: {
          A: Marker.Disc,
          B: Marker.Square,
        },
      };
      const config = {
        groupBy: { column: "col1", map: "mm1" },
      } satisfies MarkerConfig;

      const packedMarkers = await MarkerResolver.resolveMarkersFromTableGroups(
        ids,
        config,
        [markerMap],
        Marker.Cross,
        loadTable,
      );

      expect(Array.from(packedMarkers)).toEqual([Marker.Disc, Marker.Square]);
    });

    it("uses the marker map default for unmapped groups", async () => {
      const ids = new Uint32Array([1]);
      const data = createMockTableData(ids, ["missing"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const markerMap: GroupValueMap<Marker> = {
        id: "mm1",
        name: "Marker Map",
        values: {},
        default: Marker.Ring,
      };
      const config = {
        groupBy: { column: "col1", map: "mm1" },
      } satisfies MarkerConfig;

      const packedMarkers = await MarkerResolver.resolveMarkersFromTableGroups(
        ids,
        config,
        [markerMap],
        Marker.Cross,
        loadTable,
      );

      expect(packedMarkers[0]).toBe(Marker.Ring);
    });

    it("returns uniform default marker when a map is specified but not found", async () => {
      const loadTable = vi.fn();
      const config = {
        groupBy: { column: "col1", map: "nonexistent" },
      } satisfies MarkerConfig;

      const packedMarkers = await MarkerResolver.resolveMarkersFromTableGroups(
        new Uint32Array([1, 2]),
        config,
        [],
        Marker.Star,
        loadTable,
      );

      expect(Array.from(packedMarkers)).toEqual([Marker.Star, Marker.Star]);
      expect(loadTable).not.toHaveBeenCalled();
    });

    it("hashes group names through the marker palette when no map is given", async () => {
      const ids = new Uint32Array([1, 2]);
      const data = createMockTableData(ids, ["groupA", "groupB"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = {
        groupBy: { column: "col1", map: undefined },
      } satisfies MarkerConfig;

      const packedMarkers = await MarkerResolver.resolveMarkersFromTableGroups(
        ids,
        config,
        [],
        Marker.Cross,
        loadTable,
      );

      expect(packedMarkers[0]).toBe(
        markerPalette[HashUtils.hash("groupA") % markerPalette.length],
      );
      expect(packedMarkers[1]).toBe(
        markerPalette[HashUtils.hash("groupB") % markerPalette.length],
      );
    });
  });

  describe("resolveConstantMarker", () => {
    it("returns the packed marker for a constant config", () => {
      const config = {
        constant: { value: Marker.Diamond },
      } satisfies MarkerConfig;
      expect(MarkerResolver.resolveConstantMarker(config)).toBe(Marker.Diamond);
    });

    it("returns undefined for table-backed configs", () => {
      const config = { from: { column: "col1" } } satisfies MarkerConfig;
      expect(MarkerResolver.resolveConstantMarker(config)).toBeUndefined();
    });
  });

  describe("resolveMarkerWithoutTable", () => {
    it("returns the packed constant marker for a constant config", () => {
      const config = {
        constant: { value: Marker.Diamond },
      } satisfies MarkerConfig;
      expect(
        MarkerResolver.resolveMarkerWithoutTable(1, config, Marker.Disc),
      ).toBe(Marker.Diamond);
    });

    it("falls back to the default marker for table-backed configs", () => {
      const fromConfig = { from: { column: "col1" } } satisfies MarkerConfig;
      const groupByConfig = {
        groupBy: { column: "col1", map: "mm1" },
      } satisfies MarkerConfig;
      expect(
        MarkerResolver.resolveMarkerWithoutTable(1, fromConfig, Marker.Disc),
      ).toBe(Marker.Disc);
      expect(
        MarkerResolver.resolveMarkerWithoutTable(
          1,
          groupByConfig,
          Marker.Square,
        ),
      ).toBe(Marker.Square);
    });
  });

  describe("resolveMarkers", () => {
    it("dispatches to constant", async () => {
      const config = {
        constant: { value: Marker.Disc },
      } satisfies MarkerConfig;
      const packedMarkers = await MarkerResolver.resolveMarkers(
        new Uint32Array([1, 2]),
        config,
        [],
        Marker.Cross,
      );
      expect(Array.from(packedMarkers)).toEqual([Marker.Disc, Marker.Disc]);
    });

    it("dispatches to from config when loadTable is given", async () => {
      const data = createMockTableData(new Uint32Array([1]), [Marker.Star]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies MarkerConfig;

      const packedMarkers = await MarkerResolver.resolveMarkers(
        new Uint32Array([1]),
        config,
        [],
        Marker.Cross,
        { loadTable },
      );

      expect(loadTable).toHaveBeenCalledOnce();
      expect(packedMarkers[0]).toBe(Marker.Star);
    });

    it("dispatches to groupBy config when loadTable is given", async () => {
      const data = createMockTableData(new Uint32Array([1]), ["A"]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const markerMap: GroupValueMap<Marker> = {
        id: "mm1",
        name: "Marker Map",
        values: { A: Marker.Diamond },
      };
      const config = {
        groupBy: { column: "col1", map: "mm1" },
      } satisfies MarkerConfig;

      const packedMarkers = await MarkerResolver.resolveMarkers(
        new Uint32Array([1]),
        config,
        [markerMap],
        Marker.Cross,
        { loadTable },
      );

      expect(loadTable).toHaveBeenCalledOnce();
      expect(packedMarkers[0]).toBe(Marker.Diamond);
    });

    it("falls back to the default marker when the config has no active source", async () => {
      const config = {} as MarkerConfig;
      const packedMarkers = await MarkerResolver.resolveMarkers(
        new Uint32Array([1, 2]),
        config,
        [],
        Marker.Ring,
      );
      expect(Array.from(packedMarkers)).toEqual([Marker.Ring, Marker.Ring]);
    });

    it("falls back to the default marker for a from config without loadTable", async () => {
      const config = { from: { column: "col1" } } satisfies MarkerConfig;

      const packedMarkers = await MarkerResolver.resolveMarkers(
        new Uint32Array([1]),
        config,
        [],
        Marker.Ring,
      );

      expect(packedMarkers[0]).toBe(Marker.Ring);
    });

    it("falls back to the default marker for a groupBy config without loadTable", async () => {
      const markerMap: GroupValueMap<Marker> = {
        id: "mm1",
        name: "Marker Map",
        values: { A: Marker.Diamond },
      };
      const config = {
        groupBy: { column: "col1", map: "mm1" },
      } satisfies MarkerConfig;

      const packedMarkers = await MarkerResolver.resolveMarkers(
        new Uint32Array([1]),
        config,
        [markerMap],
        Marker.Ring,
        {},
      );

      expect(packedMarkers[0]).toBe(Marker.Ring);
    });

    it("throws when the signal is already aborted", async () => {
      const controller = new AbortController();
      controller.abort();
      const config = {
        constant: { value: Marker.Disc },
      } satisfies MarkerConfig;

      await expect(
        MarkerResolver.resolveMarkers(
          new Uint32Array([1]),
          config,
          [],
          Marker.Cross,
          {
            signal: controller.signal,
          },
        ),
      ).rejects.toThrow();
    });
  });
});
