import { describe, expect, it, vi } from "vitest";

import type { IDArray, OrderConfig, TableData } from "@tissuumaps/core";

import { OrderResolver } from "./OrderResolver";

function createMockTableData(ids: IDArray, values: unknown[]): TableData {
  return {
    getIds: () => ids,
    getSize: () => ids.length,
    getNames: () => undefined,
    close: vi.fn(),
    loadValues: vi.fn().mockResolvedValue(values),
    loadUniqueValueCounts: vi.fn(),
    suggestColumnQueries: vi.fn(),
    resolveColumnQuery: vi.fn(),
  };
}

describe("OrderResolver", () => {
  describe("resolveOrder", () => {
    it("orders the items by ascending value of the column", async () => {
      const ids = new Uint32Array([1, 2, 3, 4]);
      const data = createMockTableData(ids, [3, "x", 1, 2]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies OrderConfig;

      const order = await OrderResolver.resolveOrder(ids, config, loadTable);

      // the unparsable value comes first
      expect(order).toEqual(new Uint32Array([1, 2, 3, 0]));
    });

    it("orders the items exactly despite outliers", async () => {
      const ids = new Uint32Array([1, 2, 3, 4]);
      const data = createMockTableData(ids, [1e6, 2, 1, 1.5]);
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies OrderConfig;

      const order = await OrderResolver.resolveOrder(ids, config, loadTable);

      expect(order).toEqual(new Uint32Array([2, 3, 1, 0]));
    });

    it("looks the values up by ID", async () => {
      const data = createMockTableData(
        new Uint32Array([1, 2, 3]),
        [30, 10, 20],
      );
      const loadTable = vi.fn().mockResolvedValue(data);
      const config = { from: { column: "col1" } } satisfies OrderConfig;

      const order = await OrderResolver.resolveOrder(
        new Uint32Array([3, 1, 9]),
        config,
        loadTable,
      );

      // ID 9 is missing and comes first
      expect(order).toEqual(new Uint32Array([2, 0, 1]));
    });

    it("rejects with the reason of an aborted signal", async () => {
      const ids = new Uint32Array([1, 2]);
      const data = createMockTableData(ids, [2, 1]);
      const controller = new AbortController();
      controller.abort(new Error("aborted"));
      const config = { from: { column: "col1" } } satisfies OrderConfig;

      await expect(
        OrderResolver.resolveOrder(ids, config, () => Promise.resolve(data), {
          signal: controller.signal,
        }),
      ).rejects.toThrow("aborted");
    });

    it("returns null without a configuration", async () => {
      const loadTable = vi.fn();

      const order = await OrderResolver.resolveOrder(
        new Uint32Array([1, 2]),
        undefined,
        loadTable,
      );

      expect(order).toBeNull();
      expect(loadTable).not.toHaveBeenCalled();
    });

    it("returns null if the table cannot be loaded", async () => {
      const config = {
        from: { table: "gone", column: "col1" },
      } satisfies OrderConfig;

      const order = await OrderResolver.resolveOrder(
        new Uint32Array([1, 2]),
        config,
        () => Promise.reject(new Error("Table with ID 'gone' not found")),
      );

      expect(order).toBeNull();
    });

    it("returns null if the column cannot be loaded", async () => {
      const ids = new Uint32Array([1, 2]);
      const data: TableData = {
        ...createMockTableData(ids, []),
        loadValues: vi
          .fn()
          .mockRejectedValue(
            new Error("Column col1 does not exist in the table"),
          ),
      };
      const config = { from: { column: "col1" } } satisfies OrderConfig;

      const order = await OrderResolver.resolveOrder(ids, config, () =>
        Promise.resolve(data),
      );

      expect(order).toBeNull();
    });
  });
});
