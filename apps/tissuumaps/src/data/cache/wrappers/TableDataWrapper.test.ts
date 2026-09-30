import { type Mock, describe, expect, it, vi } from "vitest";

import type { TableData } from "@tissuumaps/core";

import { TableDataWrapper } from "./TableDataWrapper";

function createTestTableData(values: unknown[]): {
  data: TableData;
  loadValues: Mock;
  loadUniqueValueCounts: Mock;
} {
  const loadValues = vi.fn().mockResolvedValue(values);
  const loadUniqueValueCounts = vi.fn();
  return {
    data: {
      close: vi.fn(),
      getIds: () => [],
      getSize: () => values.length,
      suggestColumnQueries: vi.fn(),
      resolveColumnQuery: vi.fn(),
      loadValues,
      loadUniqueValueCounts,
      loadValueRange: vi.fn(),
    },
    loadValues,
    loadUniqueValueCounts,
  };
}

describe("TableDataWrapper", () => {
  describe("loadUniqueValueCounts", () => {
    it("counts the values from the shared column load", async () => {
      const { data, loadValues, loadUniqueValueCounts } = createTestTableData([
        "b",
        "a",
        "b",
      ]);
      const wrapper = new TableDataWrapper(data);

      await wrapper.loadValues("col1");
      const counts = await wrapper.loadUniqueValueCounts("col1");

      expect(counts).toEqual(
        new Map([
          ["b", 2],
          ["a", 1],
        ]),
      );
      expect(loadValues).toHaveBeenCalledTimes(1);
      expect(loadUniqueValueCounts).not.toHaveBeenCalled();
    });
  });
});
