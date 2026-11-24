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
      const { data, loadValues } = createTestTableData(["b", "a", "b"]);
      delete data.loadUniqueValueCounts;
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
    });

    it("uses the wrapped data's counts if it counts the values itself", async () => {
      const { data, loadValues, loadUniqueValueCounts } = createTestTableData(
        [],
      );
      const expected = new Map([["a", 3]]);
      loadUniqueValueCounts.mockResolvedValue(expected);
      const wrapper = new TableDataWrapper(data);

      await expect(wrapper.loadUniqueValueCounts("col1")).resolves.toBe(
        expected,
      );
      expect(loadUniqueValueCounts).toHaveBeenCalledTimes(1);
      expect(loadValues).not.toHaveBeenCalled();
    });
  });

  describe("loadValueRange", () => {
    it("computes the range from the shared column load", async () => {
      const { data, loadValues } = createTestTableData([]);
      loadValues.mockResolvedValue(new Float32Array([2, -1, 3]));
      delete data.loadValueRange;
      const wrapper = new TableDataWrapper(data);

      await wrapper.loadValues("col1");
      const range = await wrapper.loadValueRange("col1");

      expect(range).toEqual([-1, 3]);
      expect(loadValues).toHaveBeenCalledTimes(1);
    });

    it("uses the wrapped data's range if it determines the range itself", async () => {
      const { data, loadValues } = createTestTableData([]);
      const loadValueRange = vi.fn().mockResolvedValue([-5, 5]);
      data.loadValueRange = loadValueRange;
      const wrapper = new TableDataWrapper(data);

      await expect(wrapper.loadValueRange("col1")).resolves.toEqual([-5, 5]);
      expect(loadValueRange).toHaveBeenCalledTimes(1);
      expect(loadValues).not.toHaveBeenCalled();
    });
  });

  describe("optional facts", () => {
    it("forwards the optional methods only if the wrapped data has them", async () => {
      const { data } = createTestTableData([]);
      const wrapperWithout = new TableDataWrapper(data);
      expect(wrapperWithout.loadQueryGroups).toBeUndefined();
      expect(wrapperWithout.describeColumn).toBeUndefined();
      expect(wrapperWithout.mapColumn).toBeUndefined();

      const groups = [
        {
          name: "Features",
          subgroups: [
            { name: "X", id: "X" },
            { name: "raw", id: "raw/X" },
          ],
        },
      ];
      const mapColumn = vi.fn().mockResolvedValue("raw/X[CD3]");
      data.loadQueryGroups = vi.fn().mockResolvedValue(groups);
      data.describeColumn = vi
        .fn()
        .mockResolvedValue({ groupId: "X", name: "CD3" });
      data.mapColumn = mapColumn;
      const wrapper = new TableDataWrapper(data);
      await expect(wrapper.loadQueryGroups!()).resolves.toBe(groups);
      await expect(wrapper.describeColumn!("X[CD3]")).resolves.toEqual({
        groupId: "X",
        name: "CD3",
      });
      await expect(
        wrapper.mapColumn!("X[CD3]", { groupId: "raw/X" }),
      ).resolves.toBe("raw/X[CD3]");
      expect(mapColumn).toHaveBeenCalledWith("X[CD3]", {
        groupId: "raw/X",
      });
    });

    it("passes the group ID through to suggest", async () => {
      const { data } = createTestTableData([]);
      const suggestColumnQueries = vi.fn().mockResolvedValue([]);
      data.suggestColumnQueries = suggestColumnQueries;
      const wrapper = new TableDataWrapper(data);
      await wrapper.suggestColumnQueries("cd", { groupId: "X" });
      expect(suggestColumnQueries).toHaveBeenCalledWith("cd", {
        groupId: "X",
      });
    });
  });

  describe("loadColumnDataType", () => {
    it("uses the type the wrapped data declares", async () => {
      const { data, loadValues } = createTestTableData([1, 2]);
      data.loadColumnDataType = vi.fn().mockResolvedValue("category");
      const wrapper = new TableDataWrapper(data);
      await expect(wrapper.loadColumnDataType("col1")).resolves.toBe(
        "category",
      );
      expect(loadValues).not.toHaveBeenCalled();
    });

    it("derives the type from the shared column load otherwise", async () => {
      const { data, loadValues } = createTestTableData(["a", "b"]);
      const wrapper = new TableDataWrapper(data);
      await wrapper.loadValues("col1");
      await expect(wrapper.loadColumnDataType("col1")).resolves.toBe(
        "category",
      );
      expect(loadValues).toHaveBeenCalledTimes(1);
    });
  });
});
