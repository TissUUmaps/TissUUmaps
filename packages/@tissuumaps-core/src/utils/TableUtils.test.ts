import { type Mock, describe, expect, it, vi } from "vitest";

import type { TableData } from "../storage/table";
import type { IDArray } from "../types/arrays";
import { TableUtils } from "./TableUtils";

/**
 * Creates table data backed by mocks, returned alongside the loaders so that
 * assertions do not have to reference the methods through `data`
 */
function createMockTableData(
  ids: IDArray,
  values: unknown[] = [],
  valueRange?: [number, number],
): { data: TableData; loadValues: Mock; loadValueRange: Mock } {
  const loadValues = vi.fn().mockResolvedValue(values);
  const loadValueRange = vi.fn().mockResolvedValue(valueRange);
  return {
    data: {
      getIds: () => ids,
      getSize: () => ids.length,
      getNames: () => undefined,
      close: vi.fn(),
      loadValues,
      loadValueRange,
      loadUniqueValueCounts: vi.fn(),
      suggestColumnQueries: vi.fn(),
      resolveColumnQuery: vi.fn(),
    },
    loadValues,
    loadValueRange,
  };
}

async function collectRows(
  ids: IDArray,
  tableData: TableData,
  options?: { signal?: AbortSignal },
): Promise<(number | undefined)[]> {
  const rows: (number | undefined)[] = [];
  await TableUtils.forEachRow(
    ids,
    tableData,
    (rowIndex, i) => {
      rows[i] = rowIndex;
    },
    options,
  );
  return rows;
}

describe("TableUtils", () => {
  describe("forEachRow", () => {
    it("maps the table's own IDs to their positions without looking them up", async () => {
      const ids = new Uint32Array([10, 20, 30]);
      const { data } = createMockTableData(ids);
      const getRowIndices = vi.spyOn(TableUtils, "getRowIndices");
      expect(await collectRows(ids, data)).toEqual([0, 1, 2]);
      expect(getRowIndices).not.toHaveBeenCalled();
      getRowIndices.mockRestore();
    });

    it("looks up the rows of other IDs, reporting missing ones as undefined", async () => {
      const { data } = createMockTableData(new Uint32Array([10, 20, 30]));
      expect(await collectRows(new Uint32Array([30, 99, 10]), data)).toEqual([
        2,
        undefined,
        0,
      ]);
    });

    it("maps an ID occurring more than once to its last row, with a warning", async () => {
      const { data } = createMockTableData(new Uint32Array([10, 20, 10]));
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      expect(await collectRows(new Uint32Array([10]), data)).toEqual([2]);
      expect(warn).toHaveBeenCalledOnce();
      expect(warn.mock.calls[0]![0]).toContain("1 duplicated");
      warn.mockRestore();
    });

    it("does not warn about unique IDs", async () => {
      const { data } = createMockTableData(new Uint32Array([10, 20, 30]));
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      await collectRows(new Uint32Array([20]), data);
      expect(warn).not.toHaveBeenCalled();
      warn.mockRestore();
    });

    it("looks up string IDs", async () => {
      const { data } = createMockTableData(["a", "b", "c"]);
      expect(await collectRows(["c", "x", "a"], data)).toEqual([
        2,
        undefined,
        0,
      ]);
    });

    it("does not match numeric IDs to string IDs", async () => {
      const { data } = createMockTableData(["1", "2"]);
      expect(await collectRows(new Uint32Array([1, 2]), data)).toEqual([
        undefined,
        undefined,
      ]);
    });

    it("handles empty IDs", async () => {
      const { data } = createMockTableData(new Uint32Array([10, 20]));
      expect(await collectRows(new Uint32Array([]), data)).toEqual([]);
    });

    it("rejects when the signal is already aborted", async () => {
      const { data } = createMockTableData(new Uint32Array([10]));
      const controller = new AbortController();
      controller.abort();
      await expect(
        collectRows(new Uint32Array([10]), data, { signal: controller.signal }),
      ).rejects.toThrow();
    });
  });

  describe("getRowIndices", () => {
    it("builds the row indices once per ID array", async () => {
      const { data } = createMockTableData(new Uint32Array([10, 20]));
      const first = await TableUtils.getRowIndices(data);
      const second = await TableUtils.getRowIndices(data);
      expect(second).toBe(first);
      expect(Array.from(first.entries())).toEqual([
        [10, 0],
        [20, 1],
      ]);
    });

    it("rebuilds for a different ID array with the same content", async () => {
      const first = await TableUtils.getRowIndices(
        createMockTableData(new Uint32Array([10, 20])).data,
      );
      const second = await TableUtils.getRowIndices(
        createMockTableData(new Uint32Array([10, 20])).data,
      );
      expect(second).not.toBe(first);
      expect(second).toEqual(first);
    });

    it("shares a pending build between concurrent callers", async () => {
      const { data } = createMockTableData(
        Uint32Array.from({ length: 5000 }, (_, i) => i),
      );
      const [first, second] = await Promise.all([
        TableUtils.getRowIndices(data),
        TableUtils.getRowIndices(data),
      ]);
      expect(second).toBe(first);
      expect(first.size).toBe(5000);
    });

    it("keeps building for other callers when one aborts", async () => {
      const { data } = createMockTableData(
        Uint32Array.from({ length: 5000 }, (_, i) => i),
      );
      const controller = new AbortController();
      const aborted = TableUtils.getRowIndices(data, {
        signal: controller.signal,
      });
      const kept = TableUtils.getRowIndices(data);
      controller.abort();
      await expect(aborted).rejects.toThrow();
      expect((await kept).size).toBe(5000);
    });

    it("rejects rather than throws for an already aborted signal", async () => {
      const { data } = createMockTableData(new Uint32Array([1, 2, 3]));
      const controller = new AbortController();
      controller.abort();
      let rowIndices: Promise<ReadonlyMap<number | string, number>> | undefined;
      expect(() => {
        rowIndices = TableUtils.getRowIndices(data, {
          signal: controller.signal,
        });
      }).not.toThrow();
      await expect(rowIndices).rejects.toThrow();
    });
  });

  describe("fillFromTableValues", () => {
    it("fills the buffer from table values using parseTableValue and packValue", async () => {
      const ids = new Uint32Array([1, 2, 3]);
      const { data } = createMockTableData(
        new Uint32Array([1, 2, 3]),
        [10, 20, 30],
      );
      const buffer = new Float32Array(3);

      await TableUtils.fillFromTableValues(
        buffer,
        data,
        ids,
        "col1",
        0,
        (value) => (typeof value === "number" ? value : undefined),
        (value) => value * 2,
      );

      expect(Array.from(buffer)).toEqual([20, 40, 60]);
    });

    it("loads the column values, but not the value range, from the given table data", async () => {
      const { data, loadValues, loadValueRange } = createMockTableData(
        new Uint32Array([1]),
        [10],
        [0, 10],
      );
      const buffer = new Float32Array(1);

      await TableUtils.fillFromTableValues(
        buffer,
        data,
        new Uint32Array([1]),
        "col1",
        0,
        (value) => (typeof value === "number" ? value : undefined),
        (value) => value,
      );

      expect(loadValues).toHaveBeenCalledWith("col1", {
        signal: undefined,
      });
      expect(loadValueRange).not.toHaveBeenCalled();
    });

    it("forwards the signal to the table data load", async () => {
      const controller = new AbortController();
      const { data, loadValues } = createMockTableData(
        new Uint32Array([1]),
        [10],
      );
      const buffer = new Float32Array(1);

      await TableUtils.fillFromTableValues(
        buffer,
        data,
        new Uint32Array([1]),
        "col1",
        0,
        (value) => (typeof value === "number" ? value : undefined),
        (value) => value,
        { signal: controller.signal },
      );

      expect(loadValues).toHaveBeenCalledWith("col1", {
        signal: controller.signal,
      });
    });

    it("uses defaultValue when parseTableValue returns undefined", async () => {
      const ids = new Uint32Array([1, 2]);
      const { data } = createMockTableData(new Uint32Array([1, 2]), ["bad", 5]);
      const buffer = new Float32Array(2);
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

      await TableUtils.fillFromTableValues(
        buffer,
        data,
        ids,
        "col1",
        99,
        (value) => (typeof value === "number" ? value : undefined),
        (value) => value,
      );

      expect(buffer[0]).toBe(99); // "bad" failed parsing → default
      expect(buffer[1]).toBe(5);
      expect(warn).toHaveBeenCalledOnce();
      warn.mockRestore();
    });

    it("uses defaultValue when the ID is missing from the table data", async () => {
      const ids = new Uint32Array([1, 2, 3]);
      const { data } = createMockTableData(new Uint32Array([1, 3]), [10, 30]);
      const buffer = new Float32Array(3);
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

      await TableUtils.fillFromTableValues(
        buffer,
        data,
        ids,
        "col1",
        -1,
        (value) => (typeof value === "number" ? value : undefined),
        (value) => value,
      );

      expect(buffer[0]).toBe(10);
      expect(buffer[1]).toBe(-1); // ID 2 missing
      expect(buffer[2]).toBe(30);
      expect(warn).toHaveBeenCalledOnce();
      warn.mockRestore();
    });

    it("matches table rows by ID rather than by position", async () => {
      const { data } = createMockTableData(
        new Uint32Array([3, 1, 2]),
        [30, 10, 20],
      );
      const buffer = new Float32Array(3);

      await TableUtils.fillFromTableValues(
        buffer,
        data,
        new Uint32Array([1, 2, 3]),
        "col1",
        0,
        (value) => (typeof value === "number" ? value : undefined),
        (value) => value,
      );

      expect(Array.from(buffer)).toEqual([10, 20, 30]);
    });

    it("passes the raw cell value to parseTableValue", async () => {
      const { data } = createMockTableData(new Uint32Array([1]), [50]);
      const buffer = new Float32Array(1);
      const parseTableValue = vi
        .fn<(value: unknown) => number | undefined>()
        .mockReturnValue(50);

      await TableUtils.fillFromTableValues(
        buffer,
        data,
        new Uint32Array([1]),
        "col1",
        0,
        parseTableValue,
        (value) => value,
      );

      expect(parseTableValue).toHaveBeenCalledWith(50);
    });

    it("throws when the signal is already aborted", async () => {
      const controller = new AbortController();
      controller.abort();
      const { data, loadValues } = createMockTableData(
        new Uint32Array([1]),
        [10],
      );
      const buffer = new Float32Array(1);

      await expect(
        TableUtils.fillFromTableValues(
          buffer,
          data,
          new Uint32Array([1]),
          "col1",
          0,
          vi.fn(),
          vi.fn(),
          { signal: controller.signal },
        ),
      ).rejects.toThrow();
      expect(loadValues).not.toHaveBeenCalled();
    });
  });

  describe("fillFromTableGroups", () => {
    it("fills the buffer by grouping table values and mapping groups", async () => {
      const ids = new Uint32Array([1, 2, 3]);
      const { data } = createMockTableData(new Uint32Array([1, 2, 3]), [
        "A",
        "B",
        "A",
      ]);
      const buffer = new Uint8Array(3);
      const mapGroupToValue = vi
        .fn<(group: string) => number>()
        .mockImplementation((group) => (group === "A" ? 10 : 20));

      await TableUtils.fillFromTableGroups(
        buffer,
        data,
        ids,
        "col1",
        0,
        mapGroupToValue,
        (value) => value,
      );

      expect(Array.from(buffer)).toEqual([10, 20, 10]);
    });

    it("loads the column values from the given table data", async () => {
      const { data, loadValues, loadValueRange } = createMockTableData(
        new Uint32Array([1]),
        ["A"],
      );
      const buffer = new Uint8Array(1);

      await TableUtils.fillFromTableGroups(
        buffer,
        data,
        new Uint32Array([1]),
        "col1",
        0,
        () => 10,
        (value) => value,
      );

      expect(loadValues).toHaveBeenCalledWith("col1", {
        signal: undefined,
      });
      // Groups never need a value range
      expect(loadValueRange).not.toHaveBeenCalled();
    });

    it("forwards the signal to the table data load", async () => {
      const controller = new AbortController();
      const { data, loadValues } = createMockTableData(new Uint32Array([1]), [
        "A",
      ]);
      const buffer = new Uint8Array(1);

      await TableUtils.fillFromTableGroups(
        buffer,
        data,
        new Uint32Array([1]),
        "col1",
        0,
        () => 10,
        (value) => value,
        { signal: controller.signal },
      );

      expect(loadValues).toHaveBeenCalledWith("col1", {
        signal: controller.signal,
      });
    });

    it("uses defaultValue when the ID is missing from the table data", async () => {
      const ids = new Uint32Array([1, 2]);
      const { data } = createMockTableData(new Uint32Array([1]), ["A"]);
      const buffer = new Uint8Array(2);
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

      await TableUtils.fillFromTableGroups(
        buffer,
        data,
        ids,
        "col1",
        99,
        () => 10,
        (value) => value,
      );

      expect(buffer[0]).toBe(10);
      expect(buffer[1]).toBe(99); // ID 2 missing
      expect(warn).toHaveBeenCalledOnce();
      warn.mockRestore();
    });

    it("warns once, rather than per item, about missing IDs", async () => {
      const ids = new Uint32Array([1, 2, 3, 4]);
      const { data } = createMockTableData(new Uint32Array([1]), ["A"]);
      const buffer = new Uint8Array(4);
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

      await TableUtils.fillFromTableGroups(
        buffer,
        data,
        ids,
        "col1",
        99,
        () => 10,
        (value) => value,
      );

      expect(warn).toHaveBeenCalledOnce();
      expect(warn.mock.calls[0]![0]).toContain("3 IDs");
      warn.mockRestore();
    });

    it("matches table rows by ID rather than by position", async () => {
      const { data } = createMockTableData(new Uint32Array([3, 1, 2]), [
        "C",
        "A",
        "B",
      ]);
      const buffer = new Uint8Array(3);

      await TableUtils.fillFromTableGroups(
        buffer,
        data,
        new Uint32Array([1, 2, 3]),
        "col1",
        0,
        (group) => group.length,
        (value) => value,
      );

      expect(Array.from(buffer)).toEqual([1, 1, 1]);
    });

    it("converts group values to strings before mapping", async () => {
      const { data } = createMockTableData(new Uint32Array([1]), [42]);
      const buffer = new Uint8Array(1);
      const mapGroupToValue = vi
        .fn<(group: string) => number>()
        .mockReturnValue(1);

      await TableUtils.fillFromTableGroups(
        buffer,
        data,
        new Uint32Array([1]),
        "col1",
        0,
        mapGroupToValue,
        (value) => value,
      );

      expect(mapGroupToValue).toHaveBeenCalledWith("42");
    });

    it("maps each distinct raw group only once", async () => {
      const ids = new Uint32Array([1, 2, 3]);
      const { data } = createMockTableData(new Uint32Array([1, 2, 3]), [
        "A",
        "A",
        "B",
      ]);
      const buffer = new Uint8Array(3);
      const mapGroupToValue = vi
        .fn<(group: string) => number>()
        .mockImplementation((group) => (group === "A" ? 10 : 20));

      await TableUtils.fillFromTableGroups(
        buffer,
        data,
        ids,
        "col1",
        0,
        mapGroupToValue,
        (value) => value,
      );

      expect(Array.from(buffer)).toEqual([10, 10, 20]);
      // Groups "A" and "B" mapped once each, despite three IDs
      expect(mapGroupToValue).toHaveBeenCalledTimes(2);
    });

    it("throws when the signal is already aborted", async () => {
      const controller = new AbortController();
      controller.abort();
      const { data, loadValues } = createMockTableData(new Uint32Array([1]), [
        "A",
      ]);
      const buffer = new Uint8Array(1);

      await expect(
        TableUtils.fillFromTableGroups(
          buffer,
          data,
          new Uint32Array([1]),
          "col1",
          0,
          vi.fn(),
          vi.fn(),
          { signal: controller.signal },
        ),
      ).rejects.toThrow();
      expect(loadValues).not.toHaveBeenCalled();
    });
  });
});
