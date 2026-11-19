import { describe, expect, it, vi } from "vitest";

import type { AppStoreApi, TableDataSource } from "@tissuumaps/core";

import {
  type ComputedTable,
  columnValues,
  computedTableCsv,
  computedTableData,
  computedTableProvider,
  computedTableType,
  csvFileName,
  forgetVersions,
  idsSurviveCsv,
  keepVersion,
  registerComputedTables,
  suggestCsvName,
  valuesByRow,
  versionOf,
} from "./computed-tables";

const table = (): ComputedTable => ({
  originId: "tables/cells",
  ids: ["a", "b", "c"],
  names: undefined,
  columns: new Map<string, Float64Array | string[]>([
    ["neighbors", Float64Array.from([3, NaN, 5])],
    ["zone", ["edge", 'say "hi", twice', ""]],
  ]),
});

describe("computedTableData", () => {
  it("reads like the other tables: the IDs of the original table, and the columns", async () => {
    const data = computedTableData(table());
    expect(data.getIds()).toEqual(["a", "b", "c"]);
    expect(data.getSize()).toBe(3);
    expect(data.getNames).toBeUndefined();
    expect(
      (await data.suggestColumnQueries("z")).map(({ query }) => query),
    ).toEqual(["zone", "neighbors"]);
    expect(await data.resolveColumnQuery("zone")).toBe("zone");
    expect(await data.resolveColumnQuery("other")).toBeNull();
    expect(await data.loadValues("zone")).toEqual([
      "edge",
      'say "hi", twice',
      "",
    ]);
    await expect(data.loadValues("other")).rejects.toThrow(
      'the table has no column "other"',
    );
  });
});

describe("computedTableProvider", () => {
  it("loads a version kept in this page, and explains one that is gone", async () => {
    const source = keepVersion(table());
    expect(source.startsWith(`${computedTableType}:`)).toBe(true);
    const dataSource = { type: computedTableType, source } as TableDataSource;
    expect(computedTableProvider.normalize(dataSource, null, null)).toBe(
      dataSource,
    );
    expect((await computedTableProvider.load(dataSource)).getSize()).toBe(3);
    forgetVersions(new Set());
    expect(versionOf(source)).toBeUndefined();
    await expect(computedTableProvider.load(dataSource)).rejects.toThrow(
      "this table was computed by the AI chat in an earlier session, and was not kept as a CSV file, so its values are gone",
    );
  });

  it("is registered once", () => {
    const providers = new Map<string, unknown>();
    const registerTableDataProvider = vi.fn((type: string, provider: unknown) =>
      providers.set(type, provider),
    );
    const appStore = {
      getState: () => ({
        tableDataProviders: providers,
        registerTableDataProvider,
      }),
    } as unknown as AppStoreApi;
    registerComputedTables(appStore);
    registerComputedTables(appStore);
    expect(registerTableDataProvider).toHaveBeenCalledTimes(1);
    expect(providers.get(computedTableType)).toBe(computedTableProvider);
  });
});

describe("columnValues", () => {
  it("keeps numbers as numbers, missing ones as NaN, and anything with text as text", () => {
    expect(columnValues(Float32Array.from([1.5, 2]))).toEqual(
      Float64Array.from([1.5, 2]),
    );
    expect(columnValues([1, null, 3])).toEqual(Float64Array.from([1, NaN, 3]));
    expect(columnValues(["a", 2, true, null])).toEqual(["a", "2", "true", ""]);
    expect(columnValues([{ a: 1 }])).toBe(
      "each value must be a number or a text",
    );
    expect(columnValues("abc")).toBe(
      "the values must be a list (an array), one value per row",
    );
  });
});

describe("valuesByRow", () => {
  it("puts values given for some items into the order of the rows", () => {
    expect(
      valuesByRow(["a", "b", "c"], ["c", "a"], Float64Array.from([3, 1])),
    ).toEqual(Float64Array.from([1, NaN, 3]));
    expect(valuesByRow(Float64Array.from([10, 20]), ["20"], ["x"])).toEqual([
      "",
      "x",
    ]);
    expect(valuesByRow(["a"], ["a", "b"], ["x"])).toBe(
      "there are 2 ids but 1 values",
    );
    expect(valuesByRow(["a"], ["z"], ["x"])).toBe(
      '1 ids are not in the table, e.g. ["z"]',
    );
  });
});

describe("CSV", () => {
  it("writes the IDs and the columns, quoting text when needed", () => {
    expect(computedTableCsv(table())).toBe(
      'id,neighbors,zone\na,3,edge\nb,,"say ""hi"", twice"\nc,5,\n',
    );
  });

  it("proposes a file name with the table, its columns and the day", () => {
    const day = new Date(2026, 10, 5);
    expect(suggestCsvName("cell_circles (chat)", ["c3_group"], day)).toBe(
      "cell_circles-c3_group-2026-11-05.csv",
    );
    expect(
      suggestCsvName("table (chat)", ["neighbors 30 µm", "zone"], day),
    ).toBe("table-neighbors_30_µm_zone-2026-11-05.csv");
    expect(suggestCsvName("(chat)", [], day)).toBe("table-2026-11-05.csv");
  });

  it("accepts only valid file names, and adds .csv", () => {
    expect(csvFileName(" vicini_cluster3 ")).toEqual({
      name: "vicini_cluster3.csv",
    });
    expect(csvFileName("Cells (near) 2.CSV")).toEqual({
      name: "Cells (near) 2.CSV",
    });
    for (const bad of ["", "a/b", "..\\x", "x?.csv", ".hidden", "con", "a."]) {
      expect(csvFileName(bad)).toEqual({
        error: `"${bad}" is not a valid file name: use letters, digits, spaces, "-", "_", "." and "( )", without folders, and at most 100 characters`,
      });
    }
  });

  it("knows which IDs come back the same from a CSV file", () => {
    expect(idsSurviveCsv(Uint32Array.from([1, 2]))).toBe(true);
    expect(idsSurviveCsv(["cell-1", "cell-2"])).toBe(true);
    expect(idsSurviveCsv(["1", "2"])).toBe(false);
  });
});
