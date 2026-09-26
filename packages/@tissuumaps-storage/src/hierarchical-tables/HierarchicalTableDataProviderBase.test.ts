import { describe, expect, it } from "vitest";

import type { IDArray, TypedArrayOrArray } from "@tissuumaps/core";

import type { HierarchicalTable } from "./HierarchicalTable";
import { HierarchicalTableDataBase } from "./HierarchicalTableDataBase";
import { HierarchicalTableDataProviderBase } from "./HierarchicalTableDataProviderBase";
import type { HierarchicalTableDataSource } from "./HierarchicalTableDataSource";

/** A table of 5 rows, with 3 IDs and 2 names */
function createTable() {
  const values: Record<string, TypedArrayOrArray<unknown>> = {
    "obs/id": new Int32Array([1, 2, 3]),
    "obs/name": ["a", "b"],
  };
  return {
    columns: [],
    numRows: 5,
    closeCalls: 0,
    readColumn(query: string) {
      const data = values[query];
      return data !== undefined
        ? Promise.resolve(data)
        : Promise.reject(new Error(`No column "${query}"`));
    },
    readRange() {
      return Promise.resolve(undefined);
    },
    close() {
      this.closeCalls++;
    },
  };
}

class TestTableData extends HierarchicalTableDataBase {}

class TestTableDataProvider extends HierarchicalTableDataProviderBase<
  HierarchicalTableDataSource,
  TestTableData
> {
  readonly name = "Test";
  private readonly _table: HierarchicalTable;

  constructor(table: HierarchicalTable) {
    super();
    this._table = table;
  }

  protected override openHierarchicalTable(): Promise<HierarchicalTable> {
    return Promise.resolve(this._table);
  }

  protected override createTableData(
    table: HierarchicalTable,
    numRows: number,
    ids: IDArray | undefined,
    names: string[] | undefined,
  ): TestTableData {
    return new TestTableData(table, numRows, ids, names);
  }
}

function load(
  table: HierarchicalTable,
  columns: { idColumn?: string; nameColumn?: string },
) {
  return new TestTableDataProvider(table).load({
    type: "test",
    source: "table.h5",
    ...columns,
  });
}

describe("HierarchicalTableDataProviderBase", () => {
  describe("load", () => {
    it("takes the row count from the ID column, then the name column, then the table", async () => {
      const ids = await load(createTable(), { idColumn: "obs/id" });
      expect(ids.getSize()).toBe(3);
      expect(ids.getIds()).toEqual(new Int32Array([1, 2, 3]));
      const names = await load(createTable(), { nameColumn: "obs/name" });
      expect(names.getSize()).toBe(2);
      expect(names.getNames()).toEqual(["a", "b"]);
      expect((await load(createTable(), {})).getSize()).toBe(5);
    });

    it("rejects ID and name columns of different lengths and closes the table", async () => {
      const table = createTable();
      await expect(
        load(table, { idColumn: "obs/id", nameColumn: "obs/name" }),
      ).rejects.toThrow("different lengths");
      expect(table.closeCalls).toBe(1);
    });

    it("closes the table when the ID column cannot be read", async () => {
      const table = createTable();
      await expect(load(table, { idColumn: "obs/missing" })).rejects.toThrow(
        'No column "obs/missing"',
      );
      expect(table.closeCalls).toBe(1);
    });
  });
});
