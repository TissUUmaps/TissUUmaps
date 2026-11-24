import { describe, expect, it } from "vitest";

import type { HierarchicalTable } from "./HierarchicalTable";
import { HierarchicalTableDataBase } from "./HierarchicalTableDataBase";

class TestTableData extends HierarchicalTableDataBase {}

/** An AnnData-like table of 3 cells and 3 genes keyed by Ensembl IDs */
function createTable() {
  const table: HierarchicalTable = {
    columns: [
      {
        kind: "matrix",
        path: "X",
        numColumns: 3,
        selectors: ["ENSG1", "ENSG2", "ENSG3"],
      },
      {
        kind: "matrix",
        path: "layers/counts",
        numColumns: 3,
        selectors: ["ENSG1", "ENSG2", "ENSG3"],
      },
      { kind: "dataset", path: "obs/_index" },
      { kind: "dataset", path: "obs/area" },
      {
        kind: "dataset",
        path: "obs/cell_type",
        dataType: "category",
      },
      { kind: "dataset", path: "uns/unknown" },
      {
        kind: "matrix",
        path: "raw/X",
        numColumns: 2,
        selectors: ["ENSG3", "ENSG1"],
      },
      { kind: "matrix", path: "obsm/X_pca", numColumns: 3 },
      { kind: "matrix", path: "obsm/X_other", numColumns: 3 },
      {
        kind: "matrix",
        path: "layers/unnamed",
        numColumns: 3,
        selectors: ["0", "1", "2"],
      },
    ],
    numRows: 3,
    columnGroups: [
      {
        name: "Annotations",
        subgroups: [{ id: "obs", name: "obs", indexPath: "obs/_index" }],
      },
      {
        name: "Features",
        subgroups: [
          { id: "X", name: "X" },
          { id: "layers/counts", name: "counts" },
          { id: "raw/X", name: "raw" },
          { id: "layers/unnamed", name: "unnamed" },
        ],
      },
      {
        name: "Embeddings",
        subgroups: [
          { id: "obsm/X_pca", name: "pca" },
          { id: "obsm/X_other", name: "other" },
        ],
      },
    ],
    readColumn: () => Promise.reject(new Error("Not read")),
    close: () => {},
  };
  return { table };
}

function queries(suggestions: { query: string }[]): string[] {
  return suggestions.map(({ query }) => query);
}

describe("HierarchicalTableDataBase", () => {
  it("lists the query groups as a tree, without their internals", async () => {
    const { table } = createTable();
    await expect(
      new TestTableData(table, 3, undefined, undefined).loadQueryGroups(),
    ).resolves.toEqual([
      { name: "Annotations", id: "obs" },
      {
        name: "Features",
        subgroups: [
          { name: "X", id: "X" },
          { name: "counts", id: "layers/counts" },
          { name: "raw", id: "raw/X" },
          { name: "unnamed", id: "layers/unnamed" },
        ],
      },
      {
        name: "Embeddings",
        subgroups: [
          { name: "pca", id: "obsm/X_pca" },
          { name: "other", id: "obsm/X_other" },
        ],
      },
    ]);
  });

  it("resolves queries as without query groups", async () => {
    const { table } = createTable();
    await expect(
      new TestTableData(table, 3, undefined, undefined).resolveColumnQuery(
        "obs/area",
      ),
    ).resolves.toBe("obs/area");
  });

  it("suggests the columns of a query group of columns, but its index", async () => {
    const { table } = createTable();
    const tableData = new TestTableData(table, 3, undefined, undefined);
    await expect(
      tableData.suggestColumnQueries("type", { groupId: "obs" }),
    ).resolves.toEqual([
      { query: "obs/cell_type", name: "cell_type" },
      { query: "obs/area", name: "area" },
    ]);
  });

  it("lists every column of a query group, matches first", async () => {
    const { table } = createTable();
    const tableData = new TestTableData(table, 3, undefined, undefined);
    // e.g. typing an annotation lists it first, then all the others
    expect(
      queries(await tableData.suggestColumnQueries("area", { groupId: "obs" })),
    ).toEqual(["obs/area", "obs/cell_type"]);
    // and typing a feature lists it first, then all the other features
    expect(
      queries(await tableData.suggestColumnQueries("ENSG2", { groupId: "X" })),
    ).toEqual(["X[ENSG2]", "X[ENSG1]", "X[ENSG3]"]);
  });

  it("names matrix columns by their selectors", async () => {
    const { table } = createTable();
    const tableData = new TestTableData(table, 3, undefined, undefined);
    expect(
      (await tableData.suggestColumnQueries("", { groupId: "X" })).map(
        ({ name }) => name,
      ),
    ).toEqual(["ENSG1", "ENSG2", "ENSG3"]);
  });

  it("describes columns by their leaf and name", async () => {
    const { table } = createTable();
    const tableData = new TestTableData(table, 3, undefined, undefined);
    await expect(tableData.describeColumn("obs/area")).resolves.toEqual({
      groupId: "obs",
      name: "area",
    });
    await expect(tableData.describeColumn("X[ENSG2]")).resolves.toEqual({
      groupId: "X",
      name: "ENSG2",
    });
    await expect(tableData.describeColumn("uns/unknown")).resolves.toBeNull();
  });

  it("returns the declared column types", async () => {
    const { table } = createTable();
    const tableData = new TestTableData(table, 3, undefined, undefined);
    await expect(tableData.loadColumnDataType("obs/cell_type")).resolves.toBe(
      "category",
    );
    await expect(
      tableData.loadColumnDataType("obs/area"),
    ).resolves.toBeUndefined();
  });

  describe("mapColumn", () => {
    function createTableData() {
      return new TestTableData(createTable().table, 3, undefined, undefined);
    }

    it("maps a feature between matrices by its selector, not its name", async () => {
      const tableData = createTableData();
      await expect(
        tableData.mapColumn("X[ENSG3]", { groupId: "raw/X" }),
      ).resolves.toBe("raw/X[ENSG3]");
      // ENSG1 and ENSG3 share the name CD3E, but are different features
      await expect(
        tableData.mapColumn("X[ENSG1]", { groupId: "raw/X" }),
      ).resolves.toBe("raw/X[ENSG1]");
      await expect(
        tableData.mapColumn("X[ENSG2]", { groupId: "raw/X" }),
      ).resolves.toBeNull();
    });

    it("maps a feature into a layer", async () => {
      await expect(
        createTableData().mapColumn("X[ENSG2]", {
          groupId: "layers/counts",
        }),
      ).resolves.toBe("layers/counts[ENSG2]");
    });

    it("maps matrices without selectors only by the same position", async () => {
      const tableData = createTableData();
      await expect(
        tableData.mapColumn("obsm/X_pca[1]", { groupId: "obsm/X_other" }),
      ).resolves.toBe("obsm/X_other[1]");
      await expect(
        tableData.mapColumn("layers/unnamed[1]", { groupId: "X" }),
      ).resolves.toBeNull();
    });

    it("maps only between the query groups of one column group", async () => {
      const tableData = createTableData();
      await expect(
        tableData.mapColumn("X[ENSG1]", { groupId: "obsm/X_pca" }),
      ).resolves.toBeNull();
      await expect(
        tableData.mapColumn("obsm/X_pca[0]", { groupId: "X" }),
      ).resolves.toBeNull();
    });

    it("throws for an unknown query group", async () => {
      const tableData = createTableData();
      await expect(
        tableData.mapColumn("X[ENSG1]", { groupId: "nope" }),
      ).rejects.toThrow('No query group "nope"');
      await expect(
        tableData.suggestColumnQueries("", { groupId: "nope" }),
      ).rejects.toThrow('No query group "nope"');
    });

    it("finds a column in its own query group of columns only", async () => {
      const tableData = createTableData();
      await expect(
        tableData.mapColumn("obs/area", { groupId: "obs" }),
      ).resolves.toBe("obs/area");
      await expect(
        tableData.mapColumn("obs/_index", { groupId: "obs" }),
      ).resolves.toBeNull();
      await expect(
        tableData.mapColumn("X[ENSG1]", { groupId: "obs" }),
      ).resolves.toBeNull();
    });
  });
});
