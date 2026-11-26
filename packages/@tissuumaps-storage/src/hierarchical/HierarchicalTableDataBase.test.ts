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
        categorical: true,
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
    sections: [
      { id: "obs", name: "Observations", indexPath: "obs/_index" },
      {
        id: "var",
        name: "Features",
        subsections: [
          { id: "X", name: "X" },
          { id: "layers/counts", name: "counts" },
          { id: "raw/X", name: "raw" },
          { id: "layers/unnamed", name: "unnamed" },
        ],
      },
      {
        id: "obsm",
        name: "Embeddings",
        subsections: [
          { id: "obsm/X_pca", name: "X_pca" },
          { id: "obsm/X_other", name: "X_other" },
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
  it("returns the sections without their internals", () => {
    const { table } = createTable();
    expect(
      new TestTableData(table, 3, undefined, undefined).getColumnSections(),
    ).toEqual([
      { id: "obs", name: "Observations" },
      {
        id: "var",
        name: "Features",
        subsections: [
          { id: "X", name: "X" },
          { id: "layers/counts", name: "counts" },
          { id: "raw/X", name: "raw" },
          { id: "layers/unnamed", name: "unnamed" },
        ],
      },
      {
        id: "obsm",
        name: "Embeddings",
        subsections: [
          { id: "obsm/X_pca", name: "X_pca" },
          { id: "obsm/X_other", name: "X_other" },
        ],
      },
    ]);
  });

  it("resolves queries as without sections", async () => {
    const { table } = createTable();
    await expect(
      new TestTableData(table, 3, undefined, undefined).resolveColumnQuery(
        "obs/area",
      ),
    ).resolves.toBe("obs/area");
  });

  it("suggests the columns of a section of columns, but its index", async () => {
    const { table } = createTable();
    const tableData = new TestTableData(table, 3, undefined, undefined);
    await expect(
      tableData.suggestColumnQueries("type", { sectionId: "obs" }),
    ).resolves.toEqual([
      { query: "obs/cell_type", name: "cell_type" },
      { query: "obs/area", name: "area" },
    ]);
  });

  it("lists every column of a section, matches first", async () => {
    const { table } = createTable();
    const tableData = new TestTableData(table, 3, undefined, undefined);
    // e.g. typing an annotation lists it first, then all the others
    expect(
      queries(
        await tableData.suggestColumnQueries("area", { sectionId: "obs" }),
      ),
    ).toEqual(["obs/area", "obs/cell_type"]);
    // and typing a feature lists it first, then all the other features
    expect(
      queries(
        await tableData.suggestColumnQueries("ENSG2", { sectionId: "X" }),
      ),
    ).toEqual(["X[ENSG2]", "X[ENSG1]", "X[ENSG3]"]);
  });

  it("names matrix columns by their selectors", async () => {
    const { table } = createTable();
    const tableData = new TestTableData(table, 3, undefined, undefined);
    expect(
      (await tableData.suggestColumnQueries("", { sectionId: "X" })).map(
        ({ name }) => name,
      ),
    ).toEqual(["ENSG1", "ENSG2", "ENSG3"]);
  });

  it("describes columns by their leaf and name", async () => {
    const { table } = createTable();
    const tableData = new TestTableData(table, 3, undefined, undefined);
    await expect(tableData.describeColumn("obs/area")).resolves.toEqual({
      sectionId: "obs",
      name: "area",
    });
    await expect(tableData.describeColumn("X[ENSG2]")).resolves.toEqual({
      sectionId: "X",
      name: "ENSG2",
    });
    await expect(tableData.describeColumn("uns/unknown")).resolves.toBeNull();
  });

  it("returns the declared type of a column", () => {
    const { table } = createTable();
    const tableData = new TestTableData(table, 3, undefined, undefined);
    expect(tableData.getColumnType("obs/cell_type")).toBe("categorical");
    expect(tableData.getColumnType("obs/area")).toBeUndefined();
  });

  describe("mapColumn", () => {
    function createTableData() {
      return new TestTableData(createTable().table, 3, undefined, undefined);
    }

    it("maps a feature between matrices by its selector, not its name", async () => {
      const tableData = createTableData();
      await expect(
        tableData.mapColumn("X[ENSG3]", { sectionId: "raw/X" }),
      ).resolves.toBe("raw/X[ENSG3]");
      // ENSG1 and ENSG3 share the name CD3E, but are different features
      await expect(
        tableData.mapColumn("X[ENSG1]", { sectionId: "raw/X" }),
      ).resolves.toBe("raw/X[ENSG1]");
      await expect(
        tableData.mapColumn("X[ENSG2]", { sectionId: "raw/X" }),
      ).resolves.toBeNull();
    });

    it("maps a feature into a layer", async () => {
      await expect(
        createTableData().mapColumn("X[ENSG2]", {
          sectionId: "layers/counts",
        }),
      ).resolves.toBe("layers/counts[ENSG2]");
    });

    it("maps no matrix without selectors, nor a column named by its index", async () => {
      const tableData = createTableData();
      await expect(
        tableData.mapColumn("obsm/X_pca[1]", { sectionId: "obsm/X_other" }),
      ).resolves.toBeNull();
      await expect(
        tableData.mapColumn("layers/unnamed[1]", { sectionId: "X" }),
      ).resolves.toBeNull();
    });

    it("maps only between the leaves of one top-level section", async () => {
      const tableData = createTableData();
      await expect(
        tableData.mapColumn("X[ENSG1]", { sectionId: "obsm/X_pca" }),
      ).resolves.toBeNull();
      await expect(
        tableData.mapColumn("obsm/X_pca[0]", { sectionId: "X" }),
      ).resolves.toBeNull();
    });

    it("throws for an unknown section", async () => {
      const tableData = createTableData();
      await expect(
        tableData.mapColumn("X[ENSG1]", { sectionId: "nope" }),
      ).rejects.toThrow("Section 'nope' not found");
      await expect(
        tableData.suggestColumnQueries("", { sectionId: "nope" }),
      ).rejects.toThrow("Section 'nope' not found");
    });

    it("finds a column in its own section of columns only", async () => {
      const tableData = createTableData();
      await expect(
        tableData.mapColumn("obs/area", { sectionId: "obs" }),
      ).resolves.toBe("obs/area");
      await expect(
        tableData.mapColumn("obs/_index", { sectionId: "obs" }),
      ).resolves.toBeNull();
      await expect(
        tableData.mapColumn("X[ENSG1]", { sectionId: "obs" }),
      ).resolves.toBeNull();
    });
  });
});
