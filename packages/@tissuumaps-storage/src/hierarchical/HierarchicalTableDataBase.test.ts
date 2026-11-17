import { describe, expect, it, vi } from "vitest";

import type { HierarchicalTable } from "./HierarchicalTable";
import { HierarchicalTableDataBase } from "./HierarchicalTableDataBase";

class TestTableData extends HierarchicalTableDataBase {}

/** An AnnData-like table of 3 cells and 3 genes keyed by Ensembl IDs */
function createTable(
  varColumns: Record<string, string[]> = {
    feature_name: ["CD3E", "CD4", "CD3E"],
  },
  namePath: string | null = "var/feature_name",
) {
  const readColumn = vi.fn((query: string) => {
    const name = query.slice("var/".length);
    const values = varColumns[name];
    return values !== undefined
      ? Promise.resolve(values)
      : Promise.reject(new Error(`No column "${query}"`));
  });
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
      ...Object.keys(varColumns).map((name) => ({
        kind: "dataset" as const,
        path: `var/${name}`,
      })),
    ],
    numRows: 3,
    queryGroups: [
      { id: "obs", name: "Annotations", indexPath: "obs/_index" },
      {
        id: "var",
        name: "Features",
        subgroups: [
          {
            id: "X",
            name: "X",
            ...(namePath !== null && { namePath }),
          },
          {
            id: "layers/counts",
            name: "counts",
            ...(namePath !== null && { namePath }),
          },
          { id: "raw/X", name: "raw" },
          { id: "layers/unnamed", name: "unnamed" },
        ],
      },
      {
        id: "obsm",
        name: "Embeddings",
        subgroups: [
          { id: "obsm/X_pca", name: "X_pca" },
          { id: "obsm/X_other", name: "X_other" },
        ],
      },
    ],
    readColumn,
    close: () => {},
  };
  return { table, readColumn };
}

function queries(suggestions: { query: string }[]): string[] {
  return suggestions.map(({ query }) => query);
}

describe("HierarchicalTableDataBase", () => {
  it("returns the query groups without their internals", () => {
    const { table } = createTable();
    expect(
      new TestTableData(table, 3, undefined, undefined).getQueryGroups(),
    ).toEqual([
      { id: "obs", name: "Annotations" },
      {
        id: "var",
        name: "Features",
        subgroups: [
          { id: "X", name: "X" },
          { id: "layers/counts", name: "counts" },
          { id: "raw/X", name: "raw" },
          { id: "layers/unnamed", name: "unnamed" },
        ],
      },
      {
        id: "obsm",
        name: "Embeddings",
        subgroups: [
          { id: "obsm/X_pca", name: "X_pca" },
          { id: "obsm/X_other", name: "X_other" },
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
      tableData.suggestColumnQueries("type", { queryGroupId: "obs" }),
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
      queries(
        await tableData.suggestColumnQueries("area", { queryGroupId: "obs" }),
      ),
    ).toEqual(["obs/area", "obs/cell_type"]);
    // and typing a feature lists it first, then all the other features
    expect(
      queries(
        await tableData.suggestColumnQueries("CD4", { queryGroupId: "X" }),
      ),
    ).toEqual(["X[ENSG2]", "X[ENSG1]", "X[ENSG3]"]);
  });

  it("names matrix columns by the conventional name column", async () => {
    const { table, readColumn } = createTable();
    const tableData = new TestTableData(table, 3, undefined, undefined);
    await expect(
      tableData.suggestColumnQueries("cd4", { queryGroupId: "X" }),
    ).resolves.toEqual([
      { query: "X[ENSG2]", name: "CD4" },
      { query: "X[ENSG1]", name: "CD3E (ENSG1)" },
      { query: "X[ENSG3]", name: "CD3E (ENSG3)" },
    ]);
    // searches by ID too, and reads the names once
    expect(
      queries(
        await tableData.suggestColumnQueries("ensg3", { queryGroupId: "X" }),
      )[0],
    ).toBe("X[ENSG3]");
    expect(readColumn).toHaveBeenCalledTimes(1);
  });

  it("names matrix columns by the chosen name column", async () => {
    const { table } = createTable(
      { feature_name: ["a", "b", "c"], symbol: ["S1", "S2", "S3"] },
      "var/symbol",
    );
    const tableData = new TestTableData(table, 3, undefined, undefined);
    await expect(tableData.describeColumn("X[ENSG2]")).resolves.toEqual({
      queryGroupId: "X",
      name: "S2",
    });
  });

  it("names matrix columns by their selectors without a name column", async () => {
    const { table } = createTable({}, null);
    const tableData = new TestTableData(table, 3, undefined, undefined);
    expect(
      (await tableData.suggestColumnQueries("", { queryGroupId: "X" })).map(
        ({ name }) => name,
      ),
    ).toEqual(["ENSG1", "ENSG2", "ENSG3"]);
  });

  it("describes columns by their leaf and name", async () => {
    const { table } = createTable();
    const tableData = new TestTableData(table, 3, undefined, undefined);
    await expect(tableData.describeColumn("obs/area")).resolves.toEqual({
      queryGroupId: "obs",
      name: "area",
    });
    await expect(tableData.describeColumn("uns/unknown")).resolves.toBeNull();
  });

  it("returns whether a column is declared categorical", () => {
    const { table } = createTable();
    const tableData = new TestTableData(table, 3, undefined, undefined);
    expect(tableData.isColumnCategorical("obs/cell_type")).toBe(true);
    expect(tableData.isColumnCategorical("obs/area")).toBeUndefined();
  });

  describe("mapColumn", () => {
    function createTableData() {
      return new TestTableData(createTable().table, 3, undefined, undefined);
    }

    it("maps a feature between matrices by its selector, not its name", async () => {
      const tableData = createTableData();
      await expect(
        tableData.mapColumn("X[ENSG3]", { queryGroupId: "raw/X" }),
      ).resolves.toBe("raw/X[ENSG3]");
      // ENSG1 and ENSG3 share the name CD3E, but are different features
      await expect(
        tableData.mapColumn("X[ENSG1]", { queryGroupId: "raw/X" }),
      ).resolves.toBe("raw/X[ENSG1]");
      await expect(
        tableData.mapColumn("X[ENSG2]", { queryGroupId: "raw/X" }),
      ).resolves.toBeNull();
    });

    it("maps a feature into a layer", async () => {
      await expect(
        createTableData().mapColumn("X[ENSG2]", {
          queryGroupId: "layers/counts",
        }),
      ).resolves.toBe("layers/counts[ENSG2]");
    });

    it("maps no matrix without selectors, nor a column named by its index", async () => {
      const tableData = createTableData();
      await expect(
        tableData.mapColumn("obsm/X_pca[1]", { queryGroupId: "obsm/X_other" }),
      ).resolves.toBeNull();
      await expect(
        tableData.mapColumn("layers/unnamed[1]", { queryGroupId: "X" }),
      ).resolves.toBeNull();
    });

    it("maps only between the leaves of one top-level query group", async () => {
      const tableData = createTableData();
      await expect(
        tableData.mapColumn("X[ENSG1]", { queryGroupId: "obsm/X_pca" }),
      ).resolves.toBeNull();
      await expect(
        tableData.mapColumn("obsm/X_pca[0]", { queryGroupId: "X" }),
      ).resolves.toBeNull();
    });

    it("throws for an unknown query group", async () => {
      const tableData = createTableData();
      await expect(
        tableData.mapColumn("X[ENSG1]", { queryGroupId: "nope" }),
      ).rejects.toThrow("Query group 'nope' not found");
      await expect(
        tableData.suggestColumnQueries("", { queryGroupId: "nope" }),
      ).rejects.toThrow("Query group 'nope' not found");
    });

    it("finds a column in its own query group of columns only", async () => {
      const tableData = createTableData();
      await expect(
        tableData.mapColumn("obs/area", { queryGroupId: "obs" }),
      ).resolves.toBe("obs/area");
      await expect(
        tableData.mapColumn("obs/_index", { queryGroupId: "obs" }),
      ).resolves.toBeNull();
      await expect(
        tableData.mapColumn("X[ENSG1]", { queryGroupId: "obs" }),
      ).resolves.toBeNull();
    });
  });
});
