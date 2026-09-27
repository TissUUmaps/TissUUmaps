import { describe, expect, it } from "vitest";

import { ColumnQueryUtils } from "./ColumnQueryUtils";
import type { HierarchicalTableColumn } from "./HierarchicalTable";

const genes = ["CD3", "CD4", "CD8"];

const columns: HierarchicalTableColumn[] = [
  { kind: "matrix", path: "X", numColumns: 250 },
  { kind: "matrix", path: "layers/counts", numColumns: 3, selectors: genes },
  { kind: "dataset", path: "obs/_index" },
  { kind: "dataset", path: "obs/area" },
  { kind: "dataset", path: "obs/cell_type" },
  { kind: "matrix", path: "obsm/spatial", numColumns: 2 },
  { kind: "dataset", path: "var/_index" },
];

describe("ColumnQueryUtils", () => {
  describe("isQueryableName", () => {
    it("rejects names with brackets", () => {
      expect(ColumnQueryUtils.isQueryableName("obs/area")).toBe(true);
      expect(ColumnQueryUtils.isQueryableName("a[1]")).toBe(false);
      expect(ColumnQueryUtils.isQueryableName("a]")).toBe(false);
    });
  });

  describe("joinPath", () => {
    it("joins a parent path and a name", () => {
      expect(ColumnQueryUtils.joinPath("obs", "area")).toBe("obs/area");
      expect(ColumnQueryUtils.joinPath("", "obs")).toBe("obs");
    });
  });

  describe("getMatrixSelectors", () => {
    it("uses unique names as selectors", () => {
      expect(ColumnQueryUtils.getMatrixSelectors(genes)).toEqual(genes);
    });

    it("selects numeric, duplicate, empty and bracketed names by index", () => {
      expect(
        ColumnQueryUtils.getMatrixSelectors([
          "7157",
          "CD3",
          "MATR3",
          "MATR3",
          "",
        ]),
      ).toEqual(["0", "CD3", "2", "3", "4"]);
      expect(ColumnQueryUtils.getMatrixSelectors(["a[1]", "b"])).toEqual([
        "0",
        "b",
      ]);
    });
  });

  describe("resolveColumn", () => {
    it("resolves dataset columns without an index", () => {
      expect(ColumnQueryUtils.resolveColumn(columns, "obs/area")).toEqual({
        column: columns[3],
        index: undefined,
      });
    });

    it("resolves matrix selectors to column indices", () => {
      expect(ColumnQueryUtils.resolveColumn(columns, "X[7]")).toEqual({
        column: columns[0],
        index: 7,
      });
      expect(
        ColumnQueryUtils.resolveColumn(columns, "layers/counts[CD4]"),
      ).toEqual({
        column: columns[1],
        index: 1,
      });
    });

    it("ignores a leading slash", () => {
      expect(ColumnQueryUtils.resolveColumn(columns, "/obs/area")).toEqual({
        column: columns[3],
        index: undefined,
      });
    });

    it("rejects malformed selectors", () => {
      expect(ColumnQueryUtils.resolveColumn(columns, "X[")).toBeNull();
      expect(ColumnQueryUtils.resolveColumn(columns, "X[1]x")).toBeNull();
    });

    it("rejects indices out of range and unknown names", () => {
      expect(ColumnQueryUtils.resolveColumn(columns, "X[250]")).toBeNull();
      expect(ColumnQueryUtils.resolveColumn(columns, "X[CD3]")).toBeNull();
      expect(
        ColumnQueryUtils.resolveColumn(columns, "layers/counts[CD9]"),
      ).toBeNull();
    });

    it("always reads a numeric selector as an index", () => {
      const matrix: HierarchicalTableColumn = {
        kind: "matrix",
        path: "X",
        numColumns: 3,
        selectors: ColumnQueryUtils.getMatrixSelectors(["2", "CD3", "CD4"]),
      };
      expect(ColumnQueryUtils.resolveColumn([matrix], "X[2]")?.index).toBe(2);
      expect(ColumnQueryUtils.resolveColumn([matrix], "X[0]")?.index).toBe(0);
    });

    it("matches paths and names ignoring case only when unambiguous", () => {
      const matrix: HierarchicalTableColumn = {
        kind: "matrix",
        path: "X",
        numColumns: 3,
        selectors: ["Cd3", "CD3", "CD4"],
      };
      const ambiguousColumns: HierarchicalTableColumn[] = [
        matrix,
        { kind: "dataset", path: "obs/Area" },
        { kind: "dataset", path: "obs/AREA" },
        { kind: "dataset", path: "obs/type" },
      ];
      expect(
        ColumnQueryUtils.resolveColumn(ambiguousColumns, "x[CD3]"),
      ).toEqual({
        column: matrix,
        index: 1,
      });
      expect(
        ColumnQueryUtils.resolveColumn(ambiguousColumns, "X[cd4]")?.index,
      ).toBe(2);
      expect(
        ColumnQueryUtils.resolveColumn(ambiguousColumns, "X[cd3]"),
      ).toBeNull();
      expect(
        ColumnQueryUtils.resolveColumn(ambiguousColumns, "obs/area"),
      ).toBeNull();
      expect(
        ColumnQueryUtils.resolveColumn(ambiguousColumns, "OBS/TYPE")?.column,
      ).toBe(ambiguousColumns[3]);
    });
  });

  describe("suggestColumnQueries", () => {
    /** Suggestions that continue the query instead of addressing a column */
    function partial(...queries: string[]) {
      return queries.map((query) => ({ query, group: true }));
    }

    /** Suggestions that address a column */
    function complete(...queries: string[]) {
      return queries.map((query) => ({ query }));
    }

    it("lists the root children for an empty query", () => {
      // a matrix addresses a column only with a selector, so `X` is partial
      expect(ColumnQueryUtils.suggestColumnQueries(columns, "")).toEqual(
        partial("X", "layers/", "obs/", "obsm/", "var/"),
      );
    });

    it("lists the children of a group", () => {
      expect(ColumnQueryUtils.suggestColumnQueries(columns, "obs/")).toEqual(
        complete("obs/_index", "obs/area", "obs/cell_type"),
      );
    });

    it("lists children matching the partial name first, ignoring case", () => {
      expect(ColumnQueryUtils.suggestColumnQueries(columns, "obs/ce")).toEqual(
        complete("obs/cell_type", "obs/_index", "obs/area"),
      );
      expect(ColumnQueryUtils.suggestColumnQueries(columns, "OBS/CE")).toEqual(
        complete("obs/cell_type", "obs/_index", "obs/area"),
      );
      expect(ColumnQueryUtils.suggestColumnQueries(columns, "obs")).toEqual(
        partial("obs/", "obsm/", "X", "layers/", "var/"),
      );
    });

    it("lists all children in path order when none matches", () => {
      expect(ColumnQueryUtils.suggestColumnQueries(columns, "obs/z")).toEqual(
        complete("obs/_index", "obs/area", "obs/cell_type"),
      );
    });

    it("ignores a leading slash", () => {
      expect(ColumnQueryUtils.suggestColumnQueries(columns, "/obs/ar")).toEqual(
        complete("obs/area", "obs/_index", "obs/cell_type"),
      );
    });

    it("keeps a slash within the selector out of the path", () => {
      const matrix: HierarchicalTableColumn = {
        kind: "matrix",
        path: "X",
        numColumns: 2,
        selectors: ["HLA-A/B", "CD3"],
      };
      expect(
        ColumnQueryUtils.suggestColumnQueries([matrix], "X[HLA-A/"),
      ).toEqual(complete("X[HLA-A/B]", "X[CD3]"));
    });

    it("expands a fully typed matrix to its columns", () => {
      expect(
        ColumnQueryUtils.suggestColumnQueries(columns, "obsm/spatial"),
      ).toEqual(complete("obsm/spatial[0]", "obsm/spatial[1]"));
    });

    it("lists matrix columns matching the typed index first", () => {
      const suggestions = ColumnQueryUtils.suggestColumnQueries(
        columns,
        "X[24",
      );
      expect(suggestions).toHaveLength(250);
      expect(suggestions.slice(0, 12)).toEqual(
        complete(
          "X[24]",
          "X[240]",
          "X[241]",
          "X[242]",
          "X[243]",
          "X[244]",
          "X[245]",
          "X[246]",
          "X[247]",
          "X[248]",
          "X[249]",
          "X[0]",
        ),
      );
    });

    it("suggests the names of a named matrix", () => {
      expect(
        ColumnQueryUtils.suggestColumnQueries(columns, "layers/counts"),
      ).toEqual(
        complete(
          "layers/counts[CD3]",
          "layers/counts[CD4]",
          "layers/counts[CD8]",
        ),
      );
      expect(
        ColumnQueryUtils.suggestColumnQueries(columns, "layers/counts[cd4"),
      ).toEqual(
        complete(
          "layers/counts[CD4]",
          "layers/counts[CD3]",
          "layers/counts[CD8]",
        ),
      );
    });

    it("lists every column of a large matrix, then the other names", () => {
      const suggestions = ColumnQueryUtils.suggestColumnQueries(columns, "X");
      expect(suggestions).toHaveLength(254);
      expect(suggestions.slice(250)).toEqual(
        partial("layers/", "obs/", "obsm/", "var/"),
      );
    });

    it("lists names equal to, then starting with, then containing the partial name, then the others", () => {
      const obs: HierarchicalTableColumn[] = [
        { kind: "dataset", path: "obs/cell_type" },
        { kind: "dataset", path: "obs/total_counts" },
        { kind: "dataset", path: "obs/type" },
      ];
      expect(ColumnQueryUtils.suggestColumnQueries(obs, "obs/t")).toEqual(
        complete("obs/total_counts", "obs/type", "obs/cell_type"),
      );
      expect(ColumnQueryUtils.suggestColumnQueries(obs, "obs/TYPE")).toEqual(
        complete("obs/type", "obs/cell_type", "obs/total_counts"),
      );
    });

    it("ranks matrix selectors like names", () => {
      const matrix: HierarchicalTableColumn = {
        kind: "matrix",
        path: "X",
        numColumns: 5,
        selectors: ["ANPEP", "DPEP1", "EP300", "EPCAM", "HEPACAM2"],
      };
      expect(ColumnQueryUtils.suggestColumnQueries([matrix], "X[ep")).toEqual(
        complete("X[EP300]", "X[EPCAM]", "X[ANPEP]", "X[DPEP1]", "X[HEPACAM2]"),
      );
    });

    it("lists the root children for an unknown group", () => {
      expect(ColumnQueryUtils.suggestColumnQueries(columns, "nope/x")).toEqual(
        partial("X", "layers/", "obs/", "obsm/", "var/"),
      );
      expect(ColumnQueryUtils.suggestColumnQueries(columns, "//x")).toEqual(
        partial("X", "layers/", "obs/", "obsm/", "var/"),
      );
    });

    it("lists the children of the nearest existing group", () => {
      expect(
        ColumnQueryUtils.suggestColumnQueries(columns, "obs/nope/x"),
      ).toEqual(complete("obs/_index", "obs/area", "obs/cell_type"));
      expect(
        ColumnQueryUtils.suggestColumnQueries(columns, "OBS/nope/"),
      ).toEqual(complete("obs/_index", "obs/area", "obs/cell_type"));
    });

    it("returns nothing for a selector on a dataset", () => {
      expect(
        ColumnQueryUtils.suggestColumnQueries(columns, "obs/area["),
      ).toEqual([]);
    });
  });

  describe("resolveColumnQuery", () => {
    it("resolves dataset columns", () => {
      expect(ColumnQueryUtils.resolveColumnQuery(columns, "obs/area")).toBe(
        "obs/area",
      );
      expect(ColumnQueryUtils.resolveColumnQuery(columns, "/obs/area")).toBe(
        "obs/area",
      );
    });

    it("resolves matrix columns within range", () => {
      expect(
        ColumnQueryUtils.resolveColumnQuery(columns, "obsm/spatial[1]"),
      ).toBe("obsm/spatial[1]");
      expect(
        ColumnQueryUtils.resolveColumnQuery(columns, "obsm/spatial[2]"),
      ).toBeNull();
    });

    it("resolves matrix columns by name", () => {
      expect(
        ColumnQueryUtils.resolveColumnQuery(columns, "layers/counts[cd8]"),
      ).toBe("layers/counts[CD8]");
      expect(
        ColumnQueryUtils.resolveColumnQuery(columns, "layers/counts[CD9]"),
      ).toBeNull();
      expect(ColumnQueryUtils.resolveColumnQuery(columns, "X[CD3]")).toBeNull();
    });

    it("canonicalizes an index to the column's selector", () => {
      expect(
        ColumnQueryUtils.resolveColumnQuery(columns, "layers/counts[1]"),
      ).toBe("layers/counts[CD4]");
      expect(ColumnQueryUtils.resolveColumnQuery(columns, "X[7]")).toBe("X[7]");
    });

    it("canonicalizes the case of paths", () => {
      expect(ColumnQueryUtils.resolveColumnQuery(columns, "OBS/Area")).toBe(
        "obs/area",
      );
    });

    it("rejects groups, bare matrices and indexed datasets", () => {
      expect(ColumnQueryUtils.resolveColumnQuery(columns, "obs")).toBeNull();
      expect(ColumnQueryUtils.resolveColumnQuery(columns, "obs/")).toBeNull();
      expect(ColumnQueryUtils.resolveColumnQuery(columns, "X")).toBeNull();
      expect(
        ColumnQueryUtils.resolveColumnQuery(columns, "obs/area[0]"),
      ).toBeNull();
      expect(ColumnQueryUtils.resolveColumnQuery(columns, "X[")).toBeNull();
    });
  });
});
