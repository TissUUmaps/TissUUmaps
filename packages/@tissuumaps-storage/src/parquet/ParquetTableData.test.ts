import { describe, expect, it } from "vitest";

import { ParquetTableData } from "./ParquetTableData";

describe("ParquetTableData", () => {
  function createTableData(columns: string[]) {
    return new ParquetTableData({}, 0, columns, [], undefined, undefined);
  }

  function suggestions(...queries: string[]) {
    return queries.map((query) => ({ query }));
  }

  describe("suggestColumnQueries", () => {
    const tableData = createTableData([
      "area",
      "Area_um2",
      "cell_type",
      "x",
      "y",
    ]);

    it("lists all columns for an empty query", async () => {
      await expect(tableData.suggestColumnQueries("")).resolves.toEqual(
        suggestions("area", "Area_um2", "cell_type", "x", "y"),
      );
    });

    it("lists case-insensitive matches first", async () => {
      await expect(tableData.suggestColumnQueries("TYPE")).resolves.toEqual(
        suggestions("cell_type", "area", "Area_um2", "x", "y"),
      );
    });

    it("lists prefix matches before other matches", async () => {
      await expect(
        createTableData([
          "cell_type",
          "total_counts",
          "type",
        ]).suggestColumnQueries("T"),
      ).resolves.toEqual(suggestions("total_counts", "type", "cell_type"));
    });

    it("lists an exact match first", async () => {
      await expect(
        createTableData(["Area_um2", "area"]).suggestColumnQueries("area"),
      ).resolves.toEqual(suggestions("area", "Area_um2"));
    });

    it("lists all columns in table order when nothing matches", async () => {
      await expect(tableData.suggestColumnQueries("z")).resolves.toEqual(
        suggestions("area", "Area_um2", "cell_type", "x", "y"),
      );
    });
  });

  describe("resolveColumnQuery", () => {
    it("resolves an existing column name", async () => {
      await expect(
        createTableData(["cell_type"]).resolveColumnQuery("cell_type"),
      ).resolves.toBe("cell_type");
    });

    it("resolves a differently cased name when it is unambiguous", async () => {
      await expect(
        createTableData(["cell_type"]).resolveColumnQuery("CELL_TYPE"),
      ).resolves.toBe("cell_type");
    });

    it("prefers the exactly named column over differently cased ones", async () => {
      await expect(
        createTableData(["Area", "area"]).resolveColumnQuery("area"),
      ).resolves.toBe("area");
    });

    it("does not resolve an ambiguous or partial name", async () => {
      await expect(
        createTableData(["Area", "AREA"]).resolveColumnQuery("area"),
      ).resolves.toBeNull();
      await expect(
        createTableData(["area"]).resolveColumnQuery("are"),
      ).resolves.toBeNull();
    });
  });
});
