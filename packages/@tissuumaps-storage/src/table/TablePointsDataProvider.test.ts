import { describe, expect, it, vi } from "vitest";

import type {
  TableCoordinateColumns,
  TableData,
  TableDataProvider,
  TableDataSource,
} from "@tissuumaps/core";

import { TablePointsDataProvider } from "./TablePointsDataProvider";

/** Table data with the given columns, resolved ignoring case */
function createTableData(
  columns: string[],
  coordinateColumns?: TableCoordinateColumns[],
): TableData {
  return {
    getIds: () => new Uint32Array(0),
    getSize: () => 0,
    close: vi.fn(),
    loadValues: vi.fn(),
    suggestColumnQueries: vi.fn(),
    resolveColumnQuery: (query: string) =>
      Promise.resolve(
        columns.find(
          (column) => column.toLowerCase() === query.toLowerCase(),
        ) ?? null,
      ),
    ...(coordinateColumns !== undefined && {
      getCoordinateColumns: () => coordinateColumns,
    }),
  };
}

function createProvider(tableData: TableData) {
  const csvProvider = {
    supports: () => Promise.resolve(true),
  } as unknown as TableDataProvider<TableDataSource, TableData>;
  const addTable = vi.fn().mockResolvedValue("table1");
  const loadTableData = vi.fn().mockResolvedValue(tableData);
  const provider = new TablePointsDataProvider({
    getTableDataProviders: () => new Map([["csv", csvProvider]]),
    addTable,
    loadTableData,
  });
  return { provider, addTable, loadTableData };
}

async function prepare(
  provider: TablePointsDataProvider,
  coordinates: { x?: string; y?: string } = {},
) {
  return await provider.prepareDataSource(
    { type: "table", source: "https://data.example/t.csv", ...coordinates },
    null,
    null,
  );
}

describe("TablePointsDataProvider", () => {
  describe("prepareDataSource", () => {
    it("takes the first coordinate columns the table records", async () => {
      const { provider, loadTableData } = createProvider(
        createTableData(
          ["x", "y"],
          [
            { name: "spatial", x: "obsm/spatial[0]", y: "obsm/spatial[1]" },
            { name: "X_umap", x: "obsm/X_umap[0]", y: "obsm/X_umap[1]" },
          ],
        ),
      );
      await expect(prepare(provider)).resolves.toEqual({
        type: "table",
        table: "table1",
        x: "obsm/spatial[0]",
        y: "obsm/spatial[1]",
      });
      expect(loadTableData).toHaveBeenCalledWith("table1", {
        signal: undefined,
      });
    });

    it("falls back to common coordinate column names", async () => {
      const { provider } = createProvider(
        createTableData(["feature_name", "X_Location", "y_location"]),
      );
      await expect(prepare(provider)).resolves.toMatchObject({
        x: "X_Location",
        y: "y_location",
      });
    });

    it("requires both columns of a pair", async () => {
      const { provider } = createProvider(
        createTableData(["x_centroid", "y", "global_x", "global_y"]),
      );
      await expect(prepare(provider)).resolves.toMatchObject({
        x: "global_x",
        y: "global_y",
      });
    });

    it("falls back to the defaults", async () => {
      const { provider } = createProvider(createTableData(["a", "b"]));
      await expect(prepare(provider)).resolves.toMatchObject({
        x: "x",
        y: "y",
      });
    });

    it("keeps the coordinate columns that are set", async () => {
      const { provider } = createProvider(
        createTableData(["x_location", "y_location"]),
      );
      await expect(prepare(provider, { x: "a" })).resolves.toMatchObject({
        x: "a",
        y: "y_location",
      });
    });

    it("does not load the table if both coordinates are set", async () => {
      const { provider, loadTableData } = createProvider(createTableData([]));
      await expect(
        prepare(provider, { x: "a", y: "b" }),
      ).resolves.toMatchObject({ x: "a", y: "b", table: "table1" });
      expect(loadTableData).not.toHaveBeenCalled();
    });

    it("leaves a data source without a source as it is", async () => {
      const { provider, addTable } = createProvider(createTableData([]));
      const dataSource = { type: "table" as const, table: "t" };
      await expect(
        provider.prepareDataSource(dataSource, null, null),
      ).resolves.toBe(dataSource);
      expect(addTable).not.toHaveBeenCalled();
    });
  });
});
