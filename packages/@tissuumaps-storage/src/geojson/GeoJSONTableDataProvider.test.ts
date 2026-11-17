import { describe, expect, it, vi } from "vitest";

import { GeoJSONShapesDataProvider } from "./GeoJSONShapesDataProvider";
import { GeoJSONTableDataProvider } from "./GeoJSONTableDataProvider";
import { runGeoJSONWorker } from "./runGeoJSONWorker";

vi.mock("./runGeoJSONWorker", () => ({ runGeoJSONWorker: vi.fn() }));

describe("GeoJSONTableDataProvider", () => {
  it("loads the feature properties read by the worker", async () => {
    vi.mocked(runGeoJSONWorker).mockResolvedValue({
      op: "table",
      ids: ["a", "b"],
      columns: ["cell_type"],
      columnValues: new Map([["cell_type", ["T", "B"]]]),
    });
    const provider = new GeoJSONTableDataProvider();
    const data = await provider.load(
      provider.normalize(
        { type: "geojson", source: "https://data.example/s.geojson" },
        null,
        null,
      ),
    );
    expect(runGeoJSONWorker).toHaveBeenCalledWith(
      {
        op: "table",
        file: undefined,
        url: "https://data.example/s.geojson",
        idProperty: undefined,
      },
      expect.anything(),
    );
    expect(data.getIds()).toEqual(["a", "b"]);
    expect(await data.loadValues("cell_type")).toEqual(["T", "B"]);
  });
});

describe("GeoJSONShapesDataProvider", () => {
  describe("prepareDataSource", () => {
    it("adds the feature properties as the table of the shapes", async () => {
      const addTable = vi.fn().mockResolvedValue("t1");
      const provider = new GeoJSONShapesDataProvider({ addTable });
      await expect(
        provider.prepareDataSource(
          { type: "geojson", source: "s.geojson", idProperty: "id" },
          null,
          null,
        ),
      ).resolves.toEqual({
        type: "geojson",
        source: "s.geojson",
        idProperty: "id",
        table: "t1",
      });
      expect(addTable).toHaveBeenCalledWith(
        { type: "geojson", source: "s.geojson", idProperty: "id" },
        { signal: undefined },
      );
    });

    it("keeps a referenced table", async () => {
      const addTable = vi.fn();
      const provider = new GeoJSONShapesDataProvider({ addTable });
      const dataSource = {
        type: "geojson" as const,
        source: "s.geojson",
        table: "t0",
      };
      await expect(
        provider.prepareDataSource(dataSource, null, null),
      ).resolves.toBe(dataSource);
      expect(addTable).not.toHaveBeenCalled();
    });

    it("adds no table without a way to add one", async () => {
      const dataSource = { type: "geojson" as const, source: "s.geojson" };
      await expect(
        new GeoJSONShapesDataProvider().prepareDataSource(
          dataSource,
          null,
          null,
        ),
      ).resolves.toBe(dataSource);
    });
  });
});
