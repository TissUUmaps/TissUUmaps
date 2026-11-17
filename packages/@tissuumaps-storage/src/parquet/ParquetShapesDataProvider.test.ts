import { describe, expect, it, vi } from "vitest";

import { ParquetShapesDataProvider } from "./ParquetShapesDataProvider";

describe("ParquetShapesDataProvider", () => {
  describe("prepareDataSource", () => {
    it("adds the file as the table of the shapes, keyed alike", async () => {
      const addTable = vi.fn().mockResolvedValue("t1");
      const provider = new ParquetShapesDataProvider({ addTable });
      await expect(
        provider.prepareDataSource(
          {
            type: "parquet",
            source: "cells.parquet",
            idColumn: "cell_id",
            geometryColumn: "geometry",
          },
          null,
          null,
        ),
      ).resolves.toEqual({
        type: "parquet",
        source: "cells.parquet",
        idColumn: "cell_id",
        geometryColumn: "geometry",
        table: "t1",
      });
      expect(addTable).toHaveBeenCalledWith(
        { type: "parquet", source: "cells.parquet", idColumn: "cell_id" },
        { signal: undefined },
      );
    });

    it("keeps a referenced table", async () => {
      const addTable = vi.fn();
      const provider = new ParquetShapesDataProvider({ addTable });
      const dataSource = {
        type: "parquet" as const,
        source: "cells.parquet",
        table: "t0",
      };
      await expect(
        provider.prepareDataSource(dataSource, null, null),
      ).resolves.toBe(dataSource);
      expect(addTable).not.toHaveBeenCalled();
    });
  });
});
