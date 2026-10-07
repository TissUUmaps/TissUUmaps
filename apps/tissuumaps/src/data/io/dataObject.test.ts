import { describe, expect, it } from "vitest";

import { createDataObjectID } from "./dataObject";

describe("createDataObjectID", () => {
  it("replaces the dots of the file name with hyphens", () => {
    expect(createDataObjectID("/a/cells.ome.zarr", [])).toBe("cells-ome-zarr");
    expect(createDataObjectID("/a/x.zarr/labels/cells", [])).toBe("cells");
    expect(
      createDataObjectID("https://data.example/a/my%20cells.v2.csv?x=1", []),
    ).toBe("my cells-v2-csv");
  });

  it("falls back to a random UUID", () => {
    expect(createDataObjectID(undefined, [])).toMatch(/^[0-9a-f-]{36}$/);
    expect(createDataObjectID("data:text/csv,a", [])).toMatch(
      /^[0-9a-f-]{36}$/,
    );
  });

  it("appends the first free numeric suffix to a taken ID", () => {
    expect(
      createDataObjectID("/a/cells.csv", [
        "cells-csv",
        "cells-csv-2",
        "cells-csv-4",
      ]),
    ).toBe("cells-csv-3");
  });
});
