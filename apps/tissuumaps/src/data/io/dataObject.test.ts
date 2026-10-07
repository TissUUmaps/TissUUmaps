import { beforeEach, describe, expect, it } from "vitest";

import type { ImageDataSource, PointsDataSource } from "@tissuumaps/core";

import { appStore } from "@/stores/app";
import { projectStore } from "@/stores/project";

import {
  addImageDataObject,
  addPointsDataObject,
  createDataObjectID,
} from "./dataObject";

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

describe("addImageDataObject", () => {
  const source = "https://data.example/cells.ome.tif";
  const dataSource: ImageDataSource = { type: "tiff", source };

  beforeEach(() => {
    projectStore.getState().clear();
    appStore.getState().setExpandedImageIds([]);
  });

  it("adds the image with an ID from its source, and expands it", () => {
    expect(addImageDataObject("Cells", "layer", source, dataSource)).toBe(
      "cells-ome-tif",
    );
    expect(projectStore.getState().images).toMatchObject([
      { id: "cells-ome-tif", name: "Cells", layer: "layer", dataSource },
    ]);
    expect(appStore.getState().expandedImageIds).toEqual(["cells-ome-tif"]);
  });

  it("expands a reused ID only once", () => {
    appStore.getState().setExpandedImageIds(["cells-ome-tif"]);
    addImageDataObject("Cells", "layer", source, dataSource);
    expect(appStore.getState().expandedImageIds).toEqual(["cells-ome-tif"]);
  });

  it("rejects an image without a layer", () => {
    expect(() =>
      addImageDataObject("Cells", undefined, source, dataSource),
    ).toThrow(/layer/);
  });
});

describe("addPointsDataObject", () => {
  beforeEach(() => {
    projectStore.getState().clear();
    appStore.getState().setExpandedPointsIds([]);
  });

  it("creates the ID from the source the data source had before it was prepared", () => {
    const preparedDataSource = {
      type: "table",
      table: "cells-csv",
    } as PointsDataSource;
    expect(
      addPointsDataObject(
        "Cells",
        "layer",
        "https://data.example/cells.csv",
        preparedDataSource,
      ),
    ).toBe("cells-csv");
    expect(projectStore.getState().points).toMatchObject([
      { id: "cells-csv", dataSource: preparedDataSource },
    ]);
  });
});
