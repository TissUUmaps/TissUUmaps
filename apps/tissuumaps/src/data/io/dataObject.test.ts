import { beforeEach, describe, expect, it } from "vitest";

import type { ImageDataSource, PointsDataSource } from "@tissuumaps/core";

import { appStore } from "@/stores/app";
import { projectStore } from "@/stores/project";

import {
  addImageDataObject,
  addPointsDataObject,
  createDataObjectID,
} from "./dataObject";

const uuid = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";

describe("createDataObjectID", () => {
  it("names the ID after the file name, with hyphens for dots and a UUID", () => {
    expect(createDataObjectID("/a/cells.ome.zarr")).toMatch(
      new RegExp(`^cells-ome-zarr-${uuid}$`),
    );
    expect(createDataObjectID("/a/x.zarr/labels/cells")).toMatch(
      new RegExp(`^cells-${uuid}$`),
    );
    expect(
      createDataObjectID("https://data.example/a/my%20cells.v2.csv?x=1"),
    ).toMatch(new RegExp(`^my cells-v2-csv-${uuid}$`));
  });

  it("falls back to the UUID alone", () => {
    expect(createDataObjectID(undefined)).toMatch(new RegExp(`^${uuid}$`));
    expect(createDataObjectID("data:text/csv,a")).toMatch(
      new RegExp(`^${uuid}$`),
    );
  });

  it("creates a new ID for the same source every time", () => {
    expect(createDataObjectID("/a/cells.csv")).not.toBe(
      createDataObjectID("/a/cells.csv"),
    );
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
    const id = addImageDataObject("Cells", "layer", source, dataSource);
    expect(id).toMatch(new RegExp(`^cells-ome-tif-${uuid}$`));
    expect(projectStore.getState().images).toMatchObject([
      { id, name: "Cells", layer: "layer", dataSource },
    ]);
    expect(appStore.getState().expandedImageIds).toEqual([id]);
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
    const id = addPointsDataObject(
      "Cells",
      "layer",
      "https://data.example/cells.csv",
      preparedDataSource,
    );
    expect(id).toMatch(new RegExp(`^cells-csv-${uuid}$`));
    expect(projectStore.getState().points).toMatchObject([
      { id, dataSource: preparedDataSource },
    ]);
  });
});
