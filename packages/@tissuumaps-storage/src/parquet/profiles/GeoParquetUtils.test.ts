import type { FileMetaData } from "hyparquet";
import { describe, expect, it } from "vitest";

import { GeoParquetUtils } from "./GeoParquetUtils";

function fakeMetadata(geo?: string): FileMetaData {
  return {
    key_value_metadata: geo !== undefined ? [{ key: "geo", value: geo }] : [],
  } as unknown as FileMetaData;
}

// As written by geopandas 1.1.1 for a SpatialData circles element
const points = fakeMetadata(
  JSON.stringify({
    primary_column: "geometry",
    columns: {
      geometry: {
        encoding: "WKB",
        crs: null,
        geometry_types: ["Point"],
        bbox: [309, 328, 2748, 2727],
      },
    },
    version: "1.0.0",
  }),
);

// As written by geopandas 1.0.1 for a SpatialData polygons element
const polygons = fakeMetadata(
  JSON.stringify({
    primary_column: "geometry",
    columns: {
      geometry: {
        encoding: "WKB",
        crs: null,
        geometry_types: ["Polygon"],
        bbox: [0, 0, 1499, 1499],
      },
    },
    version: "1.0.0",
  }),
);

describe("GeoParquetUtils", () => {
  describe("getGeoColumns", () => {
    it("reads the geometry columns of a GeoParquet file", () => {
      expect(GeoParquetUtils.getGeoColumns(points)).toEqual([
        {
          name: "geometry",
          primary: true,
          geometryTypes: ["Point"],
          bbox: [309, 328, 2748, 2727],
        },
      ]);
    });

    it("reads no columns from a file without GeoParquet metadata", () => {
      expect(GeoParquetUtils.getGeoColumns(fakeMetadata())).toEqual([]);
    });

    it("skips columns that are not encoded as WKB", () => {
      const metadata = fakeMetadata(
        JSON.stringify({
          primary_column: "geometry",
          columns: { geometry: { encoding: "point" } },
        }),
      );
      expect(GeoParquetUtils.getGeoColumns(metadata)).toEqual([]);
    });

    it("reads no bounds from a column without a complete bounding box", () => {
      const metadata = fakeMetadata(
        JSON.stringify({
          primary_column: "geometry",
          columns: {
            geometry: { encoding: "WKB", geometry_types: ["Point"], bbox: [0] },
          },
        }),
      );
      expect(GeoParquetUtils.getGeoColumns(metadata)[0]!.bbox).toBeUndefined();
    });

    it("reads the 2D bounds of a 3D bounding box", () => {
      const metadata = fakeMetadata(
        JSON.stringify({
          primary_column: "geometry",
          columns: {
            geometry: { encoding: "WKB", bbox: [0, 1, 2, 10, 11, 12] },
          },
        }),
      );
      expect(GeoParquetUtils.getGeoColumns(metadata)[0]!.bbox).toEqual([
        0, 1, 10, 11,
      ]);
    });
  });

  describe("getPrimaryColumn", () => {
    it("returns the column marked as primary", () => {
      const geoColumns = GeoParquetUtils.getGeoColumns(
        fakeMetadata(
          JSON.stringify({
            primary_column: "outline",
            columns: {
              centroid: { encoding: "WKB", geometry_types: ["Point"] },
              outline: { encoding: "WKB", geometry_types: ["Polygon"] },
            },
          }),
        ),
      );
      expect(GeoParquetUtils.getPrimaryColumn(geoColumns)?.name).toBe(
        "outline",
      );
    });

    it("returns no column for a file without geometry columns", () => {
      expect(GeoParquetUtils.getPrimaryColumn([])).toBeUndefined();
    });
  });

  describe("getCoordinateColumns", () => {
    it("derives a coordinate column pair per point geometry column", () => {
      const geoColumns = GeoParquetUtils.getGeoColumns(points);
      expect(GeoParquetUtils.getCoordinateColumns(geoColumns)).toEqual([
        { column: "geometry[x]", geometryColumn: "geometry", axis: "x" },
        { column: "geometry[y]", geometryColumn: "geometry", axis: "y" },
      ]);
    });

    it("derives a pair from a column that declares no geometry types", () => {
      const geoColumns = GeoParquetUtils.getGeoColumns(
        fakeMetadata(
          JSON.stringify({
            primary_column: "geometry",
            columns: { geometry: { encoding: "WKB", geometry_types: [] } },
          }),
        ),
      );
      expect(GeoParquetUtils.getCoordinateColumns(geoColumns)).toEqual([
        { column: "geometry[x]", geometryColumn: "geometry", axis: "x" },
        { column: "geometry[y]", geometryColumn: "geometry", axis: "y" },
      ]);
    });

    it("derives no coordinate columns from polygons", () => {
      const geoColumns = GeoParquetUtils.getGeoColumns(polygons);
      expect(GeoParquetUtils.getCoordinateColumns(geoColumns)).toEqual([]);
    });
  });
});
