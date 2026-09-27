import type { Geometry } from "geojson";
import type { AsyncBuffer, FileMetaData } from "hyparquet";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ParquetUtils } from "../ParquetUtils";
import { GeoParquetUtils } from "./GeoParquetUtils";

function fakeMetadata(geo?: string, numRows = 0): FileMetaData {
  return {
    key_value_metadata: geo !== undefined ? [{ key: "geo", value: geo }] : [],
    num_rows: BigInt(numRows),
  } as unknown as FileMetaData;
}

const buffer: AsyncBuffer = {
  byteLength: 8,
  slice: () => Promise.resolve(new ArrayBuffer(0)),
};

/**
 * Feeds the given chunks of decoded geometries to whatever reads the column,
 * as the Parquet reader would, and reports the file as fully read
 */
function fakeGeometryChunks(chunks: (Geometry | null | undefined)[][]) {
  return vi
    .spyOn(ParquetUtils, "readColumnChunks")
    .mockImplementation((buffer, _metadata, _column, onChunk, onProgress) => {
      let rowStart = 0;
      for (const chunk of chunks) {
        onChunk(chunk, rowStart);
        rowStart += chunk.length;
      }
      onProgress(buffer.byteLength, buffer.byteLength);
      return Promise.resolve();
    });
}

function point(x: number, y: number): Geometry {
  return { type: "Point", coordinates: [x, y] };
}

/** A triangle with its corner at the given position */
function triangle(x: number, y: number): number[][] {
  return [
    [x, y],
    [x + 1, y],
    [x, y + 1],
  ];
}

const polygon: Geometry = { type: "Polygon", coordinates: [triangle(0, 0)] };
const multiPolygon: Geometry = {
  type: "MultiPolygon",
  coordinates: [[triangle(2, 2)], [triangle(4, 4)]],
};

// As written by geopandas 1.1.1 for a SpatialData circles element
const pointsGeo = JSON.stringify({
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
});
const points = fakeMetadata(pointsGeo);

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
  afterEach(() => {
    vi.restoreAllMocks();
  });

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

  describe("isPointColumn", () => {
    it("recognizes a column of points, in any dimension", () => {
      const [geoColumn] = GeoParquetUtils.getGeoColumns(points);
      expect(GeoParquetUtils.isPointColumn(geoColumn!)).toBe(true);
      expect(
        GeoParquetUtils.isPointColumn({
          ...geoColumn!,
          geometryTypes: ["Point", "Point Z"],
        }),
      ).toBe(true);
    });

    it("does not recognize a column declaring no geometry types", () => {
      const [geoColumn] = GeoParquetUtils.getGeoColumns(points);
      expect(
        GeoParquetUtils.isPointColumn({ ...geoColumn!, geometryTypes: [] }),
      ).toBe(false);
    });

    it("does not recognize a column mixing points with other geometries", () => {
      const [geoColumn] = GeoParquetUtils.getGeoColumns(points);
      expect(
        GeoParquetUtils.isPointColumn({
          ...geoColumn!,
          geometryTypes: ["Point", "Polygon"],
        }),
      ).toBe(false);
    });
  });

  describe("replaceGeometryColumns", () => {
    it("replaces a point column by its coordinate columns, appended last", () => {
      expect(
        GeoParquetUtils.replaceGeometryColumns(points, ["geometry", "radius"]),
      ).toEqual({
        columns: ["radius", "geometry[x]", "geometry[y]"],
        coordinateColumns: [
          { column: "geometry[x]", geometryColumn: "geometry", axis: "x" },
          { column: "geometry[y]", geometryColumn: "geometry", axis: "y" },
        ],
      });
    });

    it("drops a polygon column without a replacement", () => {
      expect(
        GeoParquetUtils.replaceGeometryColumns(polygons, ["geometry", "area"]),
      ).toEqual({ columns: ["area"], coordinateColumns: [] });
    });

    it("keeps the columns of a file without GeoParquet metadata", () => {
      expect(
        GeoParquetUtils.replaceGeometryColumns(fakeMetadata(), ["x", "y"]),
      ).toEqual({ columns: ["x", "y"], coordinateColumns: [] });
    });
  });

  describe("getAxisRange", () => {
    it("reads the range of each axis from the bounding box", () => {
      expect(GeoParquetUtils.getAxisRange(points, "geometry", "x")).toEqual([
        309, 2748,
      ]);
      expect(GeoParquetUtils.getAxisRange(points, "geometry", "y")).toEqual([
        328, 2727,
      ]);
    });

    it("reads no range from a column without a bounding box", () => {
      const metadata = fakeMetadata(
        JSON.stringify({
          primary_column: "geometry",
          columns: { geometry: { encoding: "WKB", geometry_types: ["Point"] } },
        }),
      );
      expect(
        GeoParquetUtils.getAxisRange(metadata, "geometry", "x"),
      ).toBeUndefined();
    });

    it("reads no range for a column that is not a geometry column", () => {
      expect(
        GeoParquetUtils.getAxisRange(points, "radius", "x"),
      ).toBeUndefined();
    });
  });

  describe("getShapesColumn", () => {
    it("resolves the primary geometry column by default", () => {
      expect(GeoParquetUtils.getShapesColumn(polygons, undefined).name).toBe(
        "geometry",
      );
    });

    it("resolves the requested geometry column", () => {
      const metadata = fakeMetadata(
        JSON.stringify({
          primary_column: "centroid",
          columns: {
            centroid: { encoding: "WKB", geometry_types: ["Point"] },
            outline: { encoding: "WKB", geometry_types: ["Polygon"] },
          },
        }),
      );
      expect(GeoParquetUtils.getShapesColumn(metadata, "outline").name).toBe(
        "outline",
      );
    });

    it("rejects a requested column that is missing or not WKB", () => {
      expect(() =>
        GeoParquetUtils.getShapesColumn(polygons, "outline"),
      ).toThrow('Geometry column "outline" is missing or not encoded as WKB');
    });

    it("rejects a file without geometry columns", () => {
      expect(() =>
        GeoParquetUtils.getShapesColumn(fakeMetadata(), undefined),
      ).toThrow("Parquet file has no GeoParquet geometry column");
    });

    it("rejects a column of points, pointing to its coordinate columns", () => {
      expect(() => GeoParquetUtils.getShapesColumn(points, undefined)).toThrow(
        'Geometry column "geometry" holds points, which are read as the ' +
          '"geometry[x]" and "geometry[y]" columns of a table',
      );
    });
  });

  describe("readGeometryColumn", () => {
    it("reads the requested column row by row across chunks", async () => {
      const readColumnChunks = fakeGeometryChunks([
        [point(1, 2), null],
        [undefined, polygon],
      ]);
      const onGeometry = vi.fn();
      const onProgress = vi.fn();
      await GeoParquetUtils.readGeometryColumn(
        buffer,
        polygons,
        "geometry",
        onGeometry,
        onProgress,
      );
      expect(readColumnChunks).toHaveBeenCalledWith(
        buffer,
        polygons,
        "geometry",
        expect.any(Function),
        onProgress,
      );
      expect(onGeometry.mock.calls).toEqual([
        [point(1, 2), 0],
        [null, 1],
        [null, 2],
        [polygon, 3],
      ]);
      expect(onProgress).toHaveBeenCalledWith(8, 8);
    });
  });

  describe("readCoordinateColumns", () => {
    const threePoints = fakeMetadata(pointsGeo, 3);

    it("reads both axes of a point column in one pass", async () => {
      const readColumnChunks = fakeGeometryChunks([
        [point(1, 2), point(3, 4)],
        [point(5, 6)],
      ]);
      await expect(
        GeoParquetUtils.readCoordinateColumns(
          buffer,
          threePoints,
          "geometry",
          () => {},
        ),
      ).resolves.toEqual({
        x: new Float32Array([1, 3, 5]),
        y: new Float32Array([2, 4, 6]),
      });
      expect(readColumnChunks).toHaveBeenCalledTimes(1);
    });

    it("rejects a row without a geometry", async () => {
      fakeGeometryChunks([[point(1, 2), null, point(5, 6)]]);
      await expect(
        GeoParquetUtils.readCoordinateColumns(
          buffer,
          threePoints,
          "geometry",
          () => {},
        ),
      ).rejects.toThrow('Missing geometry in column "geometry"');
    });

    it("rejects a row that is not a point", async () => {
      fakeGeometryChunks([[point(1, 2), polygon, point(5, 6)]]);
      await expect(
        GeoParquetUtils.readCoordinateColumns(
          buffer,
          threePoints,
          "geometry",
          () => {},
        ),
      ).rejects.toThrow('Column "geometry" contains a Polygon geometry');
    });
  });

  describe("readShapes", () => {
    const geoColumn = GeoParquetUtils.getShapesColumn(polygons, undefined);

    function readShapes(options?: {
      ids?: Uint32Array;
      names?: string[];
      onProgress?: (progress: number, total: number) => void;
    }) {
      return GeoParquetUtils.readShapes(buffer, polygons, geoColumn, {
        ids: options?.ids,
        names: options?.names,
        onProgress: options?.onProgress ?? (() => {}),
      });
    }

    it("reads polygons and multi-polygons with the IDs and names of their rows", async () => {
      fakeGeometryChunks([[polygon], [multiPolygon]]);
      const { geometry, ids, names } = await readShapes({
        ids: new Uint32Array([7, 8]),
        names: ["a", "b"],
      });
      expect(geometry).toEqual({
        shapePolygonOffsets: new Uint32Array([0, 1, 3]),
        polygonRingOffsets: new Uint32Array([0, 1, 2, 3]),
        ringVertexOffsets: new Uint32Array([0, 3, 6, 9]),
        coords: new Float32Array([
          ...triangle(0, 0).flat(),
          ...triangle(2, 2).flat(),
          ...triangle(4, 4).flat(),
        ]),
      });
      expect([...ids]).toEqual([7, 8]);
      expect(names).toEqual(["a", "b"]);
    });

    it("keys the shapes by row number without IDs", async () => {
      fakeGeometryChunks([[polygon, polygon]]);
      const { ids, names } = await readShapes();
      expect([...ids]).toEqual([0, 1]);
      expect(names).toBeUndefined();
    });

    it("skips rows without a geometry and points, with a warning", async () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      fakeGeometryChunks([[null, point(1, 2), polygon]]);
      const { ids } = await readShapes();
      expect([...ids]).toEqual([2]);
      expect(warn.mock.calls).toEqual([
        ["Skipping row without geometry."],
        ['Skipped 1 points in column "geometry", which are not shapes.'],
      ]);
    });

    it("skips geometries that are not polygons, with a warning", async () => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      fakeGeometryChunks([
        [
          {
            type: "LineString",
            coordinates: [
              [0, 0],
              [1, 1],
            ],
          },
          polygon,
        ],
      ]);
      const { ids } = await readShapes();
      expect([...ids]).toEqual([1]);
      expect(warn).toHaveBeenCalledWith(
        "Unsupported geometry type: LineString",
      );
    });

    it("rejects a column holding points only, pointing to its coordinate columns", async () => {
      vi.spyOn(console, "warn").mockImplementation(() => {});
      fakeGeometryChunks([[point(1, 2), point(3, 4)]]);
      await expect(readShapes()).rejects.toThrow(
        'Geometry column "geometry" holds points, which are read as the ' +
          '"geometry[x]" and "geometry[y]" columns of a table',
      );
    });

    it("rejects a column without a valid geometry", async () => {
      vi.spyOn(console, "warn").mockImplementation(() => {});
      fakeGeometryChunks([[null, null]]);
      await expect(readShapes()).rejects.toThrow(
        'No valid geometries found in column "geometry"',
      );
    });

    it("reports the read progress", async () => {
      fakeGeometryChunks([[polygon]]);
      const onProgress = vi.fn();
      await readShapes({ onProgress });
      expect(onProgress).toHaveBeenCalledWith(8, 8);
    });
  });
});
