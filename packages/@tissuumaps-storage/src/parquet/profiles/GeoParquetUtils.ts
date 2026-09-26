import type { Geometry } from "geojson";
import type { AsyncBuffer, FileMetaData } from "hyparquet";

import type { IDArray, ShapesGeometry } from "@tissuumaps/core";

import { ShapesGeometryBuilder } from "../../common/ShapesGeometryBuilder";
import { ParquetUtils } from "../ParquetUtils";

/**
 * A geometry column of a GeoParquet file
 *
 * Only columns encoded as WKB are described: other encodings are not decoded
 * by the Parquet reader and are read as their raw values.
 */
export type GeoColumn = {
  /** Name of the column */
  name: string;

  /** Whether the column is the file's primary geometry column */
  primary: boolean;

  /** The geometry types in the column, as listed in the `geo` metadata */
  geometryTypes: string[];

  /** The column's `[minX, minY, maxX, maxY]` bounds, if listed */
  bbox: [number, number, number, number] | undefined;
};

/** A column of point coordinates derived from a point geometry column */
export type CoordinateColumn = {
  /** Name of the derived column, e.g. `geometry[x]` */
  column: string;

  /** Name of the geometry column the coordinates are read from */
  geometryColumn: string;

  /** The axis the coordinates are read on */
  axis: "x" | "y";
};

/** The `geo` metadata of a GeoParquet file, as written by the GeoParquet spec */
type GeoMetadata = {
  primary_column?: string;
  columns?: {
    [name: string]: {
      encoding?: string;
      geometry_types?: string[];
      bbox?: number[];
    };
  };
};

/**
 * Reads the `geo` metadata and the geometry columns of a GeoParquet file
 *
 * GeoParquet describes its geometry columns in the `geo` metadata. Point
 * geometry columns are read as a pair of coordinate columns selected from the
 * geometry column, e.g. `geometry[x]` and `geometry[y]`, so that point
 * geometries can be used wherever a numeric column is expected. Polygon
 * geometry columns are read as shapes.
 */
export class GeoParquetUtils {
  /** Reads the 2D bounds of a `bbox`, which lists Z bounds too for 3D columns */
  private static _readBBox(
    bbox: number[] | undefined,
  ): [number, number, number, number] | undefined {
    if (bbox?.length === 4) {
      return [bbox[0]!, bbox[1]!, bbox[2]!, bbox[3]!];
    }
    if (bbox?.length === 6) {
      return [bbox[0]!, bbox[1]!, bbox[3]!, bbox[4]!];
    }
    return undefined;
  }

  /**
   * Returns the geometry columns listed in the `geo` metadata of a file
   *
   * @param metadata - The file metadata
   * @returns The geometry columns, in metadata order, or an empty array for
   * files without GeoParquet metadata
   */
  static getGeoColumns(metadata: FileMetaData): GeoColumn[] {
    const geo = metadata.key_value_metadata?.find(({ key }) => key === "geo");
    if (geo?.value === undefined) {
      return [];
    }
    const { primary_column, columns = {} } = JSON.parse(
      geo.value,
    ) as GeoMetadata;
    return Object.entries(columns)
      .filter(([, column]) => column.encoding === "WKB")
      .map(([name, column]) => ({
        name,
        primary: name === primary_column,
        geometryTypes: column.geometry_types ?? [],
        bbox: GeoParquetUtils._readBBox(column.bbox),
      }));
  }

  /**
   * Returns the primary geometry column of a file
   *
   * @param geoColumns - The geometry columns of the file
   * @returns The column marked as primary, the first geometry column of files
   * that do not mark one, or `undefined` for files without geometry columns
   */
  static getPrimaryColumn(geoColumns: GeoColumn[]): GeoColumn | undefined {
    return geoColumns.find(({ primary }) => primary) ?? geoColumns[0];
  }

  /**
   * Returns whether a geometry column contains points only
   *
   * @param geoColumn - The geometry column
   * @returns Whether every geometry type of the column is a point
   */
  static isPointColumn(geoColumn: GeoColumn): boolean {
    return (
      geoColumn.geometryTypes.length > 0 &&
      geoColumn.geometryTypes.every((geometryType) =>
        geometryType.startsWith("Point"),
      )
    );
  }

  /**
   * Returns the coordinate columns derived from the point geometry columns
   *
   * This is the only place a coordinate column name is built. The name is
   * never parsed back: the mapping travels with the column list, so that a
   * real column whose name looks like one is not mistaken for one.
   *
   * A column that declares no geometry types may hold points too, so it gets
   * the pair as well: reading it fails on the first row that is not a point.
   *
   * @param geoColumns - The geometry columns of the file
   * @returns The derived coordinate columns, each with the geometry column it
   * reads and the axis it reads on
   */
  static getCoordinateColumns(geoColumns: GeoColumn[]): CoordinateColumn[] {
    return geoColumns
      .filter(
        (geoColumn) =>
          geoColumn.geometryTypes.length === 0 ||
          GeoParquetUtils.isPointColumn(geoColumn),
      )
      .flatMap(({ name }) => [
        { column: `${name}[x]`, geometryColumn: name, axis: "x" as const },
        { column: `${name}[y]`, geometryColumn: name, axis: "y" as const },
      ]);
  }

  /**
   * Replaces the geometry columns of a file by the coordinate columns derived
   * from them
   *
   * @param metadata - The file metadata
   * @param columns - The columns of the file, in schema order
   * @returns The columns a table exposes, without the geometry columns and
   * with the derived coordinate columns appended, and those coordinate columns
   */
  static replaceGeometryColumns(
    metadata: FileMetaData,
    columns: string[],
  ): { columns: string[]; coordinateColumns: CoordinateColumn[] } {
    const geoColumns = GeoParquetUtils.getGeoColumns(metadata);
    const coordinateColumns = GeoParquetUtils.getCoordinateColumns(geoColumns);
    return {
      columns: [
        ...columns.filter(
          (column) => !geoColumns.some(({ name }) => name === column),
        ),
        ...coordinateColumns.map(({ column }) => column),
      ],
      coordinateColumns,
    };
  }

  /**
   * Returns the range of one axis of a geometry column from its `bbox`
   *
   * @param metadata - The file metadata
   * @param column - The geometry column
   * @param axis - The axis to read the range of
   * @returns The [min, max] range, or `undefined` if the column lists no
   * `bbox`
   */
  static getAxisRange(
    metadata: FileMetaData,
    column: string,
    axis: "x" | "y",
  ): [number, number] | undefined {
    const { bbox } = GeoParquetUtils.getGeoColumns(metadata).find(
      ({ name }) => name === column,
    ) ?? { bbox: undefined };
    if (bbox === undefined) {
      return undefined;
    }
    return axis === "x" ? [bbox[0], bbox[2]] : [bbox[1], bbox[3]];
  }

  /**
   * Reads a geometry column row by row
   *
   * @param buffer - The file to read from
   * @param metadata - The file metadata
   * @param column - The geometry column to read
   * @param onGeometry - Called with the geometry of each row, `null` for a
   * row without one
   * @param onProgress - Callback reporting the read progress
   */
  static readGeometryColumn(
    buffer: AsyncBuffer,
    metadata: FileMetaData,
    column: string,
    onGeometry: (geometry: Geometry | null, row: number) => void,
    onProgress: (progress: number, total: number) => void,
  ): Promise<void> {
    return ParquetUtils.readColumnChunks(
      buffer,
      metadata,
      column,
      (columnData, rowStart) => {
        // WKB columns are decoded to GeoJSON geometries by the Parquet reader
        const geometries = columnData as (Geometry | null | undefined)[];
        for (let i = 0; i < geometries.length; i++) {
          onGeometry(geometries[i] ?? null, rowStart + i);
        }
      },
      onProgress,
    );
  }

  /**
   * Reads both coordinate axes of a point geometry column in one pass
   *
   * The WKB column is decoded once for both axes, so that a point cloud reading
   * its x and y from the same column does not decode it twice.
   *
   * @param buffer - The file to read from
   * @param metadata - The file metadata
   * @param column - The point geometry column to read
   * @param onProgress - Callback reporting the read progress
   * @returns The x and y coordinates of every row
   * @throws Error if a row holds no geometry, or one that is not a point
   */
  static async readCoordinateColumns(
    buffer: AsyncBuffer,
    metadata: FileMetaData,
    column: string,
    onProgress: (progress: number, total: number) => void,
  ): Promise<{ x: Float32Array; y: Float32Array }> {
    const numRows = ParquetUtils.getNumRows(metadata);
    const x = new Float32Array(numRows);
    const y = new Float32Array(numRows);
    await GeoParquetUtils.readGeometryColumn(
      buffer,
      metadata,
      column,
      (geometry, row) => {
        if (geometry === null) {
          throw new Error(`Missing geometry in column "${column}"`);
        }
        if (geometry.type !== "Point") {
          throw new Error(
            `Column "${column}" contains a ${geometry.type} geometry`,
          );
        }
        x[row] = geometry.coordinates[0]!;
        y[row] = geometry.coordinates[1]!;
      },
      onProgress,
    );
    return { x, y };
  }

  /**
   * Resolves the geometry column shapes are read from
   *
   * @param metadata - The file metadata
   * @param geometryColumn - The requested column, or `undefined` for the
   * primary geometry column of the file
   * @returns The geometry column
   * @throws Error if the requested column is missing or not encoded as WKB, if
   * the file has no geometry column, or if the column holds points
   */
  static getShapesColumn(
    metadata: FileMetaData,
    geometryColumn: string | undefined,
  ): GeoColumn {
    const geoColumns = GeoParquetUtils.getGeoColumns(metadata);
    const geoColumn =
      geometryColumn !== undefined
        ? geoColumns.find(({ name }) => name === geometryColumn)
        : GeoParquetUtils.getPrimaryColumn(geoColumns);
    if (geoColumn === undefined) {
      throw new Error(
        geometryColumn !== undefined
          ? `Geometry column "${geometryColumn}" is missing or not encoded as WKB`
          : "Parquet file has no GeoParquet geometry column",
      );
    }
    if (GeoParquetUtils.isPointColumn(geoColumn)) {
      throw new Error(GeoParquetUtils._pointColumnMessage(geoColumn.name));
    }
    return geoColumn;
  }

  /**
   * Reads the polygons of a geometry column as shapes
   *
   * Rows without a geometry are skipped, and so are points, which are not
   * shapes (see {@link GeoParquetUtils.getShapesColumn}).
   *
   * @param buffer - The file to read from
   * @param metadata - The file metadata
   * @param geoColumn - The geometry column to read
   * @param options - The IDs and names of the rows, `undefined` to key the
   * shapes by row number, and a callback reporting the read progress
   * @returns The shapes geometry with the IDs and names of the shapes
   * @throws Error if the column holds no valid geometry
   */
  static async readShapes(
    buffer: AsyncBuffer,
    metadata: FileMetaData,
    geoColumn: GeoColumn,
    options: {
      ids: IDArray | undefined;
      names: string[] | undefined;
      onProgress: (progress: number, total: number) => void;
    },
  ): Promise<{
    geometry: ShapesGeometry;
    ids: IDArray;
    names: string[] | undefined;
  }> {
    const { ids: rowIds, names: rowNames, onProgress } = options;
    const builder = new ShapesGeometryBuilder();
    // A column whose "geo" metadata lists no geometry types is only found to
    // hold points while decoding it
    let numPoints = 0;
    await GeoParquetUtils.readGeometryColumn(
      buffer,
      metadata,
      geoColumn.name,
      (rowGeometry, row) => {
        if (rowGeometry === null) {
          console.warn("Skipping row without geometry.");
          return;
        }
        if (rowGeometry.type === "Point") {
          numPoints++;
          return;
        }
        GeoParquetUtils._addShape(
          builder,
          rowGeometry,
          rowIds !== undefined ? rowIds[row]! : row,
          rowNames?.[row],
        );
      },
      onProgress,
    );
    if (builder.size === 0) {
      throw new Error(
        numPoints > 0
          ? GeoParquetUtils._pointColumnMessage(geoColumn.name)
          : `No valid geometries found in column "${geoColumn.name}"`,
      );
    }
    if (numPoints > 0) {
      console.warn(
        `Skipped ${numPoints} points in column "${geoColumn.name}", which are ` +
          `not shapes.`,
      );
    }
    return builder.build();
  }

  /** The error for a column of points asked for as shapes */
  private static _pointColumnMessage(name: string): string {
    return (
      `Geometry column "${name}" holds points, which are read as ` +
      `the "${name}[x]" and "${name}[y]" columns of a table`
    );
  }

  /**
   * Appends a GeoJSON geometry, as decoded from a WKB column, as one shape
   *
   * @param builder - The builder of the shapes geometry under construction
   * @param geometry - The geometry to add
   * @param id - The ID of the shape
   * @param name - The name of the shape, if any
   */
  private static _addShape(
    builder: ShapesGeometryBuilder,
    geometry: Geometry,
    id: number | string,
    name?: string,
  ): void {
    if (geometry.type === "Polygon") {
      builder.addShape([geometry.coordinates], id, name);
      return;
    }
    if (geometry.type === "MultiPolygon") {
      builder.addShape(geometry.coordinates, id, name);
      return;
    }
    console.warn(`Unsupported geometry type: ${geometry.type}`);
  }
}
