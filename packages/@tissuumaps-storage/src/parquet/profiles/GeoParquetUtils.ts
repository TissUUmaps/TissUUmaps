import type { FileMetaData } from "hyparquet";

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
 * Helpers for the `geo` metadata of a GeoParquet file
 *
 * GeoParquet describes its geometry columns in the `geo` metadata. Point
 * geometry columns are read as a pair of coordinate columns selected from the
 * geometry column, e.g. `geometry[x]` and `geometry[y]`, so that point
 * geometries can be used wherever a numeric column is expected.
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
   * Reads the geometry columns of a file
   *
   * @param metadata - The file metadata
   * @returns The geometry columns, in metadata order, or an empty array for
   * files without GeoParquet metadata
   */
  static readGeoColumns(metadata: FileMetaData): GeoColumn[] {
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
}
