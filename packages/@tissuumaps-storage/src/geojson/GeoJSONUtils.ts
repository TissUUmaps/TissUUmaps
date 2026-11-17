import type { Feature, GeoJSON, Geometry } from "geojson";

import {
  ArrayUtils,
  type IDArray,
  type TypedArrayOrArray,
} from "@tissuumaps/core";

/**
 * Reads the features of GeoJSON data: their IDs, and their properties as the
 * columns of a table keyed by the same IDs as the shapes
 */
export class GeoJSONUtils {
  /**
   * Reads the properties of the features that are read as shapes (see
   * `parseGeoJSON`), keyed by the same IDs
   *
   * A property whose values are all numbers (or missing) is a numeric column,
   * with `NaN` for missing values; one whose values are all booleans is a
   * boolean column; any other property is a string column, with the empty
   * string for missing values and JSON for nested values.
   *
   * @param geo - The GeoJSON data
   * @param idProperty - The property holding the feature IDs, if any
   * @returns The IDs and the columns
   */
  static readProperties(
    geo: GeoJSON<Geometry | null>,
    idProperty: string | undefined,
  ): {
    ids: IDArray;
    columns: string[];
    columnValues: Map<string, TypedArrayOrArray<unknown>>;
  } {
    if (geo === null) {
      throw new Error("GeoJSON data must not be null.");
    }
    const features: Feature<Geometry | null>[] =
      geo.type === "FeatureCollection"
        ? geo.features
        : geo.type === "Feature"
          ? [geo]
          : [];
    const ids: (number | string)[] = [];
    const rows: Record<string, unknown>[] = [];
    const columns: string[] = [];
    const seenColumns = new Set<string>();
    for (let i = 0; i < features.length; i++) {
      const feature = features[i]!;
      if (
        feature.geometry === null ||
        !GeoJSONUtils.isPolygonal(feature.geometry)
      ) {
        continue;
      }
      ids.push(
        idProperty !== undefined
          ? GeoJSONUtils.readFeatureId(feature, idProperty)
          : i,
      );
      const properties = feature.properties ?? {};
      rows.push(properties);
      for (const key of Object.keys(properties)) {
        if (!seenColumns.has(key)) {
          seenColumns.add(key);
          columns.push(key);
        }
      }
    }
    const columnValues = new Map<string, TypedArrayOrArray<unknown>>();
    for (const column of columns) {
      const values = rows.map((row) => row[column]);
      const present = values.filter(
        (value) => value !== undefined && value !== null,
      );
      if (present.every((value) => typeof value === "number")) {
        columnValues.set(
          column,
          Float64Array.from(values, (value) =>
            typeof value === "number" ? value : NaN,
          ),
        );
      } else if (present.every((value) => typeof value === "boolean")) {
        columnValues.set(column, values);
      } else {
        columnValues.set(
          column,
          values.map((value) =>
            value === undefined || value === null
              ? ""
              : typeof value === "object"
                ? JSON.stringify(value)
                : String(value as string | number | boolean),
          ),
        );
      }
    }
    return { ids: ArrayUtils.toIDArray(ids), columns, columnValues };
  }

  /**
   * @param geometry - A GeoJSON geometry
   * @returns Whether the geometry is read as a shape
   */
  static isPolygonal(geometry: Geometry): boolean {
    return geometry.type === "Polygon" || geometry.type === "MultiPolygon";
  }

  /**
   * Reads the ID of a feature from a property
   *
   * @param feature - The feature
   * @param idProperty - The property holding the ID
   * @returns The ID, a string or a safe integer
   * @throws Error if the feature is missing the ID or it is of another type
   */
  static readFeatureId(
    feature: Feature<Geometry | null>,
    idProperty: string,
  ): number | string {
    const id = feature.properties?.[idProperty] as unknown;
    if (id === undefined || id === null || id === "") {
      throw new Error(`Feature is missing ID '${idProperty}'`);
    }
    if (typeof id === "string") {
      return id;
    }
    if (typeof id === "number" && Number.isSafeInteger(id)) {
      return id;
    }
    throw new Error(`Feature has invalid ID '${idProperty}'`);
  }
}
