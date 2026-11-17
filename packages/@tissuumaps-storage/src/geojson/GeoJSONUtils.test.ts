import type { FeatureCollection, Geometry } from "geojson";
import { describe, expect, it } from "vitest";

import { GeoJSONUtils } from "./GeoJSONUtils";

const square: Geometry = {
  type: "Polygon",
  coordinates: [
    [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 0],
    ],
  ],
};

function collection(
  features: {
    geometry?: Geometry | null;
    properties: Record<string, unknown> | null;
  }[],
): FeatureCollection<Geometry | null> {
  return {
    type: "FeatureCollection",
    features: features.map(({ geometry = square, properties }) => ({
      type: "Feature",
      geometry,
      properties,
    })),
  };
}

describe("GeoJSONUtils", () => {
  describe("readProperties", () => {
    it("reads the properties as typed columns", () => {
      const { ids, columns, columnValues } = GeoJSONUtils.readProperties(
        collection([
          { properties: { cell_type: "T", area: 2, tumor: true } },
          { properties: { cell_type: "B", area: null, tumor: false } },
          { properties: { cell_type: 3, nested: { a: 1 } } },
        ]),
        undefined,
      );
      expect(Array.from(ids as ArrayLike<number>)).toEqual([0, 1, 2]);
      expect(columns).toEqual(["cell_type", "area", "tumor", "nested"]);
      expect(columnValues.get("cell_type")).toEqual(["T", "B", "3"]);
      expect(columnValues.get("area")).toEqual(new Float64Array([2, NaN, NaN]));
      expect(columnValues.get("tumor")).toEqual([true, false, undefined]);
      expect(columnValues.get("nested")).toEqual(["", "", '{"a":1}']);
    });

    it("keys the rows by the IDs of the shapes", () => {
      const { ids } = GeoJSONUtils.readProperties(
        collection([
          { properties: { id: "a" } },
          { geometry: null, properties: { id: "skipped" } },
          {
            geometry: { type: "Point", coordinates: [0, 0] },
            properties: { id: "point" },
          },
          { properties: { id: "c" } },
        ]),
        "id",
      );
      expect(ids).toEqual(["a", "c"]);
    });

    it("keys the rows by feature index without an ID property", () => {
      const { ids } = GeoJSONUtils.readProperties(
        collection([{ geometry: null, properties: {} }, { properties: {} }]),
        undefined,
      );
      expect(Array.from(ids as ArrayLike<number>)).toEqual([1]);
    });

    it("throws for a feature missing its ID", () => {
      expect(() =>
        GeoJSONUtils.readProperties(collection([{ properties: {} }]), "id"),
      ).toThrow("missing ID");
    });
  });
});
