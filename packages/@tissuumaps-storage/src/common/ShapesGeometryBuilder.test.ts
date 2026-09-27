import { describe, expect, it, vi } from "vitest";

import {
  ShapesGeometryBuilder,
  type ShapesRing,
} from "./ShapesGeometryBuilder";

const square: ShapesRing = [
  [0, 0],
  [4, 0],
  [4, 4],
  [0, 4],
];

const hole: ShapesRing = [
  [1, 1],
  [2, 1],
  [2, 2],
];

describe("ShapesGeometryBuilder", () => {
  it("adds a polygon as one shape with one ring", () => {
    const builder = new ShapesGeometryBuilder();
    builder.addShape([[square]], 7);
    const { geometry, ids, names } = builder.build();
    expect(Array.from(geometry.shapePolygonOffsets)).toEqual([0, 1]);
    expect(Array.from(geometry.polygonRingOffsets)).toEqual([0, 1]);
    expect(Array.from(geometry.ringVertexOffsets)).toEqual([0, 4]);
    expect(Array.from(geometry.coords)).toEqual([0, 0, 4, 0, 4, 4, 0, 4]);
    expect(ids).toEqual(new Float64Array([7]));
    expect(names).toBeUndefined();
  });

  it("adds the holes of a polygon as further rings", () => {
    const builder = new ShapesGeometryBuilder();
    builder.addShape([[square, hole]], 0);
    const { geometry } = builder.build();
    expect(Array.from(geometry.shapePolygonOffsets)).toEqual([0, 1]);
    expect(Array.from(geometry.polygonRingOffsets)).toEqual([0, 2]);
    expect(Array.from(geometry.ringVertexOffsets)).toEqual([0, 4, 7]);
  });

  it("adds several polygons as one shape", () => {
    const builder = new ShapesGeometryBuilder();
    builder.addShape([[square], [square, hole]], 0);
    const { geometry } = builder.build();
    expect(Array.from(geometry.shapePolygonOffsets)).toEqual([0, 2]);
    expect(Array.from(geometry.polygonRingOffsets)).toEqual([0, 1, 3]);
    expect(Array.from(geometry.ringVertexOffsets)).toEqual([0, 4, 8, 11]);
  });

  it("adds several shapes with their IDs and names", () => {
    const builder = new ShapesGeometryBuilder();
    builder.addShape([[square]], 10, "first");
    builder.addShape([[square]], 20, "second");
    const { geometry, ids, names } = builder.build();
    expect(Array.from(geometry.shapePolygonOffsets)).toEqual([0, 1, 2]);
    expect(Array.from(geometry.polygonRingOffsets)).toEqual([0, 1, 2]);
    expect(ids).toEqual(new Float64Array([10, 20]));
    expect(names).toEqual(["first", "second"]);
  });

  it("skips a polygon without a valid shell", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const builder = new ShapesGeometryBuilder();
    builder.addShape(
      [
        [
          [
            [0, 0],
            [1, 1],
          ],
        ],
      ],
      7,
    );
    const { geometry, ids } = builder.build();
    expect(Array.from(geometry.shapePolygonOffsets)).toEqual([0]);
    expect(ids).toEqual(new Uint32Array(0));
    warn.mockRestore();
  });

  it("skips a shape without polygons", () => {
    const builder = new ShapesGeometryBuilder();
    builder.addShape([], 7);
    const { geometry, ids } = builder.build();
    expect(Array.from(geometry.shapePolygonOffsets)).toEqual([0]);
    expect(ids).toEqual(new Uint32Array(0));
  });

  it("keeps the IDs aligned with the geometry when a shape is skipped", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const builder = new ShapesGeometryBuilder();
    builder.addShape([[square]], 10);
    builder.addShape([], 20);
    builder.addShape([[square]], 30);
    const { geometry, ids } = builder.build();
    expect(geometry.shapePolygonOffsets.length - 1).toBe(2);
    expect(ids).toEqual(new Float64Array([10, 30]));
    warn.mockRestore();
  });

  it("reads no names when only some shapes are named", () => {
    const builder = new ShapesGeometryBuilder();
    builder.addShape([[square]], 10, "first");
    builder.addShape([[square]], 20);
    expect(builder.build().names).toBeUndefined();
  });

  it("refuses to add a shape once the geometry is built", () => {
    const builder = new ShapesGeometryBuilder();
    builder.addShape([[square]], 10);
    const { ids } = builder.build();
    expect(() => builder.addShape([[square]], 20)).toThrow(
      "Shapes cannot be added once the geometry is built.",
    );
    expect(() => builder.build()).toThrow(
      "The geometry has already been built.",
    );
    expect(ids).toEqual(new Float64Array([10]));
  });

  it("collects string IDs", () => {
    const builder = new ShapesGeometryBuilder();
    builder.addShape([[square]], "a");
    builder.addShape([[square]], "b");
    expect(builder.build().ids).toEqual(["a", "b"]);
  });

  it("refuses to build shapes mixing integer and string IDs", () => {
    const builder = new ShapesGeometryBuilder();
    builder.addShape([[square]], 1);
    builder.addShape([[square]], "b");
    expect(() => builder.build()).toThrow("ID is not a safe integer: b");
  });

  it("counts the shapes added so far", () => {
    const builder = new ShapesGeometryBuilder();
    expect(builder.size).toBe(0);
    builder.addShape([[square]], 10);
    expect(builder.size).toBe(1);
  });
});
