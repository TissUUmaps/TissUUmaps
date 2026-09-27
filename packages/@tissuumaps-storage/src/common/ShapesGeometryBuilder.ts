import {
  ArrayUtils,
  type IDArray,
  type ShapesGeometry,
} from "@tissuumaps/core";

/** A ring of a polygon, as a sequence of positions holding x and y first */
export type ShapesRing = readonly (readonly number[])[];

/** A polygon, as its shell ring followed by its hole rings */
export type ShapesPolygon = readonly ShapesRing[];

/**
 * Accumulator building a {@link ShapesGeometry} one shape at a time
 *
 * The CSR offset arrays are grown as shapes are added, so the shapes of a file
 * can be consumed as they are decoded rather than collected first. The IDs and
 * the names are collected alongside, so that they stay aligned with the
 * geometry when a shape is skipped. IDs are either all integers or all strings
 * (see `IDArray`), which is only checked when building.
 */
export class ShapesGeometryBuilder {
  private readonly _shapePolygonOffsets: number[] = [0];
  private readonly _polygonRingOffsets: number[] = [0];
  private readonly _ringVertexOffsets: number[] = [0];
  private readonly _coords: number[] = [];
  private readonly _ids: (number | string)[] = [];
  private readonly _names: string[] = [];
  private _built = false;

  /**
   * Adds one shape
   *
   * Polygons without a valid shell are skipped, as are rings with fewer than
   * three vertices. A shape left without polygons is not added, and
   * contributes neither an ID nor a name.
   *
   * @param polygons - The polygons the shape is made of
   * @param id - The ID of the shape
   * @param name - The name of the shape, if any
   * @throws Error if the geometry has already been built
   */
  addShape(
    polygons: readonly ShapesPolygon[],
    id: number | string,
    name?: string,
  ): void {
    if (this._built) {
      throw new Error("Shapes cannot be added once the geometry is built.");
    }
    let polygonsAdded = false;
    for (const rings of polygons) {
      if (rings.length === 0 || rings[0]!.length < 3) {
        console.warn("Skipping polygon without a valid shell.");
        continue;
      }
      for (const ring of rings) {
        if (ring.length < 3) {
          console.warn("Skipping invalid ring with fewer than three vertices.");
          continue;
        }
        for (const position of ring) {
          this._coords.push(position[0]!, position[1]!);
        }
        this._ringVertexOffsets.push(this._coords.length / 2);
      }
      this._polygonRingOffsets.push(this._ringVertexOffsets.length - 1);
      polygonsAdded = true;
    }
    if (!polygonsAdded) {
      return;
    }
    this._shapePolygonOffsets.push(this._polygonRingOffsets.length - 1);
    this._ids.push(id);
    if (name !== undefined) {
      this._names.push(name);
    }
  }

  /** The number of shapes added so far */
  get size(): number {
    return this._ids.length;
  }

  /**
   * Builds the geometry of the shapes added so far
   *
   * The builder is spent afterwards, so that nothing can change a result it
   * has already handed out.
   *
   * @returns The shapes geometry, in typed arrays ready to be transferred,
   * along with the IDs and the names of the added shapes. Names are returned
   * only if every added shape was given one.
   * @throws Error if the geometry has already been built, or if the IDs mix
   * integers and strings (see `ArrayUtils.toIDArray`)
   */
  build(): {
    geometry: ShapesGeometry;
    ids: IDArray;
    names: string[] | undefined;
  } {
    if (this._built) {
      throw new Error("The geometry has already been built.");
    }
    this._built = true;
    return {
      geometry: {
        shapePolygonOffsets: new Uint32Array(this._shapePolygonOffsets),
        polygonRingOffsets: new Uint32Array(this._polygonRingOffsets),
        ringVertexOffsets: new Uint32Array(this._ringVertexOffsets),
        coords: new Float32Array(this._coords),
      },
      ids: ArrayUtils.toIDArray(this._ids),
      names: this._names.length === this._ids.length ? this._names : undefined,
    };
  }
}
