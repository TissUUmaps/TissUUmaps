import { afterEach, describe, expect, it, vi } from "vitest";

import { AsyncUtils, type Rect, type ShapesGeometry } from "@tissuumaps/core";

import { WebGLShapesRasterizer } from "./WebGLShapesRasterizer";

/** A closed ring, given as a list of `[x, y]` vertices (no repeated first/last) */
type Ring = [number, number][];
/** A polygon: shell ring first, then any hole rings */
type Polygon = Ring[];
/** A (multi-)polygon shape */
type Shape = Polygon[];

/**
 * Assembles a CSR-style {@link ShapesGeometry} from a nested
 * shapes -> polygons -> rings -> vertices description
 */
function createTestGeometry(shapes: Shape[]): ShapesGeometry {
  const shapePolygonOffsets: number[] = [0];
  const polygonRingOffsets: number[] = [0];
  const ringVertexOffsets: number[] = [0];
  const coords: number[] = [];
  for (const shape of shapes) {
    for (const polygon of shape) {
      for (const ring of polygon) {
        for (const [x, y] of ring) {
          coords.push(x, y);
        }
        ringVertexOffsets.push(coords.length / 2);
      }
      polygonRingOffsets.push(ringVertexOffsets.length - 1);
    }
    shapePolygonOffsets.push(polygonRingOffsets.length - 1);
  }
  return {
    shapePolygonOffsets: new Uint32Array(shapePolygonOffsets),
    polygonRingOffsets: new Uint32Array(polygonRingOffsets),
    ringVertexOffsets: new Uint32Array(ringVertexOffsets),
    coords: new Float32Array(coords),
  };
}

/** An axis-aligned square `[x0, x1] x [y0, y1]` as a single-ring polygon shape */
function createTestSquare(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
): Shape {
  return [
    [
      [
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
      ],
    ],
  ];
}

const unitBounds: Rect = { x: 0, y: 0, width: 1, height: 1 };

/**
 * Rasterizes shapes with {@link WebGLShapesRasterizer.rasterizeScanlines} and
 * decodes the packed buffer into the shapes and bins of every scanline, for
 * inspection
 *
 * The shapes of a scanline are the ones its bins reference, keyed by shape
 * index in the order they are laid out in. Also checks that these shapes and
 * their edges fill the buffer's scanline section exactly, i.e. that nothing
 * is written outside of them and nothing is left unreferenced.
 */
async function rasterizeAndDecode(
  numScanlines: number,
  numBins: number,
  scanlinePadding: number,
  binPadding: number,
  geometry: ShapesGeometry,
  shapesMask: Uint8Array | undefined,
  objectBounds: Rect,
  options?: { signal?: AbortSignal; shapeOrder?: Uint32Array },
) {
  const buffer = await WebGLShapesRasterizer.rasterizeScanlines(
    numScanlines,
    numBins,
    scanlinePadding,
    binPadding,
    geometry,
    shapesMask,
    objectBounds,
    options,
  );
  const uint32Buffer = new Uint32Array(buffer);
  const float32Buffer = new Float32Array(buffer);
  const scanlines: {
    bins: number[][];
    shapes: Map<
      number,
      {
        xMin: number;
        xMax: number;
        edges: { v0x: number; v0y: number; v1x: number; v1y: number }[];
      }
    >;
  }[] = [];
  let totalNumScanlineShapes = 0;
  let totalNumScanlineShapeEdges = 0;
  let totalNumScanlineBinShapes = 0;
  for (let s = 0; s < (buffer.byteLength > 0 ? numScanlines : 0); s++) {
    const bins: number[][] = [];
    const shapeTexelOffsets = new Set<number>();
    for (let b = 0; b < numBins; b++) {
      const binEntryValueOffset = 2 * (s * numBins + b);
      const shapeReferenceValueOffset = uint32Buffer[binEntryValueOffset]!;
      const numShapes = uint32Buffer[binEntryValueOffset + 1]!;
      const bin: number[] = [];
      for (let i = 0; i < numShapes; i++) {
        const shapeTexelOffset = uint32Buffer[shapeReferenceValueOffset + i]!;
        shapeTexelOffsets.add(shapeTexelOffset);
        bin.push(uint32Buffer[4 * shapeTexelOffset]!);
      }
      bins.push(bin);
      totalNumScanlineBinShapes += numShapes;
    }
    const shapes: (typeof scanlines)[number]["shapes"] = new Map();
    for (const shapeTexelOffset of [...shapeTexelOffsets].sort(
      (a, b) => a - b,
    )) {
      const shapeValueOffset = 4 * shapeTexelOffset;
      const numEdges = uint32Buffer[shapeValueOffset + 1]!;
      const edges = [];
      for (let e = 0; e < numEdges; e++) {
        const edgeValueOffset = 4 * (shapeTexelOffset + 1 + e);
        edges.push({
          v0x: float32Buffer[edgeValueOffset]!,
          v0y: float32Buffer[edgeValueOffset + 1]!,
          v1x: float32Buffer[edgeValueOffset + 2]!,
          v1y: float32Buffer[edgeValueOffset + 3]!,
        });
      }
      shapes.set(uint32Buffer[shapeValueOffset]!, {
        xMin: float32Buffer[shapeValueOffset + 2]!,
        xMax: float32Buffer[shapeValueOffset + 3]!,
        edges,
      });
      totalNumScanlineShapes++;
      totalNumScanlineShapeEdges += numEdges;
    }
    scanlines.push({ bins, shapes });
  }
  if (buffer.byteLength > 0) {
    const numBinTableTexels = Math.ceil((numScanlines * numBins) / 2);
    const numShapeReferenceTexels = Math.ceil(totalNumScanlineBinShapes / 4);
    expect(
      numBinTableTexels +
        numShapeReferenceTexels +
        totalNumScanlineShapes +
        totalNumScanlineShapeEdges,
    ).toBe(buffer.byteLength / 16);
  }
  return {
    scanlines,
    totalNumScanlineShapes,
    totalNumScanlineShapeEdges,
    totalNumScanlineBinShapes,
  };
}

describe("WebGLShapesRasterizer.computeGridSize", () => {
  it("sizes the scanlines to hold the given number of edges of a typical shape", async () => {
    // 4 edges over a height of 1 -> 0.25 per edge; 2 edges -> scanline height 0.5
    const geometry = createTestGeometry([createTestSquare(0, 0, 1, 1)]);

    await expect(
      WebGLShapesRasterizer.computeGridSize(
        geometry,
        undefined,
        { x: 0, y: 0, width: 20, height: 10 },
        2,
        1,
        0,
      ),
    ).resolves.toEqual({
      numScanlines: 20,
      numBins: 20,
      scanlinePadding: 0,
      binPadding: 0,
    });
  });

  it("pads by a fraction of the median shape height and width", async () => {
    // scanline height 0.5 and bin width 1 (see above) -> half the median
    // shape height (0.5) is one scanline, half its width (0.5) half a bin
    const geometry = createTestGeometry([createTestSquare(0, 0, 1, 1)]);

    await expect(
      WebGLShapesRasterizer.computeGridSize(
        geometry,
        undefined,
        { x: 0, y: 0, width: 20, height: 10 },
        2,
        1,
        0.5,
      ),
    ).resolves.toEqual({
      numScanlines: 20,
      numBins: 20,
      scanlinePadding: 1,
      binPadding: 0.5,
    });
  });

  it("does not scale the padding with the bin width factor", async () => {
    // bin width 2 -> half the median shape width (0.5) is a quarter bin
    const geometry = createTestGeometry([createTestSquare(0, 0, 1, 1)]);

    const { numBins, binPadding } = await WebGLShapesRasterizer.computeGridSize(
      geometry,
      undefined,
      { x: 0, y: 0, width: 20, height: 10 },
      2,
      2,
      0.5,
    );

    expect(numBins).toBe(10);
    expect(binPadding).toBe(0.25);
  });

  it("weights the heights per edge of the shapes by their area", async () => {
    // heights per edge 0.25 (area 1), 0.25 (area 1) and 1 (area 40) -> 1,
    // i.e. scanline height 4 (unweighted: 0.25, i.e. scanline height 1)
    const geometry = createTestGeometry([
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0, 0, 10, 4),
    ]);

    await expect(
      WebGLShapesRasterizer.computeGridSize(
        geometry,
        undefined,
        { x: 0, y: 0, width: 20, height: 12 },
        4,
        1,
        0,
      ),
    ).resolves.toEqual({
      numScanlines: 3,
      numBins: 20,
      scanlinePadding: 0,
      binPadding: 0,
    });
  });

  it("counts the edges of holes, but not zero-length edges", async () => {
    // 4 shell edges (one duplicated vertex) + 4 hole edges over a height of 4
    // -> 0.5 per edge
    const geometry = createTestGeometry([
      [
        [
          [
            [0, 0],
            [4, 0],
            [4, 0],
            [4, 4],
            [0, 4],
          ],
          [
            [1, 1],
            [3, 1],
            [3, 3],
            [1, 3],
          ],
        ],
      ],
    ]);

    await expect(
      WebGLShapesRasterizer.computeGridSize(
        geometry,
        undefined,
        { x: 0, y: 0, width: 20, height: 10 },
        1,
        1,
        0,
      ),
    ).resolves.toEqual({
      numScanlines: 20,
      numBins: 5,
      scanlinePadding: 0,
      binPadding: 0,
    });
  });

  it("leaves out shapes without edges when sizing the scanlines", async () => {
    const geometry = createTestGeometry([
      [
        [
          [
            [5, 5],
            [5, 5],
            [5, 5],
          ],
        ],
      ],
      createTestSquare(0, 0, 1, 1),
    ]);

    const { numScanlines } = await WebGLShapesRasterizer.computeGridSize(
      geometry,
      undefined,
      { x: 0, y: 0, width: 20, height: 10 },
      2,
      1,
      0,
    );

    expect(numScanlines).toBe(20);
  });

  it("sizes the bins like the median shape width, times the bin width factor", async () => {
    // widths 1, 2 and 10 -> median 2, times 2.5 -> bin width 5
    const geometry = createTestGeometry([
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0, 0, 2, 1),
      createTestSquare(0, 0, 10, 4),
    ]);

    const { numBins } = await WebGLShapesRasterizer.computeGridSize(
      geometry,
      undefined,
      { x: 0, y: 0, width: 20, height: 10 },
      4,
      2.5,
      0,
    );

    expect(numBins).toBe(4);
  });

  it("only considers the included shapes", async () => {
    // 5 x 5 square: 1.25 per edge, times 4 edges -> scanline height 5
    const geometry = createTestGeometry([
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0, 0, 5, 5),
    ]);

    await expect(
      WebGLShapesRasterizer.computeGridSize(
        geometry,
        new Uint8Array([0, 1]),
        { x: 0, y: 0, width: 20, height: 10 },
        4,
        1,
        0,
      ),
    ).resolves.toEqual({
      numScanlines: 2,
      numBins: 4,
      scanlinePadding: 0,
      binPadding: 0,
    });
  });

  it("uses at least one scanline and bin", async () => {
    const geometry = createTestGeometry([createTestSquare(0, 0, 100, 100)]);

    await expect(
      WebGLShapesRasterizer.computeGridSize(
        geometry,
        undefined,
        unitBounds,
        4,
        1,
        0,
      ),
    ).resolves.toEqual({
      numScanlines: 1,
      numBins: 1,
      scanlinePadding: 0,
      binPadding: 0,
    });
  });

  it("uses one scanline and bin without included shapes", async () => {
    const geometry = createTestGeometry([createTestSquare(0, 0, 1, 1)]);

    await expect(
      WebGLShapesRasterizer.computeGridSize(
        geometry,
        new Uint8Array([0]),
        unitBounds,
        4,
        1,
        0,
      ),
    ).resolves.toEqual({
      numScanlines: 1,
      numBins: 1,
      scanlinePadding: 0,
      binPadding: 0,
    });
  });

  it("limits the number of scanlines and of bins per scanline", async () => {
    // zero-width shapes -> as many bins as allowed (2 edges over a height of 1,
    // unweighted without area -> scanline height 1)
    const zeroWidthGeometry = createTestGeometry([
      createTestSquare(0, 0, 0, 1),
    ]);
    // zero-height shapes -> as many scanlines as allowed
    const zeroHeightGeometry = createTestGeometry([
      createTestSquare(0, 0, 1, 0),
    ]);

    await expect(
      WebGLShapesRasterizer.computeGridSize(
        zeroWidthGeometry,
        undefined,
        unitBounds,
        2,
        1,
        0,
      ),
    ).resolves.toEqual({
      numScanlines: 1,
      numBins: 4096,
      scanlinePadding: 0,
      binPadding: 0,
    });
    await expect(
      WebGLShapesRasterizer.computeGridSize(
        zeroHeightGeometry,
        undefined,
        unitBounds,
        2,
        1,
        0,
      ),
    ).resolves.toEqual({
      numScanlines: 4096,
      numBins: 1,
      scanlinePadding: 0,
      binPadding: 0,
    });
  });

  it("scales both down together to limit the total number of bins", async () => {
    // 4096 x 4096 bins before scaling down to at most 2^21 in total
    const geometry = createTestGeometry([createTestSquare(0, 0, 1, 1)]);

    const { numScanlines, numBins } =
      await WebGLShapesRasterizer.computeGridSize(
        geometry,
        undefined,
        { x: 0, y: 0, width: 10000, height: 10000 },
        4,
        1,
        0,
      );

    expect(numScanlines).toBe(numBins);
    expect(numScanlines * numBins).toBeLessThanOrEqual(2 ** 21);
    expect((numScanlines + 1) * (numBins + 1)).toBeGreaterThan(2 ** 21);
  });

  it("rejects when the abort signal is already aborted", async () => {
    const geometry = createTestGeometry([createTestSquare(0, 0, 1, 1)]);
    const controller = new AbortController();
    controller.abort();

    await expect(
      WebGLShapesRasterizer.computeGridSize(
        geometry,
        undefined,
        unitBounds,
        4,
        1,
        0,
        { signal: controller.signal },
      ),
    ).rejects.toThrow();
  });
});

describe("WebGLShapesRasterizer.rasterizeScanlines", () => {
  it("rasterizes a single full-extent square into one scanline", async () => {
    const geometry = createTestGeometry([createTestSquare(0, 0, 1, 1)]);

    const {
      scanlines,
      totalNumScanlineShapes,
      totalNumScanlineShapeEdges,
      totalNumScanlineBinShapes,
    } = await rasterizeAndDecode(
      1,
      1,
      0.5,
      0.5,
      geometry,
      undefined,
      unitBounds,
    );

    expect(scanlines).toHaveLength(1);
    expect(totalNumScanlineShapes).toBe(1);
    expect(totalNumScanlineShapeEdges).toBe(4);
    expect(totalNumScanlineBinShapes).toBe(1);

    const scanline = scanlines[0]!;
    expect(scanline.bins).toEqual([[0]]);

    expect([...scanline.shapes.keys()]).toEqual([0]);
    const scanlineShape = scanline.shapes.get(0)!;
    expect(scanlineShape.xMin).toBe(0);
    expect(scanlineShape.xMax).toBe(1);
    expect(scanlineShape.edges).toEqual([
      { v0x: 0, v0y: 0, v1x: 1, v1y: 0 },
      { v0x: 1, v0y: 0, v1x: 1, v1y: 1 },
      { v0x: 1, v0y: 1, v1x: 0, v1y: 1 },
      { v0x: 0, v0y: 1, v1x: 0, v1y: 0 },
    ]);
  });

  it("drops the edges of a hole that reaches beyond the scanlines of its shell", async () => {
    // with 8 scanlines: shape 0 covers scanlines 0-2 (with slack), the shell of
    // shape 1 scanlines 5-7, but the (invalid) hole of shape 1 reaches down to
    // scanline 0
    const geometry = createTestGeometry([
      createTestSquare(0, 0, 1, 0.25),
      [
        [
          [
            [0, 0.75],
            [1, 0.75],
            [1, 1],
            [0, 1],
          ],
          [
            [0.25, 0.1],
            [0.25, 0.9],
            [0.75, 0.9],
            [0.75, 0.1],
          ],
        ],
      ],
    ]);

    const { scanlines } = await rasterizeAndDecode(
      8,
      1,
      0.5,
      0.5,
      geometry,
      undefined,
      unitBounds,
    );

    expect([...scanlines[1]!.shapes.keys()]).toEqual([0]);
    expect(scanlines[4]!.shapes.size).toBe(0);
    expect([...scanlines[6]!.shapes.keys()]).toEqual([1]);
  });

  it("distributes a shape and its edges across multiple scanlines", async () => {
    const geometry = createTestGeometry([createTestSquare(0, 0, 1, 1)]);

    const { scanlines, totalNumScanlineShapes, totalNumScanlineShapeEdges } =
      await rasterizeAndDecode(4, 1, 0.5, 0.5, geometry, undefined, unitBounds);

    expect(scanlines).toHaveLength(4);
    // the shape spans all scanlines -> one shape entry per scanline
    expect(totalNumScanlineShapes).toBe(4);
    // vertical edges straddle all scanlines, horizontal edges only their own
    // (the adjacent ones are more than half a scanline away)
    expect(totalNumScanlineShapeEdges).toBe(3 + 2 + 2 + 3);

    expect(scanlines[0]!.shapes.get(0)!.edges).toHaveLength(3);
    expect(scanlines[1]!.shapes.get(0)!.edges).toHaveLength(2);
    expect(scanlines[2]!.shapes.get(0)!.edges).toHaveLength(2);
    expect(scanlines[3]!.shapes.get(0)!.edges).toHaveLength(3);
  });

  it("leaves scanlines outside a shape's vertical extent empty", async () => {
    // square occupying only the top third of a 3-unit-tall object
    const geometry = createTestGeometry([createTestSquare(0, 2, 1, 3)]);

    const { scanlines } = await rasterizeAndDecode(
      3,
      1,
      0.5,
      0.5,
      geometry,
      undefined,
      { x: 0, y: 0, width: 1, height: 3 },
    );

    const emptyScanline = scanlines[0]!;
    expect(emptyScanline.shapes.size).toBe(0);
    expect(emptyScanline.bins).toEqual([[]]);

    const populatedScanline = scanlines[2]!;
    expect(populatedScanline.shapes.has(0)).toBe(true);
  });

  it("pads shapes and their edges by half a scanline, as slack for strokes", async () => {
    // on 8 scanlines of height 1: the shape spans scanlines 3-4, and reaches
    // within half a scanline of scanlines 2 and 5, but not of scanlines 1 and 6
    const geometry = createTestGeometry([
      createTestSquare(0, 3.125, 0.5, 4.875),
    ]);

    const { scanlines, totalNumScanlineShapes, totalNumScanlineShapeEdges } =
      await rasterizeAndDecode(8, 1, 0.5, 0.5, geometry, undefined, {
        x: 0,
        y: 0,
        width: 1,
        height: 8,
      });

    expect(scanlines[1]!.shapes.size).toBe(0);
    expect([...scanlines[2]!.shapes.keys()]).toEqual([0]);
    expect([...scanlines[5]!.shapes.keys()]).toEqual([0]);
    expect(scanlines[6]!.shapes.size).toBe(0);
    expect(totalNumScanlineShapes).toBe(4);
    // the bottom edge (y = 3.125) is padded onto scanline 2, but not 4, the top
    // edge (y = 4.875) onto scanline 5, but not 3
    expect(scanlines[2]!.shapes.get(0)!.edges).toEqual([
      { v0x: 0, v0y: 3.125, v1x: 0.5, v1y: 3.125 },
      { v0x: 0.5, v0y: 3.125, v1x: 0.5, v1y: 4.875 },
      { v0x: 0, v0y: 4.875, v1x: 0, v1y: 3.125 },
    ]);
    expect(scanlines[5]!.shapes.get(0)!.edges).toEqual([
      { v0x: 0.5, v0y: 3.125, v1x: 0.5, v1y: 4.875 },
      { v0x: 0.5, v0y: 4.875, v1x: 0, v1y: 4.875 },
      { v0x: 0, v0y: 4.875, v1x: 0, v1y: 3.125 },
    ]);
    expect(scanlines[3]!.shapes.get(0)!.edges).toHaveLength(3);
    expect(scanlines[4]!.shapes.get(0)!.edges).toHaveLength(3);
    expect(totalNumScanlineShapeEdges).toBe(3 + 3 + 3 + 3);
  });

  it("respects the shape mask and compacts the shape index", async () => {
    const geometry = createTestGeometry([
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0, 0, 1, 1),
    ]);

    const { scanlines, totalNumScanlineShapes } = await rasterizeAndDecode(
      1,
      1,
      0.5,
      0.5,
      geometry,
      new Uint8Array([0, 1, 0]),
      unitBounds,
    );

    expect(totalNumScanlineShapes).toBe(1);
    // second shape (masked in) is compacted to index 0
    expect([...scanlines[0]!.shapes.keys()]).toEqual([0]);
  });

  it("assigns ascending compacted indices to multiple included shapes", async () => {
    const geometry = createTestGeometry([
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0, 0, 1, 1),
    ]);

    const { scanlines, totalNumScanlineShapes } = await rasterizeAndDecode(
      1,
      1,
      0.5,
      0.5,
      geometry,
      new Uint8Array([1, 0, 1]),
      unitBounds,
    );

    expect(totalNumScanlineShapes).toBe(2);
    expect([...scanlines[0]!.shapes.keys()]).toEqual([0, 1]);
  });

  it("ignores zero-length edges", async () => {
    // triangle with a repeated vertex introducing a zero-length edge
    const geometry = createTestGeometry([
      [
        [
          [
            [0, 0],
            [1, 0],
            [1, 0], // duplicate -> zero-length edge
            [0, 1],
          ],
        ],
      ],
    ]);

    const { totalNumScanlineShapeEdges, scanlines } = await rasterizeAndDecode(
      1,
      1,
      0.5,
      0.5,
      geometry,
      undefined,
      unitBounds,
    );

    // 4 ring segments, but the zero-length one is dropped
    expect(totalNumScanlineShapeEdges).toBe(3);
    expect(scanlines[0]!.shapes.get(0)!.edges).toHaveLength(3);
  });

  it("includes hole edges but bounds the shape by its shell only", async () => {
    // 4x4 square shell with a 2x2 square hole
    const geometry = createTestGeometry([
      [
        [
          [
            [0, 0],
            [4, 0],
            [4, 4],
            [0, 4],
          ],
          [
            [1, 1],
            [3, 1],
            [3, 3],
            [1, 3],
          ],
        ],
      ],
    ]);

    const { scanlines, totalNumScanlineShapes, totalNumScanlineShapeEdges } =
      await rasterizeAndDecode(1, 1, 0.5, 0.5, geometry, undefined, {
        x: 0,
        y: 0,
        width: 4,
        height: 4,
      });

    expect(totalNumScanlineShapes).toBe(1);
    // shell (4 edges) + hole (4 edges)
    expect(totalNumScanlineShapeEdges).toBe(8);

    const scanlineShape = scanlines[0]!.shapes.get(0)!;
    expect(scanlineShape.edges).toHaveLength(8);
    // bounds come from the shell, ignoring the hole
    expect(scanlineShape.xMin).toBe(0);
    expect(scanlineShape.xMax).toBe(4);
  });

  it("merges polygons of a multi-polygon shape under one shape index", async () => {
    const geometry = createTestGeometry([
      [createTestSquare(0, 0, 1, 1)[0]!, createTestSquare(2, 0, 3, 1)[0]!],
    ]);

    const { scanlines, totalNumScanlineShapes, totalNumScanlineShapeEdges } =
      await rasterizeAndDecode(1, 1, 0.5, 0.5, geometry, undefined, {
        x: 0,
        y: 0,
        width: 3,
        height: 1,
      });

    expect(totalNumScanlineShapes).toBe(1);
    expect(totalNumScanlineShapeEdges).toBe(8);

    const scanlineShape = scanlines[0]!.shapes.get(0)!;
    // bounds span both polygons
    expect(scanlineShape.xMin).toBe(0);
    expect(scanlineShape.xMax).toBe(3);
    expect(scanlineShape.edges).toHaveLength(8);
  });

  it("lists a shape in the bins its x-range spans", async () => {
    // x-range [0.4, 0.6] with 8 bins -> bins [3, 4], padded to [2, 5]
    const geometry = createTestGeometry([createTestSquare(0.4, 0, 0.6, 1)]);

    const { scanlines } = await rasterizeAndDecode(
      1,
      8,
      0.5,
      0.5,
      geometry,
      undefined,
      unitBounds,
    );

    expect(scanlines[0]!.bins).toEqual([[], [], [0], [0], [0], [0], [], []]);
  });

  it("pads the bins by half a bin on either side, as slack for strokes", async () => {
    // x-range [0.32, 0.34] with 8 bins -> bin 2, padded to [0.2575, 0.4025] ->
    // bins [2, 3] (half a bin reaches into bin 3, but not into bin 1)
    const geometry = createTestGeometry([createTestSquare(0.32, 0, 0.34, 1)]);

    const { scanlines } = await rasterizeAndDecode(
      1,
      8,
      0.5,
      0.5,
      geometry,
      undefined,
      unitBounds,
    );

    expect(scanlines[0]!.bins).toEqual([[], [], [0], [0], [], [], [], []]);
  });

  it("pads the scanlines and bins by their own paddings", async () => {
    // y- and x-range [0.32, 0.34] with 8 scanlines and bins -> scanline and bin
    // 2 unpadded, and [0.195, 0.465] -> [1, 3] padded by one scanline or bin
    const geometry = createTestGeometry([
      createTestSquare(0.32, 0.32, 0.34, 0.34),
    ]);
    const createPaddedScanlines = (
      scanlinePadding: number,
      binPadding: number,
    ) =>
      rasterizeAndDecode(
        8,
        8,
        scanlinePadding,
        binPadding,
        geometry,
        undefined,
        unitBounds,
      );
    const unpadded = await createPaddedScanlines(0, 0);
    const scanlinePadded = await createPaddedScanlines(1, 0);
    const binPadded = await createPaddedScanlines(0, 1);

    expect(unpadded.scanlines.map((scanline) => scanline.shapes.size)).toEqual([
      0, 0, 1, 0, 0, 0, 0, 0,
    ]);
    expect(unpadded.scanlines[2]!.bins).toEqual([
      [],
      [],
      [0],
      [],
      [],
      [],
      [],
      [],
    ]);
    expect(
      scanlinePadded.scanlines.map((scanline) => scanline.shapes.size),
    ).toEqual([0, 1, 1, 1, 0, 0, 0, 0]);
    expect(scanlinePadded.scanlines[2]!.bins).toEqual([
      [],
      [],
      [0],
      [],
      [],
      [],
      [],
      [],
    ]);
    expect(binPadded.scanlines.map((scanline) => scanline.shapes.size)).toEqual(
      [0, 0, 1, 0, 0, 0, 0, 0],
    );
    expect(binPadded.scanlines[2]!.bins).toEqual([
      [],
      [0],
      [0],
      [0],
      [],
      [],
      [],
      [],
    ]);
  });

  it("lists the included shapes per bin in ascending compacted order", async () => {
    // shape 1 is masked out, shape 2 becomes compacted index 1
    const geometry = createTestGeometry([
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0.75, 0, 1, 1),
    ]);

    const { scanlines } = await rasterizeAndDecode(
      1,
      4,
      0.5,
      0.5,
      geometry,
      new Uint8Array([1, 0, 1]),
      unitBounds,
    );

    // shape 2 spans bin 3, padded to bins [2, 3]
    expect(scanlines[0]!.bins).toEqual([[0], [0], [0, 1], [0, 1]]);
  });

  it("lists the shapes per bin in the given order", async () => {
    const geometry = createTestGeometry([
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0, 0, 1, 1),
    ]);

    const { scanlines } = await rasterizeAndDecode(
      1,
      1,
      0.5,
      0.5,
      geometry,
      undefined,
      unitBounds,
      { shapeOrder: new Uint32Array([2, 0, 1]) },
    );

    expect(scanlines[0]!.bins).toEqual([[2, 0, 1]]);
  });

  it("orders the included shapes by their compacted index", async () => {
    // shape 1 is masked out, shape 3 becomes compacted index 2
    const geometry = createTestGeometry([
      createTestSquare(0, 0, 0.25, 1),
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0, 0, 0.5, 1),
      createTestSquare(0, 0, 0.75, 1),
    ]);

    const { scanlines } = await rasterizeAndDecode(
      1,
      1,
      0.5,
      0.5,
      geometry,
      new Uint8Array([1, 0, 1, 1]),
      unitBounds,
      { shapeOrder: new Uint32Array([2, 0, 1]) },
    );

    expect(scanlines[0]!.bins).toEqual([[2, 0, 1]]);
    expect(scanlines[0]!.shapes.get(2)!.xMax).toBe(0.75);
  });

  it("rejects a shape order that does not hold one index per included shape", async () => {
    const geometry = createTestGeometry([
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0, 0, 1, 1),
    ]);

    await expect(
      rasterizeAndDecode(
        1,
        1,
        0.5,
        0.5,
        geometry,
        new Uint8Array([1, 0]),
        unitBounds,
        { shapeOrder: new Uint32Array([0, 1]) },
      ),
    ).rejects.toThrow("expected one per included shape (1)");
  });

  it("bins a multi-polygon shape by its x-range on each scanline", async () => {
    // with 8 scanlines and 8 bins: the lower-left polygon covers scanlines 0-2
    // and bins 0-2, the upper-right one scanlines 5-7 and bins 5-7 (with slack)
    const geometry = createTestGeometry([
      [
        ...createTestSquare(0, 0, 0.25, 0.25),
        ...createTestSquare(0.75, 0.75, 1, 1),
      ],
    ]);

    const { scanlines, totalNumScanlineBinShapes } = await rasterizeAndDecode(
      8,
      8,
      0.5,
      0.5,
      geometry,
      undefined,
      unitBounds,
    );

    expect(scanlines.map((scanline) => scanline.bins)).toEqual([
      [[0], [0], [0], [], [], [], [], []],
      [[0], [0], [0], [], [], [], [], []],
      [[0], [0], [0], [], [], [], [], []],
      [[], [], [], [], [], [], [], []],
      [[], [], [], [], [], [], [], []],
      [[], [], [], [], [], [0], [0], [0]],
      [[], [], [], [], [], [0], [0], [0]],
      [[], [], [], [], [], [0], [0], [0]],
    ]);
    expect(totalNumScanlineBinShapes).toBe(3 * 3 + 3 * 3);
  });

  it("returns an empty result when there are no bins", async () => {
    const geometry = createTestGeometry([createTestSquare(0, 0, 1, 1)]);

    const {
      scanlines,
      totalNumScanlineShapes,
      totalNumScanlineShapeEdges,
      totalNumScanlineBinShapes,
    } = await rasterizeAndDecode(
      1,
      0,
      0.5,
      0.5,
      geometry,
      undefined,
      unitBounds,
    );

    expect(scanlines).toEqual([]);
    expect(totalNumScanlineShapes).toBe(0);
    expect(totalNumScanlineShapeEdges).toBe(0);
    expect(totalNumScanlineBinShapes).toBe(0);
  });

  it("returns an empty result when there are no scanlines", async () => {
    const geometry = createTestGeometry([createTestSquare(0, 0, 1, 1)]);

    const {
      scanlines,
      totalNumScanlineShapes,
      totalNumScanlineShapeEdges,
      totalNumScanlineBinShapes,
    } = await rasterizeAndDecode(
      0,
      1,
      0.5,
      0.5,
      geometry,
      undefined,
      unitBounds,
    );

    expect(scanlines).toEqual([]);
    expect(totalNumScanlineShapes).toBe(0);
    expect(totalNumScanlineShapeEdges).toBe(0);
    expect(totalNumScanlineBinShapes).toBe(0);
  });

  it("rejects object bounds without a positive width or height", async () => {
    const geometry = createTestGeometry([createTestSquare(0, 0, 1, 1)]);

    await expect(
      rasterizeAndDecode(4, 1, 0.5, 0.5, geometry, undefined, {
        x: 0,
        y: 0,
        width: 1,
        height: 0,
      }),
    ).rejects.toThrow("positive width and height");
  });

  it("returns empty scanlines for geometry without shapes", async () => {
    const geometry = createTestGeometry([]);

    const { scanlines, totalNumScanlineShapes, totalNumScanlineShapeEdges } =
      await rasterizeAndDecode(3, 1, 0.5, 0.5, geometry, undefined, unitBounds);

    expect(scanlines).toHaveLength(3);
    expect(totalNumScanlineShapes).toBe(0);
    expect(totalNumScanlineShapeEdges).toBe(0);
    for (const scanline of scanlines) {
      expect(scanline.shapes.size).toBe(0);
      expect(scanline.bins).toEqual([[]]);
    }
  });

  it("returns empty scanlines when every shape is masked out", async () => {
    const geometry = createTestGeometry([
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0, 0, 1, 1),
    ]);

    const { scanlines, totalNumScanlineShapes, totalNumScanlineShapeEdges } =
      await rasterizeAndDecode(
        2,
        1,
        0.5,
        0.5,
        geometry,
        new Uint8Array([0, 0]),
        unitBounds,
      );

    expect(totalNumScanlineShapes).toBe(0);
    expect(totalNumScanlineShapeEdges).toBe(0);
    for (const scanline of scanlines) {
      expect(scanline.shapes.size).toBe(0);
      expect(scanline.bins).toEqual([[]]);
    }
  });

  it("rejects when the abort signal is already aborted", async () => {
    const geometry = createTestGeometry([createTestSquare(0, 0, 1, 1)]);
    const controller = new AbortController();
    controller.abort();

    await expect(
      rasterizeAndDecode(1, 1, 0.5, 0.5, geometry, undefined, unitBounds, {
        signal: controller.signal,
      }),
    ).rejects.toThrow();
  });

  describe("yielding", () => {
    afterEach(() => {
      vi.restoreAllMocks();
    });

    it("passes the signal to the yielder once per shape and pass", async () => {
      const geometry = createTestGeometry([
        createTestSquare(0, 0, 1, 1),
        createTestSquare(0, 0, 1, 1),
      ]);
      const controller = new AbortController();
      const maybeYield = vi.fn(() => Promise.resolve());
      vi.spyOn(AsyncUtils, "createYielder").mockReturnValue(maybeYield);

      await rasterizeAndDecode(
        1,
        1,
        0.5,
        0.5,
        geometry,
        undefined,
        unitBounds,
        { signal: controller.signal },
      );

      expect(maybeYield).toHaveBeenCalledTimes(2 * 2);
      expect(maybeYield).toHaveBeenCalledWith({ signal: controller.signal });
    });

    it("yields for masked-out shapes in the first pass only", async () => {
      const geometry = createTestGeometry([
        createTestSquare(0, 0, 1, 1),
        createTestSquare(0, 0, 1, 1),
      ]);
      const maybeYield = vi.fn(() => Promise.resolve());
      vi.spyOn(AsyncUtils, "createYielder").mockReturnValue(maybeYield);

      await rasterizeAndDecode(
        1,
        1,
        0.5,
        0.5,
        geometry,
        new Uint8Array([0, 0]),
        unitBounds,
      );

      expect(maybeYield).toHaveBeenCalledTimes(2);
    });

    it("rejects when the signal is aborted mid-iteration", async () => {
      const geometry = createTestGeometry([
        createTestSquare(0, 0, 1, 1),
        createTestSquare(0, 0, 1, 1),
      ]);
      const controller = new AbortController();
      // Abort on the first yield so the second one observes the abort
      const maybeYield = vi.fn((opts?: { signal?: AbortSignal }) => {
        opts?.signal?.throwIfAborted();
        controller.abort();
        return Promise.resolve();
      });
      vi.spyOn(AsyncUtils, "createYielder").mockReturnValue(maybeYield);

      await expect(
        rasterizeAndDecode(1, 1, 0.5, 0.5, geometry, undefined, unitBounds, {
          signal: controller.signal,
        }),
      ).rejects.toThrow();
      expect(maybeYield).toHaveBeenCalledTimes(2);
    });
  });
});

describe("WebGLShapesRasterizer.rasterizeScanlines layout", () => {
  it("packs the bin table, shape references, shape header, and edges of a single shape", async () => {
    const geometry = createTestGeometry([createTestSquare(0, 0, 1, 1)]);

    const buffer = await WebGLShapesRasterizer.rasterizeScanlines(
      1,
      1,
      0.5,
      0.5,
      geometry,
      undefined,
      unitBounds,
    );

    // bin table (4) + shape references (4) + shape header (4) + 4 edges (16) = 28 values
    expect(buffer.byteLength).toBe(28 * 4);
    const uint32Buffer = new Uint32Array(buffer);
    const float32Buffer = new Float32Array(buffer);

    // bin table (texel 0): one bin entry of shape reference value offset and shape count
    expect([...uint32Buffer.slice(0, 2)]).toEqual([4, 1]);

    // shape references (texel 1): the shape header's texel offset
    expect(uint32Buffer[4]).toBe(2);

    // shape header (texel 2): shape index, edge count, xMin, xMax
    expect(uint32Buffer[8]).toBe(0);
    expect(uint32Buffer[9]).toBe(4);
    expect(float32Buffer[10]).toBe(0);
    expect(float32Buffer[11]).toBe(1);

    // edges (texels 3..6)
    expect([...float32Buffer.slice(12, 28)]).toEqual([
      0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0,
    ]);
  });

  it("aligns the buffer size to the requested multiple", async () => {
    const geometry = createTestGeometry([createTestSquare(0, 0, 1, 1)]);

    const buffer = await WebGLShapesRasterizer.rasterizeScanlines(
      1,
      1,
      0.5,
      0.5,
      geometry,
      undefined,
      unitBounds,
      { align: 16 },
    );

    // 28 values aligned up to the next multiple of 16 -> 32 values
    expect(buffer.byteLength).toBe(32 * 4);
    // padding is left zero-initialized
    expect([...new Uint32Array(buffer).slice(28, 32)]).toEqual([0, 0, 0, 0]);
  });

  it("writes a bin entry per scanline and bin, in scanline-major order", async () => {
    // square in the top third and left fifth, so scanline 0 is empty,
    // scanlines 1 (stroke padding) and 2 are populated, and only in bins 0-1
    const geometry = createTestGeometry([createTestSquare(0, 2, 0.2, 3)]);

    const buffer = await WebGLShapesRasterizer.rasterizeScanlines(
      3,
      4,
      0.5,
      0.5,
      geometry,
      undefined,
      {
        x: 0,
        y: 0,
        width: 1,
        height: 3,
      },
    );
    const uint32Buffer = new Uint32Array(buffer);

    // bin table (texels 0..5): shape reference value offset and shape count
    // per bin; empty bins point to where the next bin's references start
    expect([...uint32Buffer.slice(0, 24)]).toEqual([
      ...[24, 0, 24, 0, 24, 0, 24, 0], // scanline 0
      ...[24, 1, 25, 1, 26, 0, 26, 0], // scanline 1
      ...[26, 1, 27, 1, 28, 0, 28, 0], // scanline 2
    ]);

    // shape references (texel 6): scanline 1's shape at texel 7 (followed by
    // its 3 edges), scanline 2's shape at texel 11
    expect([...uint32Buffer.slice(24, 28)]).toEqual([7, 7, 11, 11]);
    expect(uint32Buffer[4 * 7 + 1]).toBe(3);
    expect(uint32Buffer[4 * 11 + 1]).toBe(4);
  });

  it("references the shapes of each bin in ascending order, sharing their data across bins", async () => {
    // shape 0 spans all 4 bins, shape 1 bin 0, padded to bins [0, 1]
    const geometry = createTestGeometry([
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0, 0, 0.125, 0.5),
    ]);

    const buffer = await WebGLShapesRasterizer.rasterizeScanlines(
      1,
      4,
      0.5,
      0.5,
      geometry,
      undefined,
      unitBounds,
    );
    const uint32Buffer = new Uint32Array(buffer);
    const float32Buffer = new Float32Array(buffer);

    // bin table (texels 0..1): bins 0-1 have two shape references, bins 2-3 one
    expect([...uint32Buffer.slice(0, 8)]).toEqual([8, 2, 10, 2, 12, 1, 13, 1]);

    // shape references (texels 2..3): bins 0-1 -> shapes 0 and 1, bins 2-3 -> shape 0
    expect([...uint32Buffer.slice(8, 14)]).toEqual([4, 9, 4, 9, 4, 4]);

    // first shape header (texel 4), its 4 edges occupy texels 5..8
    expect(uint32Buffer[16]).toBe(0);
    expect(uint32Buffer[17]).toBe(4);
    expect(float32Buffer[18]).toBe(0);
    expect(float32Buffer[19]).toBe(1);

    // second shape header follows at texel 9, bounded by the smaller square
    expect(uint32Buffer[36]).toBe(1);
    expect(uint32Buffer[37]).toBe(4);
    expect(float32Buffer[38]).toBe(0);
    expect(float32Buffer[39]).toBe(0.125);
  });

  it("sizes the buffer to match the rasterized totals", async () => {
    const geometry = createTestGeometry([
      createTestSquare(0, 0, 1, 1),
      createTestSquare(0, 0, 0.5, 0.5),
    ]);

    const buffer = await WebGLShapesRasterizer.rasterizeScanlines(
      3,
      3,
      0.5,
      0.5,
      geometry,
      undefined,
      unitBounds,
    );
    const {
      scanlines,
      totalNumScanlineShapes,
      totalNumScanlineShapeEdges,
      totalNumScanlineBinShapes,
    } = await rasterizeAndDecode(
      3,
      3,
      0.5,
      0.5,
      geometry,
      undefined,
      unitBounds,
    );

    // two bin entries and four shape references per texel (4 values each),
    // plus one texel per shape entry and per edge entry
    const expectedValues =
      4 * Math.ceil((scanlines.length * 3) / 2) +
      4 * Math.ceil(totalNumScanlineBinShapes / 4) +
      4 * totalNumScanlineShapes +
      4 * totalNumScanlineShapeEdges;
    expect(buffer.byteLength).toBe(expectedValues * 4);
  });
});
