import {
  AsyncUtils,
  MathUtils,
  type Rect,
  type ShapesGeometry,
} from "@tissuumaps/core";

/**
 * Rasterizes shapes into the scanline data consumed by the shapes shaders
 *
 * Shapes are not triangulated. Instead, the fragment shader determines per
 * fragment which shape it falls into, by computing a winding number over the
 * polygon edges near that fragment and applying the even-odd rule, so that
 * holes are cut out whatever the orientation of their rings. To keep that
 * affordable, the object's bounding box is divided into horizontal scanlines,
 * each of which holds only the shapes and edges that reach into it, and each
 * scanline into x-bins, each of which lists only the shapes that reach into
 * it: {@link rasterizeScanlines} rasterizes them straight into the layout the
 * GPU expects.
 *
 * All coordinates are in data coordinates, as is the fragment shader.
 */
export class WebGLShapesRasterizer {
  /**
   * Maximum number of scanlines, and of bins per scanline
   *
   * Keeps both numbers finite before {@link computeGridSize} scales them down
   * to {@link _maxNumGridCells} in total, as shapes without height or width
   * (a median height per edge or width of zero) would otherwise ask for
   * infinitely many scanlines or bins, and scaling those down yields `NaN`.
   * Neither the data layout nor the shader requires this limit, and the memory
   * is bounded by the total. As a side effect, objects of extreme aspect ratio
   * (e.g. a long strip of small shapes) get cells larger than intended along
   * their long axis, even where the total would allow more.
   */
  private static readonly _maxGridSize = 4096;

  /**
   * Maximum number of bins across all scanlines
   *
   * Bounds the memory spent on the grid itself, whatever the shapes: the bin
   * table holds two bins per RGBA32UI texel, and {@link rasterizeScanlines}
   * keeps a 32-bit counter per bin. This is a memory budget, not a limit of
   * the data layout or the shader.
   */
  private static readonly _maxNumGridCells = 2 ** 21;

  /**
   * Computes the numbers of scanlines and of bins per scanline for a shapes
   * object from the sizes and edge counts of its shapes, and the paddings of
   * the shapes and edges in them
   *
   * The two are chosen independently, as they bound different costs of a
   * fragment (see `WebGLShapesRenderOptions`).
   *
   * The scanline height is chosen so that a scanline holds about
   * `edgesPerScanline` edges of a typical shape. A shape's edges are spread
   * over its height, so a scanline of a given height holds about as many of
   * them as it takes the shape's height per edge to fill it. The typical
   * height per edge is the median over the shapes, weighted by the area of the
   * bounding boxes of their polygon shells, as fragments, and thus the cost of
   * testing edges, are distributed by area: a few large, detailed shapes among
   * many small ones cover most fragments when zoomed in. Shapes without edges
   * are left out, and if none of the others has an area, the median is
   * unweighted instead.
   *
   * The bin width is the median width of the bounding boxes of the shapes'
   * polygon shells, times `binWidthFactor`.
   *
   * Both numbers are clamped to {@link _maxGridSize}, and scaled down together
   * to {@link _maxNumGridCells} bins in total.
   *
   * The paddings are `shapePadding` times the median height and width of the
   * shapes, returned as the fractions of a scanline and of a bin that
   * {@link rasterizeScanlines} expects.
   *
   * Yields between shapes, i.e. long-running computations neither block the
   * event loop nor ignore an abort for long.
   *
   * @param geometry - Geometry for all shapes in the object
   * @param shapesMask - Per-shape inclusion mask, or `undefined` if all shapes are included
   * @param objectBounds - Bounding box of all shapes
   * @param edgesPerScanline - See `WebGLShapesRenderOptions.edgesPerScanline`
   * @param binWidthFactor - See `WebGLShapesRenderOptions.binWidthFactor`
   * @param shapePadding - See `WebGLShapesRenderOptions.shapePadding`
   * @param options - Optional abort signal
   * @returns The numbers of scanlines and of bins per scanline, or one of
   * each if no shape is included, and the paddings as fractions of a
   * scanline's height and of a bin's width (none if no shape is included)
   */
  static async computeGridSize(
    geometry: ShapesGeometry,
    shapesMask: Uint8Array | undefined,
    objectBounds: Rect,
    edgesPerScanline: number,
    binWidthFactor: number,
    shapePadding: number,
    options?: { signal?: AbortSignal },
  ): Promise<{
    numScanlines: number;
    numBins: number;
    scanlinePadding: number;
    binPadding: number;
  }> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const {
      shapePolygonOffsets,
      polygonRingOffsets,
      ringVertexOffsets,
      coords,
    } = geometry;
    const numShapes = shapePolygonOffsets.length - 1;
    const shapeWidths = new Float64Array(numShapes);
    const shapeHeights = new Float64Array(numShapes);
    const shapeHeightsPerEdge = new Float64Array(numShapes);
    const shapeAreas = new Float64Array(numShapes);
    let numIncludedShapes = 0;
    let numIncludedShapesWithEdges = 0;
    const maybeYield = AsyncUtils.createYielder();
    for (let s = 0; s < numShapes; s++) {
      if (shapesMask === undefined || shapesMask[s]! > 0) {
        let xMin = Infinity,
          yMin = Infinity,
          xMax = -Infinity,
          yMax = -Infinity;
        let numEdges = 0;
        for (
          let p = shapePolygonOffsets[s]!;
          p < shapePolygonOffsets[s + 1]!;
          p++
        ) {
          const shellRing = polygonRingOffsets[p]!;
          const shellVertexStart = ringVertexOffsets[shellRing]!;
          const shellVertexEnd = ringVertexOffsets[shellRing + 1]!;
          for (let v = shellVertexStart; v < shellVertexEnd; v++) {
            const x = coords[2 * v]!;
            const y = coords[2 * v + 1]!;
            xMin = Math.min(xMin, x);
            yMin = Math.min(yMin, y);
            xMax = Math.max(xMax, x);
            yMax = Math.max(yMax, y);
          }
          // count the edges of all rings, as rasterizeScanlines does
          for (let r = shellRing; r < polygonRingOffsets[p + 1]!; r++) {
            const ringVertexStart = ringVertexOffsets[r]!;
            const ringVertexEnd = ringVertexOffsets[r + 1]!;
            const nRingVertices = ringVertexEnd - ringVertexStart;
            for (let i = 0; i < nRingVertices; ++i) {
              const v0 = ringVertexStart + i;
              const v1 = ringVertexStart + ((i + 1) % nRingVertices);
              if (
                coords[2 * v0] !== coords[2 * v1] ||
                coords[2 * v0 + 1] !== coords[2 * v1 + 1]
              ) {
                numEdges++; // zero-length edges are ignored
              }
            }
          }
        }
        if (xMin <= xMax && yMin <= yMax) {
          shapeWidths[numIncludedShapes] = xMax - xMin;
          shapeHeights[numIncludedShapes] = yMax - yMin;
          numIncludedShapes++;
          if (numEdges > 0) {
            shapeHeightsPerEdge[numIncludedShapesWithEdges] =
              (yMax - yMin) / numEdges;
            shapeAreas[numIncludedShapesWithEdges] =
              (xMax - xMin) * (yMax - yMin);
            numIncludedShapesWithEdges++;
          }
        }
      }
      await maybeYield({ signal });
    }
    if (numIncludedShapes === 0) {
      return { numScanlines: 1, numBins: 1, scanlinePadding: 0, binPadding: 0 };
    }
    let numScanlines = 1;
    if (numIncludedShapesWithEdges > 0) {
      const medianShapeHeightPerEdge = MathUtils.computeWeightedMedian(
        shapeHeightsPerEdge.subarray(0, numIncludedShapesWithEdges),
        shapeAreas.subarray(0, numIncludedShapesWithEdges),
      );
      const scanlineHeight = edgesPerScanline * medianShapeHeightPerEdge;
      numScanlines = MathUtils.clamp(
        Math.round(objectBounds.height / scanlineHeight),
        1,
        WebGLShapesRasterizer._maxGridSize,
      );
    }
    const medianShapeWidth = shapeWidths.subarray(0, numIncludedShapes).sort()[
      numIncludedShapes >> 1
    ]!;
    const medianShapeHeight = shapeHeights
      .subarray(0, numIncludedShapes)
      .sort()[numIncludedShapes >> 1]!;
    let numBins = MathUtils.clamp(
      Math.round(objectBounds.width / (binWidthFactor * medianShapeWidth)),
      1,
      WebGLShapesRasterizer._maxGridSize,
    );
    if (numScanlines * numBins > WebGLShapesRasterizer._maxNumGridCells) {
      const scale = Math.sqrt(
        WebGLShapesRasterizer._maxNumGridCells / (numScanlines * numBins),
      );
      numScanlines = Math.max(1, Math.floor(numScanlines * scale));
      numBins = Math.max(1, Math.floor(numBins * scale));
    }
    return {
      numScanlines,
      numBins,
      scanlinePadding:
        (shapePadding * medianShapeHeight * numScanlines) / objectBounds.height,
      binPadding:
        (shapePadding * medianShapeWidth * numBins) / objectBounds.width,
    };
  }

  /**
   * Rasterizes shapes into scanline data, packed into a flat
   * {@link ArrayBuffer} suitable for uploading as an RGBA32UI texture
   *
   * Divides `objectBounds` into `numScanlines` horizontal scanlines, and
   * assigns each shape to the scanlines spanned by the bounding box of its
   * polygon shells, and each edge to the scanlines spanned by its own bounding
   * box, both padded by `scanlinePadding` above and below. Both are
   * conservative: a shape or edge may be assigned to scanlines it does not
   * actually reach into, but never to too few. Per scanline, it also lists the
   * shapes per x-bin, i.e. per column of `numBins` equally wide columns,
   * assigning each shape to the bins spanned by its x-range on that scanline,
   * padded by `binPadding` to the left and right. The paddings let the strokes
   * the fragment shader draws around a shape, which reach beyond it, find the
   * shape from the neighboring scanlines and bins as well (see
   * `WebGLShapesRenderOptions.shapePadding`). Every bin lists its shapes in
   * the order the fragment shader composites them in, which is `shapeOrder` if
   * given, and ascending order otherwise.
   *
   * Shapes are identified by their index among the *included* shapes, i.e.
   * `shapesMask` compacts the indices. Holes contribute their edges, but not
   * their bounds, as a shape is bounded by the shells of its polygons only.
   * Accordingly, the edges of a hole that reaches beyond the scanlines of the
   * shape's shells up to and including its own polygon, which is invalid
   * geometry, are dropped in those scanlines.
   *
   * Values are grouped into texels of four 32-bit values, in three sections:
   * the bin table, the bins' shape refs, and the scanlines' shapes and edges.
   * The shapes fragment shader documents the layout, and is the authority on
   * it.
   *
   * The shapes are rasterized twice, keeping only the index of every included
   * shape in between: the first pass only counts the shapes and edges on every
   * scanline and the shapes in every bin, which determines where everything
   * goes, and the second pass writes the shapes, their edges and the bins'
   * shape refs right there, in draw order. Apart from the buffer and
   * these indices, this only takes a few counters per scanline and one per bin.
   *
   * Yields between shapes, i.e. long-running rasterizations neither block the
   * event loop nor ignore an abort for long.
   *
   * @param numScanlines - Number of horizontal scanlines
   * @param numBins - Number of x-bins per scanline
   * @param scanlinePadding - Fraction of a scanline's height that shapes and
   * edges are padded by above and below (see {@link computeGridSize})
   * @param binPadding - Fraction of a bin's width that shapes are padded by to
   * the left and right (see {@link computeGridSize})
   * @param geometry - Geometry for all shapes in the object
   * @param shapesMask - Per-shape inclusion mask, or `undefined` if all shapes are included
   * @param objectBounds - Bounding box of all shapes
   * @param options - Optional abort signal, alignment, in 32-bit values, to
   * pad the buffer to (used to fill whole texture lines), and draw order
   * (`shapeOrder`, the indices of the included shapes in that order)
   * @returns The packed buffer. Without scanlines or bins, there is nothing to
   * rasterize into, and the buffer is empty
   * @throws Error if `objectBounds` has a non-positive width or height, as shapes
   * cannot be rasterized into a degenerate bounding box
   * @throws Error if `shapeOrder` does not hold one index per included shape
   */
  static async rasterizeScanlines(
    numScanlines: number,
    numBins: number,
    scanlinePadding: number,
    binPadding: number,
    geometry: ShapesGeometry,
    shapesMask: Uint8Array | undefined,
    objectBounds: Rect,
    options?: {
      signal?: AbortSignal;
      align?: number;
      shapeOrder?: Uint32Array;
    },
  ): Promise<ArrayBuffer> {
    const { signal, align = 1, shapeOrder } = options ?? {};
    signal?.throwIfAborted();
    if (objectBounds.width <= 0 || objectBounds.height <= 0) {
      throw new Error("Object bounds must have a positive width and height");
    }
    if (numScanlines < 1 || numBins < 1) {
      return new ArrayBuffer(0);
    }
    const { shapePolygonOffsets } = geometry;
    const maybeYield = AsyncUtils.createYielder();
    // the current shape, per scanline: whether it is on the scanline, and its
    // x-range and number of edges there (reset once the shape is done)
    const scanlineMeta = {
      xMins: new Float64Array(numScanlines),
      xMaxs: new Float64Array(numScanlines),
      numEdges: new Uint32Array(numScanlines),
      hasShape: new Uint8Array(numScanlines),
    };
    // the texel offset each scanline's next shape goes to (second pass only)
    const scanlineTexelOffsets = new Uint32Array(numScanlines);
    // first pass: count the shapes & edges per scanline, and the shapes per bin
    const scanlineNumShapes = new Uint32Array(numScanlines);
    const scanlineNumEdges = new Uint32Array(numScanlines);
    const binNumShapes = new Uint32Array(numScanlines * numBins);
    let totalNumBinShapes = 0;
    // the index in `geometry` of every included shape
    const includedShapes = new Uint32Array(shapePolygonOffsets.length - 1);
    let numIncludedShapes = 0;
    for (let s = 0; s < shapePolygonOffsets.length - 1; s++) {
      if (shapesMask === undefined || shapesMask[s]! > 0) {
        includedShapes[numIncludedShapes++] = s;
        const { firstShapeScanline, lastShapeScanline } =
          WebGLShapesRasterizer._rasterizeShape(
            s,
            geometry,
            objectBounds,
            numScanlines,
            scanlinePadding,
            scanlineMeta,
          );
        for (
          let shapeScanline = firstShapeScanline;
          shapeScanline <= lastShapeScanline;
          shapeScanline++
        ) {
          if (scanlineMeta.hasShape[shapeScanline] !== 0) {
            scanlineNumShapes[shapeScanline]!++;
            scanlineNumEdges[shapeScanline]! +=
              scanlineMeta.numEdges[shapeScanline]!;
            const { firstBin: firstShapeBin, lastBin: lastShapeBin } =
              WebGLShapesRasterizer._computeBinRange(
                scanlineMeta.xMins[shapeScanline]!,
                scanlineMeta.xMaxs[shapeScanline]!,
                numBins,
                binPadding,
                objectBounds,
              );
            for (
              let shapeBin = firstShapeBin;
              shapeBin <= lastShapeBin;
              shapeBin++
            ) {
              binNumShapes[shapeScanline * numBins + shapeBin]!++;
              totalNumBinShapes++;
            }
            scanlineMeta.hasShape[shapeScanline] = 0;
          }
        }
      }
      await maybeYield({ signal });
    }
    if (shapeOrder !== undefined && shapeOrder.length !== numIncludedShapes) {
      throw new Error(
        `The shape order holds ${shapeOrder.length} indices, expected one per included shape (${numIncludedShapes})`,
      );
    }
    // lay out the buffer
    const numBinTexels = Math.ceil((numScanlines * numBins) / 2);
    const numShapeRefTexels = Math.ceil(totalNumBinShapes / 4);
    let numScanlineTexels = 0;
    for (let scanline = 0; scanline < numScanlines; scanline++) {
      scanlineTexelOffsets[scanline] =
        numBinTexels + numShapeRefTexels + numScanlineTexels;
      numScanlineTexels +=
        scanlineNumShapes[scanline]! + scanlineNumEdges[scanline]!;
    }
    const buffer = new ArrayBuffer(
      MathUtils.align(
        4 * numBinTexels + // bin table -> bin entry B
          4 * numShapeRefTexels + // shape refs -> shape ref R
          4 * numScanlineTexels, // scanline S -> shape P -> shape header, edges
        align,
      ) * 4, // 4 bytes per 32-bit value
    );
    const uint32Buffer = new Uint32Array(buffer);
    const float32Buffer = new Float32Array(buffer);
    // bin table, turning the bins' shape counts into the value offsets their
    // next shape refs go to
    const binShapeRefValueOffsets = binNumShapes;
    let shapeRefValueOffset = 4 * numBinTexels;
    for (let bin = 0; bin < numScanlines * numBins; bin++) {
      const numBinShapes = binShapeRefValueOffsets[bin]!;
      binShapeRefValueOffsets[bin] = shapeRefValueOffset;
      uint32Buffer[2 * bin] = shapeRefValueOffset;
      uint32Buffer[2 * bin + 1] = numBinShapes;
      shapeRefValueOffset += numBinShapes;
    }
    // second pass: write the shapes, their edges and the bins' shape refs, in
    // the order the shapes are composited in
    for (let i = 0; i < numIncludedShapes; i++) {
      const shapeIndex = shapeOrder !== undefined ? shapeOrder[i]! : i;
      const { firstShapeScanline, lastShapeScanline } =
        WebGLShapesRasterizer._rasterizeShape(
          includedShapes[shapeIndex]!,
          geometry,
          objectBounds,
          numScanlines,
          scanlinePadding,
          scanlineMeta,
          { float32Buffer, scanlineTexelOffsets },
        );
      for (
        let shapeScanline = firstShapeScanline;
        shapeScanline <= lastShapeScanline;
        shapeScanline++
      ) {
        if (scanlineMeta.hasShape[shapeScanline] !== 0) {
          const shapeTexelOffset = scanlineTexelOffsets[shapeScanline]!;
          const shapeValueOffset = 4 * shapeTexelOffset;
          uint32Buffer[shapeValueOffset] = shapeIndex;
          uint32Buffer[shapeValueOffset + 1] =
            scanlineMeta.numEdges[shapeScanline]!;
          float32Buffer[shapeValueOffset + 2] =
            scanlineMeta.xMins[shapeScanline]!;
          float32Buffer[shapeValueOffset + 3] =
            scanlineMeta.xMaxs[shapeScanline]!;
          const { firstBin: firstShapeBin, lastBin: lastShapeBin } =
            WebGLShapesRasterizer._computeBinRange(
              scanlineMeta.xMins[shapeScanline]!,
              scanlineMeta.xMaxs[shapeScanline]!,
              numBins,
              binPadding,
              objectBounds,
            );
          for (
            let shapeBin = firstShapeBin;
            shapeBin <= lastShapeBin;
            shapeBin++
          ) {
            const index = binShapeRefValueOffsets[
              shapeScanline * numBins + shapeBin
            ]!++;
            uint32Buffer[index] = shapeTexelOffset;
          }
          scanlineTexelOffsets[shapeScanline] =
            shapeTexelOffset + 1 + scanlineMeta.numEdges[shapeScanline]!;
          scanlineMeta.hasShape[shapeScanline] = 0;
        }
      }
      await maybeYield({ signal });
    }
    return buffer;
  }

  /**
   * Assigns a shape to the scanlines of its polygons' shells, and its edges to
   * theirs, for {@link rasterizeScanlines}
   *
   * Records the shape's x-range and number of edges on every scanline it is
   * on in `scanlineMeta`, which must not have a shape on any scanline yet,
   * and, if given an `output`, also writes its edges to the buffer, right after
   * the shape's header texel at the scanline's texel offset. The shapes and
   * edges are padded like in {@link rasterizeScanlines}, and the edges of a
   * hole are only assigned to the scanlines the shells of the shape's
   * polygons up to and including the hole's have been assigned to.
   *
   * @param s - Index of the shape in `geometry`
   * @param geometry - Geometry for all shapes in the object
   * @param objectBounds - Bounding box of all shapes
   * @param numScanlines - Number of horizontal scanlines
   * @param scanlinePadding - See {@link rasterizeScanlines}
   * @param scanlineMeta - The current shape on each scanline, see
   * {@link rasterizeScanlines}
   * @param output - The buffer to write the edges to, and the texel offset of
   * the shape's header on every scanline, if any
   * @returns The range of scanlines the shape's polygons span, including those
   * in between the polygons of a multi-polygon shape, which do not have the
   * shape
   */
  private static _rasterizeShape(
    s: number,
    geometry: ShapesGeometry,
    objectBounds: Rect,
    numScanlines: number,
    scanlinePadding: number,
    scanlineMeta: {
      hasShape: Uint8Array;
      xMins: Float64Array;
      xMaxs: Float64Array;
      numEdges: Uint32Array;
    },
    output?: { float32Buffer: Float32Array; scanlineTexelOffsets: Uint32Array },
  ): { firstShapeScanline: number; lastShapeScanline: number } {
    const {
      shapePolygonOffsets,
      polygonRingOffsets,
      ringVertexOffsets,
      coords,
    } = geometry;
    const { xMins, xMaxs, numEdges, hasShape } = scanlineMeta;
    let firstShapeScanline = numScanlines;
    let lastShapeScanline = -1;
    for (
      let p = shapePolygonOffsets[s]!;
      p < shapePolygonOffsets[s + 1]!;
      p++
    ) {
      const polygonRingStart = polygonRingOffsets[p]!;
      const polygonRingEnd = polygonRingOffsets[p + 1]!;
      // compute the bounding box of the polygon's shell
      let xMin = Infinity,
        yMin = Infinity,
        xMax = -Infinity,
        yMax = -Infinity;
      const shellVertexStart = ringVertexOffsets[polygonRingStart]!;
      const shellVertexEnd = ringVertexOffsets[polygonRingStart + 1]!;
      for (let v = shellVertexStart; v < shellVertexEnd; v++) {
        const x = coords[2 * v]!;
        const y = coords[2 * v + 1]!;
        xMin = Math.min(xMin, x);
        yMin = Math.min(yMin, y);
        xMax = Math.max(xMax, x);
        yMax = Math.max(yMax, y);
      }
      const firstPolygonScanline = MathUtils.clamp(
        Math.floor(
          (numScanlines * (yMin - objectBounds.y)) / objectBounds.height -
            scanlinePadding, // slack below, for strokes
        ),
        0,
        numScanlines - 1,
      );
      const lastPolygonScanline = MathUtils.clamp(
        Math.floor(
          (numScanlines * (yMax - objectBounds.y)) / objectBounds.height +
            scanlinePadding, // slack above, for strokes
        ),
        0,
        numScanlines - 1,
      );
      firstShapeScanline = Math.min(firstShapeScanline, firstPolygonScanline);
      lastShapeScanline = Math.max(lastShapeScanline, lastPolygonScanline);
      for (
        let polygonScanline = firstPolygonScanline;
        polygonScanline <= lastPolygonScanline;
        polygonScanline++
      ) {
        if (hasShape[polygonScanline] === 0) {
          hasShape[polygonScanline] = 1;
          xMins[polygonScanline] = xMin;
          xMaxs[polygonScanline] = xMax;
          numEdges[polygonScanline] = 0;
        } else {
          xMins[polygonScanline] = Math.min(xMins[polygonScanline]!, xMin);
          xMaxs[polygonScanline] = Math.max(xMaxs[polygonScanline]!, xMax);
        }
      }
      // add the polygon's edges to the scanlines
      for (let r = polygonRingStart; r < polygonRingEnd; r++) {
        const ringVertexStart = ringVertexOffsets[r]!;
        const ringVertexEnd = ringVertexOffsets[r + 1]!;
        const nRingVertices = ringVertexEnd - ringVertexStart;
        for (let i = 0; i < nRingVertices; ++i) {
          const v0 = ringVertexStart + ((i + 0) % nRingVertices);
          const v1 = ringVertexStart + ((i + 1) % nRingVertices);
          const v0x = coords[2 * v0]!;
          const v0y = coords[2 * v0 + 1]!;
          const v1x = coords[2 * v1]!;
          const v1y = coords[2 * v1 + 1]!;
          if (v0x === v1x && v0y === v1y) {
            continue; // ignore zero-length edges
          }
          const firstEdgeScanline = MathUtils.clamp(
            Math.floor(
              (numScanlines * (Math.min(v0y, v1y) - objectBounds.y)) /
                objectBounds.height -
                scanlinePadding, // slack below, for strokes
            ),
            0,
            numScanlines - 1,
          );
          const lastEdgeScanline = MathUtils.clamp(
            Math.floor(
              (numScanlines * (Math.max(v0y, v1y) - objectBounds.y)) /
                objectBounds.height +
                scanlinePadding, // slack above, for strokes
            ),
            0,
            numScanlines - 1,
          );
          for (
            let edgeScanline = firstEdgeScanline;
            edgeScanline <= lastEdgeScanline;
            edgeScanline++
          ) {
            if (hasShape[edgeScanline] === 0) {
              continue; // hole beyond the scanlines of the shape's shells
            }
            if (output !== undefined) {
              const { float32Buffer, scanlineTexelOffsets } = output;
              const edgeValueOffset =
                4 *
                (scanlineTexelOffsets[edgeScanline]! +
                  1 +
                  numEdges[edgeScanline]!);
              float32Buffer[edgeValueOffset] = v0x;
              float32Buffer[edgeValueOffset + 1] = v0y;
              float32Buffer[edgeValueOffset + 2] = v1x;
              float32Buffer[edgeValueOffset + 3] = v1y;
            }
            numEdges[edgeScanline]!++;
          }
        }
      }
    }
    return { firstShapeScanline, lastShapeScanline };
  }

  /**
   * Computes the range of bins a shape is listed in on a scanline, for
   * {@link rasterizeScanlines}
   *
   * @param xMin - Minimum x coordinate of the shape on the scanline
   * @param xMax - Maximum x coordinate of the shape on the scanline
   * @param numBins - Number of x-bins per scanline
   * @param binPadding - See {@link rasterizeScanlines}
   * @param objectBounds - Bounding box of all shapes
   * @returns The first and the last bin, both clamped to the scanline
   */
  private static _computeBinRange(
    xMin: number,
    xMax: number,
    numBins: number,
    binPadding: number,
    objectBounds: Rect,
  ): { firstBin: number; lastBin: number } {
    // slack to the left, for strokes
    const firstBin = MathUtils.clamp(
      Math.floor(
        (numBins * (xMin - objectBounds.x)) / objectBounds.width - binPadding,
      ),
      0,
      numBins - 1,
    );
    // slack to the right, for strokes
    const lastBin = MathUtils.clamp(
      Math.floor(
        (numBins * (xMax - objectBounds.x)) / objectBounds.width + binPadding,
      ),
      0,
      numBins - 1,
    );
    return { firstBin, lastBin };
  }
}
