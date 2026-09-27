import { deepEqual } from "fast-equals";
import type { mat3 } from "gl-matrix";

import {
  AsyncUtils,
  type Color,
  ColorUtils,
  ConfigUtils,
  GeometryUtils,
  type GroupValueMap,
  type Rect,
  type Shapes,
  type ShapesData,
  type ShapesGeometry,
  type Table,
  type TableData,
  TransformUtils,
  type WebGLShapesRenderOptions,
  defaultShapeFillColor,
  defaultShapeFillOpacity,
  defaultShapeFillVisibility,
  defaultShapeOpacity,
  defaultShapeStrokeColor,
  defaultShapeStrokeOpacity,
  defaultShapeStrokeVisibility,
  defaultShapeVisibility,
  projectDefaults,
} from "@tissuumaps/core";

import shapesFragmentShader from "../assets/shaders/shapes.frag?raw";
import shapesVertexShader from "../assets/shaders/shapes.vert?raw";
import { ColorResolver } from "../resolvers/ColorResolver";
import { OpacityResolver } from "../resolvers/OpacityResolver";
import { VisibilityResolver } from "../resolvers/VisibilityResolver";
import type { WebGLContext } from "./WebGLContext";
import {
  type ObjectRef,
  type RenderedObjectBase,
  WebGLRendererBase,
} from "./WebGLRendererBase";
import { WebGLShapesRasterizer } from "./WebGLShapesRasterizer";
import { WebGLUtils } from "./WebGLUtils";

/**
 * WebGL renderer for two-dimensional shape clouds
 *
 * Shapes are rasterized on the GPU via a scanline-based algorithm. Each shapes
 * object is represented by a quad covering its bounds within the viewport,
 * whose fragment shader samples a scanline data texture to determine polygon
 * membership, fill colors, and stroke colors.
 *
 * Every object owns its textures. Synchronizing compares the current model
 * against the snapshot those textures were built from, and rebuilds only the
 * ones whose inputs changed. Layer- and object-level properties (transforms,
 * visibility and opacity) are shader uniforms, so changing them never touches
 * the textures, and never requires a synchronization (see
 * {@link WebGLRendererBase.setModel}).
 */
export class WebGLShapesRenderer extends WebGLRendererBase<
  Shapes,
  ShapesData,
  WebGLShapesSyncContext,
  PreparedShapes,
  RenderedShapes
> {
  private static readonly _scanlineDataTextureWidth = 4096; // see fragment shader
  private static readonly _shapeColorsTextureWidth = 4096; // see fragment shader
  private static readonly _numValuesPerScanlineDataTextureLine =
    4 * WebGLShapesRenderer._scanlineDataTextureWidth; // 4 values per RGBA32UI texel
  private static readonly _numValuesPerShapeColorsTextureLine =
    1 * WebGLShapesRenderer._shapeColorsTextureWidth; // 1 value per R32UI texel
  private static readonly _textureUnits = {
    SCANLINE_DATA: 1, // unit 0 is used by the points renderer
    SHAPE_FILL_COLORS: 2,
    SHAPE_STROKE_COLORS: 3,
  };

  renderOptions: WebGLShapesRenderOptions =
    projectDefaults.glOptions.shapesRenderOptions;
  private readonly _program: WebGLProgram;
  private readonly _uniformLocations: {
    quad: WebGLUniformLocation;
    viewportToWorldMatrix: WebGLUniformLocation;
    worldToDataMatrix: WebGLUniformLocation;
    numScanlines: WebGLUniformLocation;
    numBins: WebGLUniformLocation;
    objectBounds: WebGLUniformLocation;
    opacityFactor: WebGLUniformLocation;
    halfStrokeWidth: WebGLUniformLocation;
    devicePixelSize: WebGLUniformLocation;
  };

  /**
   * Creates the shader program and retrieves uniform locations
   *
   * @param context - The WebGL context to use for rendering
   */
  constructor(context: WebGLContext) {
    super(context);
    this._program = context.createProgram(
      shapesVertexShader,
      shapesFragmentShader,
    );
    this._uniformLocations = {
      quad: context.getUniformLocation(this._program, "u_quad"),
      viewportToWorldMatrix: context.getUniformLocation(
        this._program,
        "u_viewportToWorldMatrix",
      ),
      worldToDataMatrix: context.getUniformLocation(
        this._program,
        "u_worldToDataMatrix",
      ),
      numScanlines: context.getUniformLocation(this._program, "u_numScanlines"),
      numBins: context.getUniformLocation(this._program, "u_numBins"),
      objectBounds: context.getUniformLocation(this._program, "u_objectBounds"),
      opacityFactor: context.getUniformLocation(
        this._program,
        "u_opacityFactor",
      ),
      halfStrokeWidth: context.getUniformLocation(
        this._program,
        "u_halfStrokeWidth",
      ),
      devicePixelSize: context.getUniformLocation(
        this._program,
        "u_devicePixelSize",
      ),
    };
    // texture units never change, so the sampler uniforms are set only once
    context.gl.useProgram(this._program);
    context.gl.uniform1i(
      context.getUniformLocation(this._program, "u_scanlineData"),
      WebGLShapesRenderer._textureUnits.SCANLINE_DATA,
    );
    context.gl.uniform1i(
      context.getUniformLocation(this._program, "u_shapeFillColors"),
      WebGLShapesRenderer._textureUnits.SHAPE_FILL_COLORS,
    );
    context.gl.uniform1i(
      context.getUniformLocation(this._program, "u_shapeStrokeColors"),
      WebGLShapesRenderer._textureUnits.SHAPE_STROKE_COLORS,
    );
    context.gl.useProgram(null);
  }

  /**
   * Issues the WebGL draw calls for all synchronized shapes
   *
   * Renders each shapes object as a quad whose fragment shader performs
   * scanline-based polygon rasterization using the per-object scanline data
   * texture, with the viewport → world matrix as a global uniform and its
   * world → data matrix, bounds, numbers of scanlines and bins, half stroke
   * width and device pixel size in data units (for anti-aliasing) and opacity
   * factor as per-pass uniforms, computed from the current model (see
   * {@link getRenderPasses}). The quad covers the object's bounds, dilated by
   * how far the anti-aliased strokes reach beyond them, within the viewport, so
   * that fragments are only shaded where the object can be; objects outside the
   * viewport, objects whose layer or object is invisible, and objects without
   * scanlines or bins are skipped.
   */
  draw(): void {
    const renderPasses = this.getRenderPasses();
    if (renderPasses.length === 0) {
      return;
    }
    this.context.gl.useProgram(this._program);
    this.context.gl.uniformMatrix3x2fv(
      this._uniformLocations.viewportToWorldMatrix,
      false,
      WebGLUtils.convertMatrixToGLMat3x2(
        WebGLUtils.createViewportToWorldMatrix(this.viewport),
      ),
    );
    const worldDevicePixelSize =
      0.5 *
      (this.viewport.width / this.context.gl.canvas.width +
        this.viewport.height / this.context.gl.canvas.height);
    this.context.enableAlphaBlending();
    for (const {
      layer,
      object: shapes,
      renderedObject: renderedShapes,
    } of renderPasses) {
      if (renderedShapes.numScanlines < 1 || renderedShapes.numBins < 1) {
        continue;
      }
      const opacityFactor = WebGLShapesRenderer.computeOpacityFactor(
        layer,
        shapes,
      );
      if (opacityFactor === 0) {
        continue;
      }
      const quad = this._computeQuad(
        renderedShapes.objectBounds,
        WebGLUtils.createDataToWorldMatrix(shapes.transform, layer.transform),
        worldDevicePixelSize,
      );
      if (quad === null) {
        continue;
      }
      this.context.gl.uniform4f(
        this._uniformLocations.quad,
        quad.x,
        quad.y,
        quad.width,
        quad.height,
      );
      this.context.gl.uniformMatrix3x2fv(
        this._uniformLocations.worldToDataMatrix,
        false,
        WebGLUtils.convertMatrixToGLMat3x2(
          WebGLUtils.createWorldToDataMatrix(shapes.transform, layer.transform),
        ),
      );
      this.context.gl.uniform1ui(
        this._uniformLocations.numScanlines,
        renderedShapes.numScanlines,
      );
      this.context.gl.uniform1ui(
        this._uniformLocations.numBins,
        renderedShapes.numBins,
      );
      this.context.gl.uniform4f(
        this._uniformLocations.objectBounds,
        renderedShapes.objectBounds.x,
        renderedShapes.objectBounds.y,
        renderedShapes.objectBounds.width,
        renderedShapes.objectBounds.height,
      );
      this.context.gl.uniform1f(
        this._uniformLocations.opacityFactor,
        opacityFactor,
      );
      this.context.gl.uniform1f(
        this._uniformLocations.halfStrokeWidth,
        (0.5 * this.renderOptions.strokeWidth) /
          (shapes.transform.scale * layer.transform.scale),
      );
      this.context.gl.uniform1f(
        this._uniformLocations.devicePixelSize,
        worldDevicePixelSize / (shapes.transform.scale * layer.transform.scale),
      );
      this.context.gl.activeTexture(
        WebGL2RenderingContext.TEXTURE0 +
          WebGLShapesRenderer._textureUnits.SCANLINE_DATA,
      );
      this.context.gl.bindTexture(
        WebGL2RenderingContext.TEXTURE_2D,
        renderedShapes.scanlineDataTexture,
      );
      this.context.gl.activeTexture(
        WebGL2RenderingContext.TEXTURE0 +
          WebGLShapesRenderer._textureUnits.SHAPE_FILL_COLORS,
      );
      this.context.gl.bindTexture(
        WebGL2RenderingContext.TEXTURE_2D,
        renderedShapes.shapeFillColorsTexture,
      );
      this.context.gl.activeTexture(
        WebGL2RenderingContext.TEXTURE0 +
          WebGLShapesRenderer._textureUnits.SHAPE_STROKE_COLORS,
      );
      this.context.gl.bindTexture(
        WebGL2RenderingContext.TEXTURE_2D,
        renderedShapes.shapeStrokeColorsTexture,
      );
      this.context.gl.drawArrays(WebGL2RenderingContext.TRIANGLE_STRIP, 0, 4);
    }
    this.context.disableAlphaBlending();
    this.context.gl.useProgram(null);
  }

  /**
   * Releases the shader program and all per-object GPU textures
   */
  destroy(): void {
    this.context.gl.deleteProgram(this._program);
    this.clearRenderedObjects();
  }

  /**
   * Gets the bounding box of all drawn shapes, in world coordinates
   *
   * Dilates the bounds of the shapes by half the stroke width, as the strokes
   * reach beyond them, but not by the anti-aliasing margin, which depends on
   * the current zoom level (see {@link _computeQuad}).
   */
  override getRenderedBounds(): Rect | null {
    const bounds = super.getRenderedBounds();
    if (bounds === null) {
      return null;
    }
    return GeometryUtils.dilate(bounds, 0.5 * this.renderOptions.strokeWidth);
  }

  /**
   * Returns the grid render options (the number of edges per scanline, the
   * bin width factor and the shape padding), which the scanline data textures
   * are rasterized for, so that changing them requires a resynchronization
   */
  protected override getRenderOptionsSyncState(): object {
    return {
      edgesPerScanline: this.renderOptions.edgesPerScanline,
      binWidthFactor: this.renderOptions.binWidthFactor,
      shapePadding: this.renderOptions.shapePadding,
    };
  }

  /**
   * Prepares everything that has to be uploaded for an object
   *
   * Decides what the object's textures need - the geometry for a new object or
   * changed grid render options (see {@link getRenderOptionsSyncState}), and
   * the resolved fill and stroke colors whose configurations or referenced
   * maps changed (see {@link _createRenderConfigSnapshot}) -
   * requests all of it before the first `await`, and then computes the bounds,
   * the grid size and the paddings, rasterizes the scanline data and folds the
   * resolved visibilities and opacities into the colors.
   *
   * See {@link WebGLRendererBase.prepareRenderedObject} for when this runs.
   *
   * @param newRef - The object to prepare
   * @param renderedShapes - The object's current GPU state, if it is reused
   * @param syncContext - The inputs of the current synchronization
   * @param options - Optional abort signal
   * @returns The snapshot the decisions were based on, the bounds, the grid
   * render options and the numbers of scanlines and bins, the packed scanline
   * data (if the geometry was loaded) and the resolved colors that have to be
   * uploaded, or `null` if the shapes have no area
   */
  protected override async prepareRenderedObject(
    newRef: ShapesRef,
    renderedShapes: RenderedShapes | undefined,
    syncContext: WebGLShapesSyncContext,
    options?: { signal?: AbortSignal },
  ): Promise<PreparedShapes | null> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const loadTable = WebGLShapesRenderer.createObjectTableLoader(
      newRef,
      syncContext,
    );
    const { edgesPerScanline, binWidthFactor, shapePadding } =
      this.renderOptions;
    const geometryChanged =
      renderedShapes === undefined ||
      renderedShapes.edgesPerScanline !== edgesPerScanline ||
      renderedShapes.binWidthFactor !== binWidthFactor ||
      renderedShapes.shapePadding !== shapePadding;
    const renderConfigSnapshot =
      WebGLShapesRenderer._createRenderConfigSnapshot(newRef, syncContext);
    const fillColorsChanged =
      WebGLShapesRenderer._checkShapeFillColorsTextureChanged(
        renderedShapes,
        renderConfigSnapshot,
      );
    const strokeColorsChanged =
      WebGLShapesRenderer._checkShapeStrokeColorsTextureChanged(
        renderedShapes,
        renderConfigSnapshot,
      );
    const geometryPromise = geometryChanged
      ? newRef.data.loadGeometry({ signal })
      : undefined;
    const colorsChanged = fillColorsChanged || strokeColorsChanged;
    const packedShapeVisibilitiesPromise = colorsChanged
      ? VisibilityResolver.resolveVisibilities(
          newRef.itemIds,
          newRef.object.shapeVisibility,
          syncContext.visibilityMaps,
          defaultShapeVisibility,
          {
            signal,
            loadTable,
            align: WebGLShapesRenderer._numValuesPerShapeColorsTextureLine,
          },
        )
      : undefined;
    const packedShapeOpacitiesPromise = colorsChanged
      ? OpacityResolver.resolveOpacities(
          newRef.itemIds,
          newRef.object.shapeOpacity,
          syncContext.opacityMaps,
          defaultShapeOpacity,
          {
            signal,
            loadTable,
            align: WebGLShapesRenderer._numValuesPerShapeColorsTextureLine,
          },
        )
      : undefined;
    const packedShapeFillColorsPromise = fillColorsChanged
      ? ColorResolver.resolveColors(
          newRef.itemIds,
          newRef.object.shapeFillColor,
          syncContext.colorMaps,
          defaultShapeFillColor,
          {
            signal,
            loadTable,
            align: WebGLShapesRenderer._numValuesPerShapeColorsTextureLine,
          },
        )
      : undefined;
    const packedShapeFillVisibilitiesPromise = fillColorsChanged
      ? VisibilityResolver.resolveVisibilities(
          newRef.itemIds,
          newRef.object.shapeFillVisibility,
          syncContext.visibilityMaps,
          defaultShapeFillVisibility,
          {
            signal,
            loadTable,
            align: WebGLShapesRenderer._numValuesPerShapeColorsTextureLine,
          },
        )
      : undefined;
    const packedShapeFillOpacitiesPromise = fillColorsChanged
      ? OpacityResolver.resolveOpacities(
          newRef.itemIds,
          newRef.object.shapeFillOpacity,
          syncContext.opacityMaps,
          defaultShapeFillOpacity,
          {
            signal,
            loadTable,
            align: WebGLShapesRenderer._numValuesPerShapeColorsTextureLine,
          },
        )
      : undefined;
    const packedShapeStrokeColorsPromise = strokeColorsChanged
      ? ColorResolver.resolveColors(
          newRef.itemIds,
          newRef.object.shapeStrokeColor,
          syncContext.colorMaps,
          defaultShapeStrokeColor,
          {
            signal,
            loadTable,
            align: WebGLShapesRenderer._numValuesPerShapeColorsTextureLine,
          },
        )
      : undefined;
    const packedShapeStrokeVisibilitiesPromise = strokeColorsChanged
      ? VisibilityResolver.resolveVisibilities(
          newRef.itemIds,
          newRef.object.shapeStrokeVisibility,
          syncContext.visibilityMaps,
          defaultShapeStrokeVisibility,
          {
            signal,
            loadTable,
            align: WebGLShapesRenderer._numValuesPerShapeColorsTextureLine,
          },
        )
      : undefined;
    const packedShapeStrokeOpacitiesPromise = strokeColorsChanged
      ? OpacityResolver.resolveOpacities(
          newRef.itemIds,
          newRef.object.shapeStrokeOpacity,
          syncContext.opacityMaps,
          defaultShapeStrokeOpacity,
          {
            signal,
            loadTable,
            align: WebGLShapesRenderer._numValuesPerShapeColorsTextureLine,
          },
        )
      : undefined;
    const [
      geometry,
      packedShapeVisibilities,
      packedShapeOpacities,
      packedShapeFillColors,
      packedShapeFillVisibilities,
      packedShapeFillOpacities,
      packedShapeStrokeColors,
      packedShapeStrokeVisibilities,
      packedShapeStrokeOpacities,
    ] = await Promise.all([
      geometryPromise,
      packedShapeVisibilitiesPromise,
      packedShapeOpacitiesPromise,
      packedShapeFillColorsPromise,
      packedShapeFillVisibilitiesPromise,
      packedShapeFillOpacitiesPromise,
      packedShapeStrokeColorsPromise,
      packedShapeStrokeVisibilitiesPromise,
      packedShapeStrokeOpacitiesPromise,
    ]);
    signal?.throwIfAborted();
    let objectBounds: Rect;
    let numScanlines: number;
    let numBins: number;
    let scanlineBuffer: Uint32Array | undefined;
    if (geometry !== undefined) {
      const newObjectBounds = await WebGLShapesRenderer._getObjectBounds(
        geometry,
        newRef.itemsMask,
        { signal },
      );
      if (newObjectBounds === null) {
        return null;
      }
      objectBounds = newObjectBounds;
      let scanlinePadding: number;
      let binPadding: number;
      ({ numScanlines, numBins, scanlinePadding, binPadding } =
        await WebGLShapesRasterizer.computeGridSize(
          geometry,
          newRef.itemsMask,
          objectBounds,
          edgesPerScanline,
          binWidthFactor,
          shapePadding,
          { signal },
        ));
      scanlineBuffer = await WebGLShapesRenderer._createScanlineBuffer(
        numScanlines,
        numBins,
        scanlinePadding,
        binPadding,
        geometry,
        newRef.itemsMask,
        objectBounds,
        { signal },
      );
      this._checkDataTextureSize(
        "scanline data",
        WebGLShapesRenderer._scanlineDataTextureWidth,
        scanlineBuffer.length /
          WebGLShapesRenderer._numValuesPerScanlineDataTextureLine,
      );
    } else if (renderedShapes !== undefined) {
      objectBounds = renderedShapes.objectBounds;
      numScanlines = renderedShapes.numScanlines;
      numBins = renderedShapes.numBins;
    } else {
      throw new Error("Geometry must be loaded for new shapes object");
    }
    for (const packedShapeColors of [
      packedShapeFillColors,
      packedShapeStrokeColors,
    ]) {
      if (packedShapeColors !== undefined) {
        this._checkDataTextureSize(
          "shape colors",
          WebGLShapesRenderer._shapeColorsTextureWidth,
          packedShapeColors.length /
            WebGLShapesRenderer._numValuesPerShapeColorsTextureLine,
        );
      }
    }
    if (
      packedShapeFillColors !== undefined &&
      packedShapeFillVisibilities !== undefined &&
      packedShapeFillOpacities !== undefined &&
      packedShapeVisibilities !== undefined &&
      packedShapeOpacities !== undefined
    ) {
      await AsyncUtils.forEach(
        packedShapeFillColors,
        (packedShapeFillColor, i) => {
          packedShapeFillColors[i] = ColorUtils.withAlpha(
            packedShapeFillColor,
            Math.min(
              packedShapeVisibilities[i]!,
              packedShapeFillVisibilities[i]!,
            ),
            Math.round(
              (packedShapeOpacities[i]! * packedShapeFillOpacities[i]!) / 255,
            ),
          );
        },
        { signal },
      );
    }
    if (
      packedShapeStrokeColors !== undefined &&
      packedShapeStrokeVisibilities !== undefined &&
      packedShapeStrokeOpacities !== undefined &&
      packedShapeVisibilities !== undefined &&
      packedShapeOpacities !== undefined
    ) {
      await AsyncUtils.forEach(
        packedShapeStrokeColors,
        (packedShapeStrokeColor, i) => {
          packedShapeStrokeColors[i] = ColorUtils.withAlpha(
            packedShapeStrokeColor,
            Math.min(
              packedShapeVisibilities[i]!,
              packedShapeStrokeVisibilities[i]!,
            ),
            Math.round(
              (packedShapeOpacities[i]! * packedShapeStrokeOpacities[i]!) / 255,
            ),
          );
        },
        { signal },
      );
    }
    return {
      objectBounds,
      edgesPerScanline,
      binWidthFactor,
      shapePadding,
      numScanlines,
      numBins,
      scanlineBuffer,
      packedShapeFillColors,
      packedShapeStrokeColors,
      renderConfigSnapshot,
    };
  }

  /**
   * Creates the textures of a new object
   *
   * @param newRef - The object
   * @param prepared - Its preparation, which holds every texture's data
   * @returns The rendered object
   */
  protected override createRenderedObject(
    newRef: ShapesRef,
    prepared: PreparedShapes,
  ): RenderedShapes {
    if (
      prepared.scanlineBuffer === undefined ||
      prepared.packedShapeFillColors === undefined ||
      prepared.packedShapeStrokeColors === undefined
    ) {
      throw new Error("All textures must be prepared for new shapes object");
    }
    return {
      ref: newRef,
      renderConfigSnapshot: prepared.renderConfigSnapshot,
      objectBounds: prepared.objectBounds,
      edgesPerScanline: prepared.edgesPerScanline,
      binWidthFactor: prepared.binWidthFactor,
      shapePadding: prepared.shapePadding,
      numScanlines: prepared.numScanlines,
      numBins: prepared.numBins,
      scanlineDataTexture: this._createScanlineDataTexture(
        prepared.scanlineBuffer,
      ),
      shapeFillColorsTexture: this._createShapeColorsTexture(
        prepared.packedShapeFillColors,
      ),
      shapeStrokeColorsTexture: this._createShapeColorsTexture(
        prepared.packedShapeStrokeColors,
      ),
    };
  }

  /**
   * Reloads the textures of an object whose data was prepared again
   *
   * Refills the colors textures in place, and creates a new scanline data
   * texture before releasing the one it replaces, so that the old one survives
   * if creating the new one throws. The snapshot is adopted last, so that an
   * update that throws is retried by the next synchronization.
   *
   * @param renderedShapes - The rendered object to update in place
   * @param prepared - Its preparation, holding the texture data that changed
   * @returns Whether any texture was reloaded
   */
  protected override updateRenderedObject(
    renderedShapes: RenderedShapes,
    prepared: PreparedShapes,
  ): boolean {
    if (prepared.scanlineBuffer !== undefined) {
      const scanlineDataTexture = this._createScanlineDataTexture(
        prepared.scanlineBuffer,
      );
      this.context.gl.deleteTexture(renderedShapes.scanlineDataTexture);
      renderedShapes.scanlineDataTexture = scanlineDataTexture;
      renderedShapes.objectBounds = prepared.objectBounds;
      renderedShapes.edgesPerScanline = prepared.edgesPerScanline;
      renderedShapes.binWidthFactor = prepared.binWidthFactor;
      renderedShapes.shapePadding = prepared.shapePadding;
      renderedShapes.numScanlines = prepared.numScanlines;
      renderedShapes.numBins = prepared.numBins;
    }
    if (prepared.packedShapeFillColors !== undefined) {
      this._loadShapeColorsTexture(
        renderedShapes.shapeFillColorsTexture,
        prepared.packedShapeFillColors,
      );
    }
    if (prepared.packedShapeStrokeColors !== undefined) {
      this._loadShapeColorsTexture(
        renderedShapes.shapeStrokeColorsTexture,
        prepared.packedShapeStrokeColors,
      );
    }
    renderedShapes.renderConfigSnapshot = prepared.renderConfigSnapshot;
    return (
      prepared.scanlineBuffer !== undefined ||
      prepared.packedShapeFillColors !== undefined ||
      prepared.packedShapeStrokeColors !== undefined
    );
  }

  /**
   * Deletes all GPU textures owned by a single rendered object
   *
   * @param renderedShapes - The rendered object whose textures to delete
   */
  protected override destroyRenderedObject(
    renderedShapes: RenderedShapes,
  ): void {
    this.context.gl.deleteTexture(renderedShapes.scanlineDataTexture);
    this.context.gl.deleteTexture(renderedShapes.shapeFillColorsTexture);
    this.context.gl.deleteTexture(renderedShapes.shapeStrokeColorsTexture);
  }

  /**
   * Builds the scanline data of a shapes object
   *
   * Rasterizes all shapes into horizontal scanlines and their x-bins, packed
   * into a 32-bit integer buffer, aligned to the lines of the scanline data
   * texture that {@link _createScanlineDataTexture} creates from it.
   *
   * @param numScanlines - Number of scanlines to rasterize into
   * @param numBins - Number of x-bins per scanline
   * @param scanlinePadding - See {@link WebGLShapesRasterizer.rasterizeScanlines}
   * @param binPadding - See {@link WebGLShapesRasterizer.rasterizeScanlines}
   * @param geometry - Geometry for all shapes in the object
   * @param shapesMask - Per-shape inclusion mask, or `undefined` if all shapes are included
   * @param objectBounds - Axis-aligned bounding box of all shapes
   * @param options - Optional abort signal
   * @returns The packed scanline data
   */
  private static async _createScanlineBuffer(
    numScanlines: number,
    numBins: number,
    scanlinePadding: number,
    binPadding: number,
    geometry: ShapesGeometry,
    shapesMask: Uint8Array | undefined,
    objectBounds: Rect,
    options?: { signal?: AbortSignal },
  ): Promise<Uint32Array> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const scanlineBuffer = await WebGLShapesRasterizer.rasterizeScanlines(
      numScanlines,
      numBins,
      scanlinePadding,
      binPadding,
      geometry,
      shapesMask,
      objectBounds,
      {
        align: WebGLShapesRenderer._numValuesPerScanlineDataTextureLine,
        signal,
      },
    );
    return new Uint32Array(scanlineBuffer);
  }

  /**
   * Throws unless a data texture of the given size fits the GPU
   *
   * Checked while an object is prepared, where a failure drops only that
   * object, with a message that says why, rather than when its textures are
   * created, where WebGL would fail with an opaque error, or render garbage.
   * The width is fixed per texture, but WebGL 2 only guarantees 2048 texels.
   *
   * @param name - What the texture holds, for the error message
   * @param width - The width of the texture, in texels
   * @param height - The number of texture lines the data needs
   * @throws Error if the width or height exceeds the maximum texture size
   */
  private _checkDataTextureSize(
    name: string,
    width: number,
    height: number,
  ): void {
    const { maxTextureSize } = this.context;
    if (width > maxTextureSize || height > maxTextureSize) {
      throw new Error(
        `The ${name} texture needs ${width}x${height} texels, but this GPU supports at most ${maxTextureSize}x${maxTextureSize}`,
      );
    }
  }

  /**
   * Uploads packed scanline data as an RGBA32UI texture
   *
   * @param scanlineBuffer - The packed scanline data, see {@link _createScanlineBuffer}
   * @returns The scanline data texture
   */
  private _createScanlineDataTexture(
    scanlineBuffer: Uint32Array,
  ): WebGLTexture {
    return this.context.createDataTexture(
      WebGL2RenderingContext.RGBA32UI,
      WebGLShapesRenderer._scanlineDataTextureWidth,
      scanlineBuffer.length /
        WebGLShapesRenderer._numValuesPerScanlineDataTextureLine,
      WebGL2RenderingContext.RGBA_INTEGER,
      WebGL2RenderingContext.UNSIGNED_INT,
      scanlineBuffer,
    );
  }

  /**
   * Uploads packed fill or stroke colors as a new R32UI texture
   *
   * @param packedShapeColors - The packed RGBA colors, one per shape, aligned
   * to the texture width
   * @returns The colors texture
   */
  private _createShapeColorsTexture(
    packedShapeColors: Uint32Array,
  ): WebGLTexture {
    return this.context.createDataTexture(
      WebGL2RenderingContext.R32UI,
      WebGLShapesRenderer._shapeColorsTextureWidth,
      packedShapeColors.length /
        WebGLShapesRenderer._numValuesPerShapeColorsTextureLine,
      WebGL2RenderingContext.RED_INTEGER,
      WebGL2RenderingContext.UNSIGNED_INT,
      packedShapeColors,
    );
  }

  /**
   * Refills an existing colors texture with packed fill or stroke colors
   *
   * The texture has to have been created by {@link _createShapeColorsTexture}
   * for the same number of shapes: its storage is immutable, so the colors
   * have to fill it exactly.
   *
   * @param texture - The colors texture to refill
   * @param packedShapeColors - The packed RGBA colors, one per shape, aligned
   * to the texture width
   */
  private _loadShapeColorsTexture(
    texture: WebGLTexture,
    packedShapeColors: Uint32Array,
  ): void {
    this.context.loadDataTexture(
      texture,
      WebGLShapesRenderer._shapeColorsTextureWidth,
      packedShapeColors.length /
        WebGLShapesRenderer._numValuesPerShapeColorsTextureLine,
      WebGL2RenderingContext.RED_INTEGER,
      WebGL2RenderingContext.UNSIGNED_INT,
      packedShapeColors,
    );
  }

  /**
   * Computes the quad to draw an object with, in viewport coordinates
   *
   * The object's bounds are transformed to world coordinates, dilated by how
   * far the anti-aliased strokes reach beyond the shapes (half the stroke
   * width, which is at least half a pixel, plus half a pixel, see fragment
   * shader), and intersected with the viewport, then expressed in viewport
   * coordinates, `[0, 1]` spanning the viewport. Fragments outside the
   * object's bounds are discarded by the fragment shader anyway, so drawing
   * only this quad changes nothing but the number of fragments shaded.
   *
   * @param objectBounds - The bounds of the object's shapes, in data coordinates
   * @param dataToWorldMatrix - The data → world matrix of the object
   * @param worldDevicePixelSize - The size of a device pixel, in world coordinates
   * @returns The quad, or `null` if the object lies outside the viewport
   */
  private _computeQuad(
    objectBounds: Rect,
    dataToWorldMatrix: mat3,
    worldDevicePixelSize: number,
  ): Rect | null {
    const worldBounds = TransformUtils.transformBoundingBox(
      objectBounds,
      dataToWorldMatrix,
    );
    const margin =
      Math.max(
        0.5 * this.renderOptions.strokeWidth,
        0.5 * worldDevicePixelSize,
      ) +
      0.5 * worldDevicePixelSize;
    const visibleBounds = GeometryUtils.intersection(
      GeometryUtils.dilate(worldBounds, margin),
      this.viewport,
    );
    if (visibleBounds === null) {
      return null;
    }
    return TransformUtils.transformBoundingBox(
      visibleBounds,
      WebGLUtils.createWorldToViewportMatrix(this.viewport),
    );
  }

  /**
   * Captures what the textures of an object are resolved from
   *
   * Every property the change predicates read has to be captured here: the
   * item-level configurations, and the maps they resolve their values from,
   * looked up now so that the predicates compare maps rather than map IDs (see
   * {@link ConfigUtils.findGroupByMap}).
   *
   * @param newRef - The object to capture the snapshot of
   * @param syncContext - The inputs of the current synchronization, holding the maps
   * @returns The snapshot
   */
  private static _createRenderConfigSnapshot(
    newRef: ShapesRef,
    syncContext: WebGLShapesSyncContext,
  ): RenderedShapes["renderConfigSnapshot"] {
    return {
      shapeVisibility: newRef.object.shapeVisibility,
      shapeOpacity: newRef.object.shapeOpacity,
      shapeFillColor: newRef.object.shapeFillColor,
      shapeFillVisibility: newRef.object.shapeFillVisibility,
      shapeFillOpacity: newRef.object.shapeFillOpacity,
      shapeStrokeColor: newRef.object.shapeStrokeColor,
      shapeStrokeVisibility: newRef.object.shapeStrokeVisibility,
      shapeStrokeOpacity: newRef.object.shapeStrokeOpacity,
      shapeVisibilityMap: ConfigUtils.findGroupByMap(
        newRef.object.shapeVisibility,
        syncContext.visibilityMaps,
      ),
      shapeOpacityMap: ConfigUtils.findGroupByMap(
        newRef.object.shapeOpacity,
        syncContext.opacityMaps,
      ),
      shapeFillColorMap: ConfigUtils.findGroupByMap(
        newRef.object.shapeFillColor,
        syncContext.colorMaps,
      ),
      shapeFillVisibilityMap: ConfigUtils.findGroupByMap(
        newRef.object.shapeFillVisibility,
        syncContext.visibilityMaps,
      ),
      shapeFillOpacityMap: ConfigUtils.findGroupByMap(
        newRef.object.shapeFillOpacity,
        syncContext.opacityMaps,
      ),
      shapeStrokeColorMap: ConfigUtils.findGroupByMap(
        newRef.object.shapeStrokeColor,
        syncContext.colorMaps,
      ),
      shapeStrokeVisibilityMap: ConfigUtils.findGroupByMap(
        newRef.object.shapeStrokeVisibility,
        syncContext.visibilityMaps,
      ),
      shapeStrokeOpacityMap: ConfigUtils.findGroupByMap(
        newRef.object.shapeStrokeOpacity,
        syncContext.opacityMaps,
      ),
    };
  }

  /**
   * Returns whether the fill colors of an object have to be resolved again
   *
   * Colors carry the resolved shape and fill visibilities and opacities in
   * their alpha channel, so they also depend on those configurations. The
   * layer- and object-level visibility and opacity are shader uniforms (see
   * {@link WebGLRendererBase.computeOpacityFactor}) and do not matter here.
   * Also true for an object that has not been rendered yet, like the other
   * predicate. Configurations are compared by value, maps by identity (see
   * {@link _createRenderConfigSnapshot}).
   */
  private static _checkShapeFillColorsTextureChanged(
    renderedShapes: RenderedShapes | undefined,
    newSnapshot: RenderedShapes["renderConfigSnapshot"],
  ): boolean {
    return (
      renderedShapes === undefined ||
      !deepEqual(
        renderedShapes.renderConfigSnapshot.shapeVisibility,
        newSnapshot.shapeVisibility,
      ) ||
      renderedShapes.renderConfigSnapshot.shapeVisibilityMap !==
        newSnapshot.shapeVisibilityMap ||
      !deepEqual(
        renderedShapes.renderConfigSnapshot.shapeOpacity,
        newSnapshot.shapeOpacity,
      ) ||
      renderedShapes.renderConfigSnapshot.shapeOpacityMap !==
        newSnapshot.shapeOpacityMap ||
      !deepEqual(
        renderedShapes.renderConfigSnapshot.shapeFillColor,
        newSnapshot.shapeFillColor,
      ) ||
      renderedShapes.renderConfigSnapshot.shapeFillColorMap !==
        newSnapshot.shapeFillColorMap ||
      !deepEqual(
        renderedShapes.renderConfigSnapshot.shapeFillVisibility,
        newSnapshot.shapeFillVisibility,
      ) ||
      renderedShapes.renderConfigSnapshot.shapeFillVisibilityMap !==
        newSnapshot.shapeFillVisibilityMap ||
      !deepEqual(
        renderedShapes.renderConfigSnapshot.shapeFillOpacity,
        newSnapshot.shapeFillOpacity,
      ) ||
      renderedShapes.renderConfigSnapshot.shapeFillOpacityMap !==
        newSnapshot.shapeFillOpacityMap
    );
  }

  /**
   * Returns whether the stroke colors of an object have to be resolved again
   *
   * See {@link _checkShapeFillColorsTextureChanged}.
   */
  private static _checkShapeStrokeColorsTextureChanged(
    renderedShapes: RenderedShapes | undefined,
    newSnapshot: RenderedShapes["renderConfigSnapshot"],
  ): boolean {
    return (
      renderedShapes === undefined ||
      !deepEqual(
        renderedShapes.renderConfigSnapshot.shapeVisibility,
        newSnapshot.shapeVisibility,
      ) ||
      renderedShapes.renderConfigSnapshot.shapeVisibilityMap !==
        newSnapshot.shapeVisibilityMap ||
      !deepEqual(
        renderedShapes.renderConfigSnapshot.shapeOpacity,
        newSnapshot.shapeOpacity,
      ) ||
      renderedShapes.renderConfigSnapshot.shapeOpacityMap !==
        newSnapshot.shapeOpacityMap ||
      !deepEqual(
        renderedShapes.renderConfigSnapshot.shapeStrokeColor,
        newSnapshot.shapeStrokeColor,
      ) ||
      renderedShapes.renderConfigSnapshot.shapeStrokeColorMap !==
        newSnapshot.shapeStrokeColorMap ||
      !deepEqual(
        renderedShapes.renderConfigSnapshot.shapeStrokeVisibility,
        newSnapshot.shapeStrokeVisibility,
      ) ||
      renderedShapes.renderConfigSnapshot.shapeStrokeVisibilityMap !==
        newSnapshot.shapeStrokeVisibilityMap ||
      !deepEqual(
        renderedShapes.renderConfigSnapshot.shapeStrokeOpacity,
        newSnapshot.shapeStrokeOpacity,
      ) ||
      renderedShapes.renderConfigSnapshot.shapeStrokeOpacityMap !==
        newSnapshot.shapeStrokeOpacityMap
    );
  }

  /**
   * Computes the axis-aligned bounding box of all (included) shapes
   *
   * @param geometry - Geometry for all shapes in the object
   * @param shapesMask - Per-shape inclusion mask, or `undefined` if all shapes are included
   * @param options - Optional abort signal
   * @returns The bounding rectangle in data-space coordinates, or `null` if the
   * shapes have no area, including shapes without any vertices, as such an
   * object can neither be rasterized into scanlines nor drawn by the fragment
   * shader
   */
  private static async _getObjectBounds(
    geometry: ShapesGeometry,
    shapesMask: Uint8Array | undefined,
    options?: { signal?: AbortSignal },
  ): Promise<Rect | null> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const {
      shapePolygonOffsets,
      polygonRingOffsets,
      ringVertexOffsets,
      coords,
    } = geometry;
    let xMin = Infinity,
      yMin = Infinity,
      xMax = -Infinity,
      yMax = -Infinity;
    const maybeYield = AsyncUtils.createYielder();
    for (let s = 0; s < shapePolygonOffsets.length - 1; s++) {
      if (shapesMask === undefined || shapesMask[s]! > 0) {
        const polygonStart = shapePolygonOffsets[s]!;
        const polygonEnd = shapePolygonOffsets[s + 1]!;
        for (let p = polygonStart; p < polygonEnd; p++) {
          const shellRing = polygonRingOffsets[p]!;
          const shellVertexStart = ringVertexOffsets[shellRing]!;
          const shellVertexEnd = ringVertexOffsets[shellRing + 1]!;
          for (let v = shellVertexStart; v < shellVertexEnd; v++) {
            const x = coords[2 * v]!;
            const y = coords[2 * v + 1]!;
            if (x < xMin) {
              xMin = x;
            }
            if (y < yMin) {
              yMin = y;
            }
            if (x > xMax) {
              xMax = x;
            }
            if (y > yMax) {
              yMax = y;
            }
          }
        }
      }
      await maybeYield({ signal });
    }
    // also true without any vertices, where the minima and maxima are infinite
    if (xMin >= xMax || yMin >= yMax) {
      return null;
    }
    return { x: xMin, y: yMin, width: xMax - xMin, height: yMax - yMin };
  }
}

/**
 * The inputs of a synchronization of the shapes renderer
 *
 * The tables and group-to-value maps that the objects resolve their
 * properties from, and the loaders for shapes and table data (see
 * {@link WebGLRendererBase.synchronize}).
 */
export type WebGLShapesSyncContext = {
  tables: Table[];
  colorMaps: GroupValueMap<Color>[];
  visibilityMaps: GroupValueMap<boolean>[];
  opacityMaps: GroupValueMap<number>[];
  loadObject: (
    shapes: Shapes,
    options?: { signal?: AbortSignal },
  ) => Promise<ShapesData>;
  loadTable: (
    table: Table,
    options?: { signal?: AbortSignal },
  ) => Promise<TableData>;
};

/**
 * A reference to a shapes object, its layer, and its loaded data
 */
type ShapesRef = ObjectRef<Shapes, ShapesData>;

/**
 * What a preparation of a shapes object uploads
 *
 * See {@link WebGLShapesRenderer.prepareRenderedObject}.
 */
type PreparedShapes = {
  objectBounds: Rect;
  edgesPerScanline: number;
  binWidthFactor: number;
  shapePadding: number;
  numScanlines: number;
  numBins: number;
  scanlineBuffer: Uint32Array | undefined;
  packedShapeFillColors: Uint32Array | undefined;
  packedShapeStrokeColors: Uint32Array | undefined;
  renderConfigSnapshot: RenderedShapes["renderConfigSnapshot"];
};

/**
 * GPU state for a single shapes object
 *
 * Holds the texture handles for scanline data, fill colors and stroke colors,
 * plus a snapshot of the model values they were built from, which the change
 * predicates compare against (see
 * {@link WebGLShapesRenderer._createRenderConfigSnapshot}). Layer- and
 * object-level properties are read from the current model when drawing (see
 * {@link WebGLRendererBase.getRenderPasses}), and are not part of the
 * snapshot. The grid render options are the ones the scanline data texture
 * was rasterized for (see
 * {@link WebGLShapesRenderer.getRenderOptionsSyncState}), and the numbers of
 * scanlines and bins are the ones it was rasterized into, which are also the
 * ones the object is drawn with.
 */
type RenderedShapes = RenderedObjectBase<Shapes, ShapesData> & {
  edgesPerScanline: number;
  binWidthFactor: number;
  shapePadding: number;
  numScanlines: number;
  numBins: number;
  scanlineDataTexture: WebGLTexture;
  shapeFillColorsTexture: WebGLTexture;
  shapeStrokeColorsTexture: WebGLTexture;
  renderConfigSnapshot: Pick<
    Shapes,
    | "shapeVisibility"
    | "shapeOpacity"
    | "shapeFillColor"
    | "shapeFillVisibility"
    | "shapeFillOpacity"
    | "shapeStrokeColor"
    | "shapeStrokeVisibility"
    | "shapeStrokeOpacity"
  > & {
    shapeVisibilityMap: GroupValueMap<boolean> | undefined;
    shapeOpacityMap: GroupValueMap<number> | undefined;
    shapeFillColorMap: GroupValueMap<Color> | undefined;
    shapeFillVisibilityMap: GroupValueMap<boolean> | undefined;
    shapeFillOpacityMap: GroupValueMap<number> | undefined;
    shapeStrokeColorMap: GroupValueMap<Color> | undefined;
    shapeStrokeVisibilityMap: GroupValueMap<boolean> | undefined;
    shapeStrokeOpacityMap: GroupValueMap<number> | undefined;
  };
};
