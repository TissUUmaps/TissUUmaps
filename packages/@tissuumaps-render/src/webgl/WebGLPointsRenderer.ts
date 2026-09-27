import { deepEqual } from "fast-equals";

import {
  AsyncUtils,
  type Color,
  type ColorConfig,
  ColorUtils,
  ConfigUtils,
  type CoordinateSpace,
  type GroupValueMap,
  type Layer,
  type Marker,
  MathUtils,
  type OpacityConfig,
  type Points,
  type PointsData,
  type PointsGeometry,
  type Rect,
  type SizeConfig,
  type Table,
  type TableData,
  type TypedArray,
  type VisibilityConfig,
  type WebGLPointsRenderOptions,
  defaultPointColor,
  defaultPointMarker,
  defaultPointOpacity,
  defaultPointSize,
  defaultPointSizeUnit,
  defaultPointVisibility,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
  projectDefaults,
} from "@tissuumaps/core";

import markersUrl from "../assets/markers/markers.png?url";
import pointsFragmentShader from "../assets/shaders/points.frag?raw";
import pointsVertexShader from "../assets/shaders/points.vert?raw";
import { ColorResolver } from "../resolvers/ColorResolver";
import { MarkerResolver } from "../resolvers/MarkerResolver";
import { OpacityResolver } from "../resolvers/OpacityResolver";
import { SizeResolver } from "../resolvers/SizeResolver";
import { VisibilityResolver } from "../resolvers/VisibilityResolver";
import type { WebGLContext } from "./WebGLContext";
import {
  type ObjectRef,
  type RenderedObjectBase,
  WebGLRendererBase,
} from "./WebGLRendererBase";
import { WebGLUtils } from "./WebGLUtils";

/**
 * WebGL renderer for two-dimensional point clouds
 *
 * Manages a shader program, a marker atlas texture, and per-object GPU
 * buffers. Points are rendered as `gl.POINTS` using per-vertex attributes (x,
 * y, size, color, marker), one draw call per object.
 *
 * Every object owns its vertex array and attribute buffers. Synchronizing
 * compares the current model against the snapshot those buffers were loaded
 * from, and re-uploads only the attributes whose inputs changed. An attribute
 * whose configuration is a constant has no buffer at all: its value is
 * supplied as a generic vertex attribute when drawing, which saves the buffer,
 * its upload and the resolve (see {@link _backAttribute}).
 * Layer- and object-level properties (transforms, point size factors,
 * visibility and opacity) are shader uniforms, so changing them never touches
 * the buffers, and never requires a synchronization (see
 * {@link WebGLRendererBase.setModel}).
 */
export class WebGLPointsRenderer extends WebGLRendererBase<
  Points,
  PointsData,
  WebGLPointsSyncContext,
  PreparedPoints,
  RenderedPoints
> {
  private static readonly _attribLocations = {
    X: 0,
    Y: 1,
    SIZE: 2,
    COLOR: 3,
    MARKER: 4,
  };
  private static readonly _textureUnits = {
    MARKER_ATLAS: 0,
  };

  renderOptions: WebGLPointsRenderOptions =
    projectDefaults.glOptions.pointsRenderOptions;
  private readonly _program: WebGLProgram;
  private readonly _uniformLocations: {
    globalPointSizeFactor: WebGLUniformLocation;
    worldToViewportMatrix: WebGLUniformLocation;
    viewportSize: WebGLUniformLocation;
    canvasSize: WebGLUniformLocation;
    dataToWorldMatrix: WebGLUniformLocation;
    pointSizeFactor: WebGLUniformLocation;
    opacityFactor: WebGLUniformLocation;
  };
  private _markerAtlasTexture: WebGLTexture | undefined;
  private _destroyed: boolean = false;

  /**
   * Creates the shader program and retrieves uniform locations
   *
   * The marker atlas texture is loaded asynchronously, so the renderer must not
   * be drawn before `onInitialized` has been called; `onError` is called instead
   * if loading it failed or was aborted, or if the renderer was destroyed
   * meanwhile, in which case the texture is released right away.
   *
   * @param context - The WebGL context to use for rendering
   * @param onInitialized - Called once the marker atlas texture has been loaded
   * @param onError - Called if the marker atlas texture could not be loaded
   * @param options - Optional abort signal
   */
  constructor(
    context: WebGLContext,
    onInitialized: () => void,
    onError: (error: Error) => void,
    options?: { signal?: AbortSignal },
  ) {
    super(context);
    const { signal } = options ?? {};
    this._program = context.createProgram(
      pointsVertexShader,
      pointsFragmentShader,
    );
    this._uniformLocations = {
      globalPointSizeFactor: context.getUniformLocation(
        this._program,
        "u_globalPointSizeFactor",
      ),
      worldToViewportMatrix: context.getUniformLocation(
        this._program,
        "u_worldToViewportMatrix",
      ),
      viewportSize: context.getUniformLocation(this._program, "u_viewportSize"),
      canvasSize: context.getUniformLocation(this._program, "u_canvasSize"),
      dataToWorldMatrix: context.getUniformLocation(
        this._program,
        "u_dataToWorldMatrix",
      ),
      pointSizeFactor: context.getUniformLocation(
        this._program,
        "u_pointSizeFactor",
      ),
      opacityFactor: context.getUniformLocation(
        this._program,
        "u_opacityFactor",
      ),
    };
    // texture units never change, so the sampler uniforms are set only once
    context.gl.useProgram(this._program);
    context.gl.uniform1i(
      context.getUniformLocation(this._program, "u_markerAtlas"),
      WebGLPointsRenderer._textureUnits.MARKER_ATLAS,
    );
    context.gl.useProgram(null);
    const initialize = async () => {
      signal?.throwIfAborted();
      const markerAtlasTexture = await context.loadImageTextureFromUrl(
        markersUrl,
        { mipmap: true, signal },
      );
      if (this._destroyed) {
        context.gl.deleteTexture(markerAtlasTexture);
        throw new Error("Renderer destroyed");
      }
      this._markerAtlasTexture = markerAtlasTexture;
    };
    initialize().then(onInitialized, onError);
  }

  /**
   * Issues the WebGL draw calls for all synchronized points
   *
   * Binds the shader program, configures the global uniforms (transform,
   * viewport, canvas size in device pixels), binds the marker atlas texture,
   * and then draws every object in its own `gl.POINTS` call with alpha
   * blending, with its data → world matrix, point size factor and opacity
   * factor as uniforms, computed from the current model (see
   * {@link getRenderPasses}), and with the constants of its per-point
   * attributes as generic vertex attributes (see {@link _backAttribute}).
   * Objects whose layer or object is invisible are skipped.
   *
   * @throws Error if the renderer has not been initialized
   */
  draw(): void {
    if (this._markerAtlasTexture === undefined) {
      throw new Error("Not initialized");
    }
    const renderPasses = this.getRenderPasses();
    if (renderPasses.length === 0) {
      return;
    }
    this.context.gl.useProgram(this._program);
    this.context.gl.uniform1f(
      this._uniformLocations.globalPointSizeFactor,
      this.renderOptions.globalPointSizeFactor,
    );
    this.context.gl.uniformMatrix3x2fv(
      this._uniformLocations.worldToViewportMatrix,
      false,
      WebGLUtils.convertMatrixToGLMat3x2(
        WebGLUtils.createWorldToViewportMatrix(this.viewport),
      ),
    );
    this.context.gl.uniform2f(
      this._uniformLocations.viewportSize,
      this.viewport.width,
      this.viewport.height,
    );
    this.context.gl.uniform2f(
      this._uniformLocations.canvasSize,
      this.context.gl.canvas.width,
      this.context.gl.canvas.height,
    );
    this.context.gl.activeTexture(
      WebGL2RenderingContext.TEXTURE0 +
        WebGLPointsRenderer._textureUnits.MARKER_ATLAS,
    );
    this.context.gl.bindTexture(
      WebGL2RenderingContext.TEXTURE_2D,
      this._markerAtlasTexture,
    );
    this.context.enableAlphaBlending();
    for (const {
      layer,
      object: points,
      renderedObject: renderedPoints,
    } of renderPasses) {
      const opacityFactor = WebGLPointsRenderer.computeOpacityFactor(
        layer,
        points,
      );
      if (opacityFactor === 0) {
        continue;
      }
      this.context.gl.uniformMatrix3x2fv(
        this._uniformLocations.dataToWorldMatrix,
        false,
        WebGLUtils.convertMatrixToGLMat3x2(
          WebGLUtils.createDataToWorldMatrix(points.transform, layer.transform),
        ),
      );
      this.context.gl.uniform1f(
        this._uniformLocations.pointSizeFactor,
        WebGLPointsRenderer._computePointSizeFactor(layer, points),
      );
      this.context.gl.uniform1f(
        this._uniformLocations.opacityFactor,
        opacityFactor,
      );
      this.context.gl.bindVertexArray(renderedPoints.vao);
      const { marker, size, color } = renderedPoints.attributes;
      if (typeof marker === "number") {
        this.context.gl.vertexAttribI4ui(
          WebGLPointsRenderer._attribLocations.MARKER,
          marker,
          0,
          0,
          0,
        );
      }
      if (typeof size === "number") {
        this.context.gl.vertexAttrib1f(
          WebGLPointsRenderer._attribLocations.SIZE,
          size,
        );
      }
      if (typeof color === "number") {
        this.context.gl.vertexAttribI4ui(
          WebGLPointsRenderer._attribLocations.COLOR,
          color,
          0,
          0,
          0,
        );
      }
      this.context.gl.drawArrays(
        WebGL2RenderingContext.POINTS,
        0,
        renderedPoints.ref.itemIds.length,
      );
    }
    this.context.gl.bindVertexArray(null);
    this.context.disableAlphaBlending();
    this.context.gl.useProgram(null);
  }

  /**
   * Releases the shader program, the marker atlas texture, and all per-object
   * GPU resources
   *
   * A marker atlas texture that is still loading is released once it has
   * loaded (see the constructor).
   */
  destroy(): void {
    this._destroyed = true;
    this.context.gl.deleteProgram(this._program);
    if (this._markerAtlasTexture !== undefined) {
      this.context.gl.deleteTexture(this._markerAtlasTexture);
      this._markerAtlasTexture = undefined;
    }
    this.clearRenderedObjects();
  }

  /**
   * Returns the state of a points object that a synchronization depends on
   *
   * Blanks out the object-level point size factor, and the units of the point
   * size configuration, on top of what the base class blanks out: the sizes are
   * uploaded without their unit, which is applied as part of the point size
   * factor uniform instead (see {@link _computePointSizeFactor}).
   *
   * @param points - The points object to return the state of
   * @returns The points object without the properties that are applied when drawing
   */
  protected override getObjectSyncState(points: Points): object {
    return {
      ...super.getObjectSyncState(points),
      pointSizeFactor: undefined,
      pointSize: WebGLPointsRenderer._stripSizeUnit(points.pointSize),
    };
  }

  /**
   * Prepares everything that has to be uploaded for an object
   *
   * Decides what the object's buffers need - the geometry for a new object,
   * and the markers, sizes and colors whose configurations or referenced maps
   * changed (see {@link _createRenderConfigSnapshot}) - requests all of it
   * before the first `await`, and then computes the bounds and folds the
   * resolved visibilities and opacities into the colors. An attribute whose
   * configuration is a constant is not resolved per point: it is prepared as
   * the one packed value all points share (see {@link _backAttribute}).
   *
   * See {@link WebGLRendererBase.prepareRenderedObject} for when this runs.
   *
   * @param newRef - The object to prepare
   * @param renderedPoints - The object's current GPU state, if it is reused
   * @param syncContext - The inputs of the current synchronization
   * @param options - Optional abort signal
   * @returns The snapshot the decisions were based on, the masked geometry
   * (new objects only), the bounds, and the attributes that have to be
   * uploaded, each resolved per point or as a constant
   */
  protected override async prepareRenderedObject(
    newRef: PointsRef,
    renderedPoints: RenderedPoints | undefined,
    syncContext: WebGLPointsSyncContext,
    options?: { signal?: AbortSignal },
  ): Promise<PreparedPoints> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const loadTable = WebGLPointsRenderer.createObjectTableLoader(
      newRef,
      syncContext,
    );
    const renderConfigSnapshot =
      WebGLPointsRenderer._createRenderConfigSnapshot(newRef, syncContext);
    const pointMarkerBufferChanged =
      WebGLPointsRenderer._checkPointMarkerBufferChanged(
        renderedPoints,
        renderConfigSnapshot,
      );
    const pointSizeBufferChanged =
      WebGLPointsRenderer._checkPointSizeBufferChanged(
        renderedPoints,
        renderConfigSnapshot,
      );
    const pointColorBufferChanged =
      WebGLPointsRenderer._checkPointColorBufferChanged(
        renderedPoints,
        renderConfigSnapshot,
      );
    const geometryPromise =
      renderedPoints === undefined
        ? newRef.data.loadGeometry({ signal })
        : undefined;
    const constantPointMarker = MarkerResolver.resolveConstantMarker(
      newRef.object.pointMarker,
    );
    const constantPointSize = SizeResolver.resolveConstantSize(
      newRef.object.pointSize,
    );
    const constantPointColor = WebGLPointsRenderer._resolveConstantColor(
      newRef.object.pointColor,
      newRef.object.pointVisibility,
      newRef.object.pointOpacity,
    );
    const packedPointMarkersPromise = pointMarkerBufferChanged
      ? (constantPointMarker ??
        MarkerResolver.resolveMarkers(
          newRef.itemIds,
          newRef.object.pointMarker,
          syncContext.markerMaps,
          defaultPointMarker,
          { signal, loadTable },
        ))
      : undefined;
    const packedPointSizesPromise = pointSizeBufferChanged
      ? (constantPointSize ??
        SizeResolver.resolveSizes(
          newRef.itemIds,
          newRef.object.pointSize,
          syncContext.sizeMaps,
          defaultPointSize,
          { signal, loadTable },
        ))
      : undefined;
    const packedPointColorsPromise = pointColorBufferChanged
      ? (constantPointColor ??
        ColorResolver.resolveColors(
          newRef.itemIds,
          newRef.object.pointColor,
          syncContext.colorMaps,
          defaultPointColor,
          { signal, loadTable },
        ))
      : undefined;
    const packedPointVisibilitiesPromise =
      pointColorBufferChanged && constantPointColor === undefined
        ? VisibilityResolver.resolveVisibilities(
            newRef.itemIds,
            newRef.object.pointVisibility,
            syncContext.visibilityMaps,
            defaultPointVisibility,
            { signal, loadTable },
          )
        : undefined;
    const packedPointOpacitiesPromise =
      pointColorBufferChanged && constantPointColor === undefined
        ? OpacityResolver.resolveOpacities(
            newRef.itemIds,
            newRef.object.pointOpacity,
            syncContext.opacityMaps,
            defaultPointOpacity,
            { signal, loadTable },
          )
        : undefined;
    const [
      geometry,
      packedPointMarkers,
      packedPointSizes,
      packedPointColors,
      packedPointVisibilities,
      packedPointOpacities,
    ] = await Promise.all([
      geometryPromise,
      packedPointMarkersPromise,
      packedPointSizesPromise,
      packedPointColorsPromise,
      packedPointVisibilitiesPromise,
      packedPointOpacitiesPromise,
    ]);
    signal?.throwIfAborted();
    let maskedGeometry: PointsGeometry | undefined;
    let objectBounds: Rect;
    if (geometry !== undefined) {
      let { xs, ys } = geometry;
      const pointsMask = newRef.itemsMask;
      if (pointsMask !== undefined) {
        let k = 0;
        const mxs = new Float32Array(newRef.itemIds.length);
        const mys = new Float32Array(newRef.itemIds.length);
        await AsyncUtils.forEach(
          pointsMask,
          (included, j) => {
            if (included > 0) {
              mxs[k] = xs[j]!;
              mys[k] = ys[j]!;
              k++;
            }
          },
          { signal },
        );
        xs = mxs;
        ys = mys;
      }
      maskedGeometry = { xs, ys };
      objectBounds = await WebGLPointsRenderer._getObjectBounds(
        maskedGeometry,
        { signal },
      );
    } else if (renderedPoints !== undefined) {
      objectBounds = renderedPoints.objectBounds;
    } else {
      throw new Error("Geometry must be loaded for new points object");
    }
    if (
      typeof packedPointColors === "object" &&
      packedPointVisibilities !== undefined &&
      packedPointOpacities !== undefined
    ) {
      await AsyncUtils.forEach(
        packedPointColors,
        (packedPointColor, i) => {
          packedPointColors[i] = ColorUtils.withAlpha(
            packedPointColor,
            packedPointVisibilities[i]!,
            packedPointOpacities[i]!,
          );
        },
        { signal },
      );
    }
    return {
      maskedGeometry,
      objectBounds,
      packedPointMarkers,
      packedPointSizes,
      packedPointColors,
      renderConfigSnapshot,
    };
  }

  /**
   * Creates the vertex array and attributes of a new object
   *
   * @param newRef - The object
   * @param prepared - Its preparation, which holds every attribute
   * @returns The rendered object
   */
  protected override createRenderedObject(
    newRef: PointsRef,
    prepared: PreparedPoints,
  ): RenderedPoints {
    if (
      prepared.maskedGeometry === undefined ||
      prepared.packedPointMarkers === undefined ||
      prepared.packedPointSizes === undefined ||
      prepared.packedPointColors === undefined
    ) {
      throw new Error("All attributes must be resolved for new points object");
    }
    return {
      ref: newRef,
      renderConfigSnapshot: prepared.renderConfigSnapshot,
      objectBounds: prepared.objectBounds,
      ...this._createVertexArray(
        prepared.maskedGeometry,
        prepared.packedPointMarkers,
        prepared.packedPointSizes,
        prepared.packedPointColors,
      ),
    };
  }

  /**
   * Backs the attributes of an object that were resolved again
   *
   * The snapshot is adopted last, so that an update that throws is retried by
   * the next synchronization.
   *
   * @param renderedPoints - The rendered object to update in place
   * @param prepared - Its preparation, holding the attributes that changed
   * @returns Whether any attribute was backed again
   */
  protected override updateRenderedObject(
    renderedPoints: RenderedPoints,
    prepared: PreparedPoints,
  ): boolean {
    const { vao, attributes } = renderedPoints;
    if (prepared.packedPointMarkers !== undefined) {
      attributes.marker = this._backAttribute(
        vao,
        WebGLPointsRenderer._attribLocations.MARKER,
        WebGL2RenderingContext.UNSIGNED_BYTE,
        prepared.packedPointMarkers,
        attributes.marker,
      );
    }
    if (prepared.packedPointSizes !== undefined) {
      attributes.size = this._backAttribute(
        vao,
        WebGLPointsRenderer._attribLocations.SIZE,
        WebGL2RenderingContext.FLOAT,
        prepared.packedPointSizes,
        attributes.size,
      );
    }
    if (prepared.packedPointColors !== undefined) {
      attributes.color = this._backAttribute(
        vao,
        WebGLPointsRenderer._attribLocations.COLOR,
        WebGL2RenderingContext.UNSIGNED_INT,
        prepared.packedPointColors,
        attributes.color,
      );
    }
    renderedPoints.renderConfigSnapshot = prepared.renderConfigSnapshot;
    return (
      prepared.packedPointMarkers !== undefined ||
      prepared.packedPointSizes !== undefined ||
      prepared.packedPointColors !== undefined
    );
  }

  /**
   * Deletes the vertex array and all attribute buffers owned by a single
   * rendered object
   *
   * @param renderedPoints - The rendered object whose GPU resources to delete
   */
  protected override destroyRenderedObject(
    renderedPoints: RenderedPoints,
  ): void {
    this.context.gl.deleteVertexArray(renderedPoints.vao);
    for (const buffer of Object.values(renderedPoints.buffers)) {
      this.context.gl.deleteBuffer(buffer);
    }
    for (const attribute of Object.values(renderedPoints.attributes)) {
      if (typeof attribute !== "number") {
        this.context.gl.deleteBuffer(attribute);
      }
    }
  }

  /**
   * Creates the vertex array of an object, configured with its coordinate
   * buffers and per-point attributes
   *
   * The coordinates always go into buffers; the other attributes go into
   * buffers or are constants, see {@link _backAttribute}.
   *
   * @param geometry - X and Y coordinates of the points, in data coordinates
   * @param packedPointMarkers - The marker index of every point, or of all points
   * @param packedPointSizes - The size of every point, or of all points
   * @param packedPointColors - The packed RGBA color of every point, or of all points
   * @returns The vertex array, the coordinate buffers and the per-point
   * attributes it is configured with
   */
  private _createVertexArray(
    geometry: PointsGeometry,
    packedPointMarkers: Uint8Array | number,
    packedPointSizes: Float32Array | number,
    packedPointColors: Uint32Array | number,
  ): Pick<RenderedPoints, "vao" | "buffers" | "attributes"> {
    const vao = this.context.createVertexArray();
    const buffers = {
      x: this.context.createBuffer(),
      y: this.context.createBuffer(),
    };
    this.context.gl.bindVertexArray(vao);
    this.context.configureVertexFloatAttribute(
      WebGL2RenderingContext.ARRAY_BUFFER,
      buffers.x,
      WebGLPointsRenderer._attribLocations.X,
      1,
      WebGL2RenderingContext.FLOAT,
    );
    this.context.configureVertexFloatAttribute(
      WebGL2RenderingContext.ARRAY_BUFFER,
      buffers.y,
      WebGLPointsRenderer._attribLocations.Y,
      1,
      WebGL2RenderingContext.FLOAT,
    );
    this.context.gl.bindVertexArray(null);
    this.context.allocateBuffer(
      WebGL2RenderingContext.ARRAY_BUFFER,
      buffers.x,
      geometry.xs,
      WebGL2RenderingContext.STATIC_DRAW,
    );
    this.context.allocateBuffer(
      WebGL2RenderingContext.ARRAY_BUFFER,
      buffers.y,
      geometry.ys,
      WebGL2RenderingContext.STATIC_DRAW,
    );
    const attributes = {
      marker: this._backAttribute(
        vao,
        WebGLPointsRenderer._attribLocations.MARKER,
        WebGL2RenderingContext.UNSIGNED_BYTE,
        packedPointMarkers,
      ),
      size: this._backAttribute(
        vao,
        WebGLPointsRenderer._attribLocations.SIZE,
        WebGL2RenderingContext.FLOAT,
        packedPointSizes,
      ),
      color: this._backAttribute(
        vao,
        WebGLPointsRenderer._attribLocations.COLOR,
        WebGL2RenderingContext.UNSIGNED_INT,
        packedPointColors,
      ),
    };
    return { vao, buffers, attributes };
  }

  /**
   * Brings a per-point attribute of an object up to date with its prepared
   * values, and returns what the attribute is now backed by
   *
   * A per-point attribute is backed either by a buffer holding one value per
   * point, or, if its configuration is a constant, by nothing: the vertex
   * attribute array is disabled in the vertex array, and the constant is
   * supplied as a generic vertex attribute when drawing (see {@link draw}).
   * Per-point values refill an existing buffer in place, or allocate one and
   * enable the attribute array if the attribute was constant before; a constant
   * releases an existing buffer and disables the attribute array. The buffer is
   * released with the vertex array bound, as that is the only way a deleted
   * buffer is detached from its attribute pointer; otherwise the vertex array
   * would keep it alive.
   *
   * @param vao - The vertex array of the object
   * @param location - The attribute location, see {@link _attribLocations}
   * @param type - The data type of the attribute: `gl.FLOAT` configures a float
   * attribute, any other type an integer attribute
   * @param values - The prepared values: one per point, or one constant
   * @param current - What the attribute is currently backed by; omitted for a
   * new object
   * @returns The buffer now holding the values, or the constant
   */
  private _backAttribute(
    vao: WebGLVertexArrayObject,
    location: number,
    type: GLenum,
    values: Exclude<TypedArray, Float64Array> | number,
    current?: WebGLBuffer | number,
  ): WebGLBuffer | number {
    if (typeof values === "number") {
      if (typeof current === "object") {
        this.context.gl.bindVertexArray(vao);
        this.context.gl.deleteBuffer(current);
        this.context.gl.disableVertexAttribArray(location);
        this.context.gl.bindVertexArray(null);
      }
      return values;
    }
    if (typeof current === "object") {
      this.context.loadBuffer(
        WebGL2RenderingContext.ARRAY_BUFFER,
        current,
        values,
      );
      return current;
    }
    const buffer = this.context.createBuffer();
    this.context.gl.bindVertexArray(vao);
    if (type === WebGL2RenderingContext.FLOAT) {
      this.context.configureVertexFloatAttribute(
        WebGL2RenderingContext.ARRAY_BUFFER,
        buffer,
        location,
        1,
        type,
      );
    } else {
      this.context.configureVertexIntAttribute(
        WebGL2RenderingContext.ARRAY_BUFFER,
        buffer,
        location,
        1,
        type,
      );
    }
    this.context.gl.bindVertexArray(null);
    this.context.allocateBuffer(
      WebGL2RenderingContext.ARRAY_BUFFER,
      buffer,
      values,
      WebGL2RenderingContext.STATIC_DRAW,
    );
    return buffer;
  }

  /**
   * Computes the factor that converts the sizes of an object to world units
   *
   * Multiplies the object- and layer-level point size factors with the
   * transform scales that the unit of the size configuration is subject to:
   * sizes in data units scale with the object and layer transforms, sizes in
   * layer units with the layer transform only, and sizes in world units with
   * neither.
   *
   * @param layer - The layer the points are drawn on, as of the current model
   * @param points - The points object being drawn, as of the current model
   */
  private static _computePointSizeFactor(layer: Layer, points: Points): number {
    const activeUnit =
      ConfigUtils.getUnit(points.pointSize) ?? defaultPointSizeUnit;
    let sizeFactor = points.pointSizeFactor * layer.pointSizeFactor;
    if (activeUnit === "data") {
      sizeFactor *= points.transform.scale;
    }
    if (activeUnit === "data" || activeUnit === "layer") {
      sizeFactor *= layer.transform.scale;
    }
    return sizeFactor;
  }

  /**
   * Copies a size configuration without its units
   *
   * The unit only affects {@link _computePointSizeFactor}, not the resolved
   * sizes, so it is left out of the snapshot that
   * {@link _checkPointSizeBufferChanged} compares.
   */
  private static _stripSizeUnit(config: SizeConfig): SizeConfig {
    const stripped = structuredClone(config);
    if (isConstantConfig<number, { unit?: CoordinateSpace }>(stripped)) {
      delete stripped.constant.unit;
    }
    if (isFromConfig<{ unit?: CoordinateSpace }>(stripped)) {
      delete stripped.from.unit;
    }
    if (isGroupByConfig<true, { unit?: CoordinateSpace }>(stripped)) {
      delete stripped.groupBy.unit;
    }
    return stripped;
  }

  /**
   * Captures what the buffers of an object are resolved from
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
    newRef: PointsRef,
    syncContext: WebGLPointsSyncContext,
  ): RenderedPoints["renderConfigSnapshot"] {
    return {
      pointMarker: newRef.object.pointMarker,
      pointSize: WebGLPointsRenderer._stripSizeUnit(newRef.object.pointSize),
      pointColor: newRef.object.pointColor,
      pointVisibility: newRef.object.pointVisibility,
      pointOpacity: newRef.object.pointOpacity,
      pointMarkerMap: ConfigUtils.findGroupByMap(
        newRef.object.pointMarker,
        syncContext.markerMaps,
      ),
      pointSizeMap: ConfigUtils.findGroupByMap(
        newRef.object.pointSize,
        syncContext.sizeMaps,
      ),
      pointColorMap: ConfigUtils.findGroupByMap(
        newRef.object.pointColor,
        syncContext.colorMaps,
      ),
      pointVisibilityMap: ConfigUtils.findGroupByMap(
        newRef.object.pointVisibility,
        syncContext.visibilityMaps,
      ),
      pointOpacityMap: ConfigUtils.findGroupByMap(
        newRef.object.pointOpacity,
        syncContext.opacityMaps,
      ),
    };
  }

  /**
   * Returns whether the markers of an object have to be resolved again
   *
   * Also true for an object that has not been rendered yet, like the other
   * predicates. Configurations are compared by value, maps by identity (see
   * {@link _createRenderConfigSnapshot}).
   */
  private static _checkPointMarkerBufferChanged(
    renderedPoints: RenderedPoints | undefined,
    newSnapshot: RenderedPoints["renderConfigSnapshot"],
  ): boolean {
    return (
      renderedPoints === undefined ||
      !deepEqual(
        renderedPoints.renderConfigSnapshot.pointMarker,
        newSnapshot.pointMarker,
      ) ||
      renderedPoints.renderConfigSnapshot.pointMarkerMap !==
        newSnapshot.pointMarkerMap
    );
  }

  /**
   * Returns whether the sizes of an object have to be resolved again
   *
   * The point size factors, transform scales and size units are shader
   * uniforms (see {@link _computePointSizeFactor}), so only the rest of the
   * point size configuration matters here.
   */
  private static _checkPointSizeBufferChanged(
    renderedPoints: RenderedPoints | undefined,
    newSnapshot: RenderedPoints["renderConfigSnapshot"],
  ): boolean {
    return (
      renderedPoints === undefined ||
      !deepEqual(
        renderedPoints.renderConfigSnapshot.pointSize,
        newSnapshot.pointSize,
      ) ||
      renderedPoints.renderConfigSnapshot.pointSizeMap !==
        newSnapshot.pointSizeMap
    );
  }

  /**
   * Returns whether the colors of an object have to be resolved again
   *
   * Colors carry the resolved point visibilities and opacities in their alpha
   * channel, so they also depend on those configurations. The layer- and
   * object-level visibility and opacity are shader uniforms (see
   * {@link WebGLRendererBase.computeOpacityFactor}) and do not matter here.
   */
  private static _checkPointColorBufferChanged(
    renderedPoints: RenderedPoints | undefined,
    newSnapshot: RenderedPoints["renderConfigSnapshot"],
  ): boolean {
    return (
      renderedPoints === undefined ||
      !deepEqual(
        renderedPoints.renderConfigSnapshot.pointColor,
        newSnapshot.pointColor,
      ) ||
      renderedPoints.renderConfigSnapshot.pointColorMap !==
        newSnapshot.pointColorMap ||
      !deepEqual(
        renderedPoints.renderConfigSnapshot.pointVisibility,
        newSnapshot.pointVisibility,
      ) ||
      renderedPoints.renderConfigSnapshot.pointVisibilityMap !==
        newSnapshot.pointVisibilityMap ||
      !deepEqual(
        renderedPoints.renderConfigSnapshot.pointOpacity,
        newSnapshot.pointOpacity,
      ) ||
      renderedPoints.renderConfigSnapshot.pointOpacityMap !==
        newSnapshot.pointOpacityMap
    );
  }

  /**
   * Resolves the packed RGBA color all points share if the color, visibility
   * and opacity configurations are all constants, `undefined` otherwise
   *
   * The three fold into one attribute (see {@link ColorUtils.withAlpha}), so
   * it is only constant if all of them are.
   */
  private static _resolveConstantColor(
    colorConfig: ColorConfig,
    visibilityConfig: VisibilityConfig,
    opacityConfig: OpacityConfig,
  ): number | undefined {
    const packedColor = ColorResolver.resolveConstantColor(colorConfig);
    const packedVisibility =
      VisibilityResolver.resolveConstantVisibility(visibilityConfig);
    const packedOpacity = OpacityResolver.resolveConstantOpacity(opacityConfig);
    if (
      packedColor === undefined ||
      packedVisibility === undefined ||
      packedOpacity === undefined
    ) {
      return undefined;
    }
    return ColorUtils.withAlpha(packedColor, packedVisibility, packedOpacity);
  }

  /**
   * Computes the axis-aligned bounding box of the given points in data coordinates
   *
   * @param geometry - X and Y coordinates of the points, in data coordinates
   * @param options - Optional abort signal
   * @returns The axis-aligned bounding box of the points in data coordinates
   * @throws Error if the coordinate arrays are empty or have different lengths
   */
  private static async _getObjectBounds(
    geometry: PointsGeometry,
    options?: { signal?: AbortSignal },
  ): Promise<Rect> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const { xs, ys } = geometry;
    if (xs.length === 0 || ys.length === 0) {
      throw new Error("Coordinate arrays must not be empty");
    }
    if (xs.length !== ys.length) {
      throw new Error("Coordinate arrays must have the same length");
    }
    const [xMin, xMax] = await MathUtils.computeRange(xs, { signal });
    const [yMin, yMax] = await MathUtils.computeRange(ys, { signal });
    return { x: xMin, y: yMin, width: xMax - xMin, height: yMax - yMin };
  }
}

/**
 * The inputs of a synchronization of the points renderer
 *
 * The tables and group-to-value maps that the objects resolve their
 * properties from, and the loaders for points and table data (see
 * {@link WebGLRendererBase.synchronize}).
 */
export type WebGLPointsSyncContext = {
  tables: Table[];
  markerMaps: GroupValueMap<Marker>[];
  sizeMaps: GroupValueMap<number>[];
  colorMaps: GroupValueMap<Color>[];
  visibilityMaps: GroupValueMap<boolean>[];
  opacityMaps: GroupValueMap<number>[];
  loadObject: (
    points: Points,
    options?: { signal?: AbortSignal },
  ) => Promise<PointsData>;
  loadTable: (
    table: Table,
    options?: { signal?: AbortSignal },
  ) => Promise<TableData>;
};

/**
 * A reference to a points object, its layer, and its loaded data
 */
type PointsRef = ObjectRef<Points, PointsData>;

/**
 * What a preparation of a points object uploads
 *
 * See {@link WebGLPointsRenderer.prepareRenderedObject}.
 */
type PreparedPoints = {
  objectBounds: Rect;
  maskedGeometry: PointsGeometry | undefined;
  packedPointMarkers: Uint8Array | number | undefined;
  packedPointSizes: Float32Array | number | undefined;
  packedPointColors: Uint32Array | number | undefined;
  renderConfigSnapshot: RenderedPoints["renderConfigSnapshot"];
};

/**
 * GPU state for a single points object
 *
 * Holds the vertex array, the coordinate buffers it is configured with, and
 * the per-point attributes, each backed by a buffer holding one value per
 * point or by one constant value for all points (see
 * {@link WebGLPointsRenderer._backAttribute}), plus a snapshot of the model
 * values they were loaded from, which the change predicates compare against
 * (see {@link WebGLPointsRenderer._createRenderConfigSnapshot}). Layer- and
 * object-level properties are read from the current model when drawing (see
 * {@link WebGLRendererBase.getRenderPasses}), and are not part of the snapshot.
 */
type RenderedPoints = RenderedObjectBase<Points, PointsData> & {
  vao: WebGLVertexArrayObject;
  buffers: {
    x: WebGLBuffer;
    y: WebGLBuffer;
  };
  attributes: {
    marker: WebGLBuffer | number;
    size: WebGLBuffer | number;
    color: WebGLBuffer | number;
  };
  renderConfigSnapshot: Pick<
    Points,
    | "pointMarker"
    | "pointSize"
    | "pointColor"
    | "pointVisibility"
    | "pointOpacity"
  > & {
    pointMarkerMap: GroupValueMap<Marker> | undefined;
    pointSizeMap: GroupValueMap<number> | undefined;
    pointColorMap: GroupValueMap<Color> | undefined;
    pointVisibilityMap: GroupValueMap<boolean> | undefined;
    pointOpacityMap: GroupValueMap<number> | undefined;
  };
};
