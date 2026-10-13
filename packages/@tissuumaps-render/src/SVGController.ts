import {
  type Dims,
  GeometryUtils,
  InteractionMode,
  type MultiPolygon,
  type Rect,
  type Vertex,
} from "@tissuumaps/core";

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";

type RectangleDrawingState = {
  startPoint: Vertex;
  currentRect: SVGRectElement;
};

type PolygonDrawingState = {
  pointerId: number;
  vertices: Vertex[];
  currentPolyline: SVGPolylineElement;
  vertexMarkers: SVGCircleElement[];
  isNearStart: boolean;
};

type FreehandDrawingState = {
  pointerId: number;
  points: Vertex[];
  currentPolyline: SVGPolylineElement;
  startHandle: SVGCircleElement;
  hasLeftStart: boolean;
  isNearStart: boolean;
};

/**
 * Controller for managing the drawing of SVG shapes
 */
export class SVGController {
  public readonly container: SVGSVGElement;
  public readonly transformNode: SVGGElement;
  public readonly shapeCompleteHandler?: (shape: MultiPolygon) => void;
  private static readonly _previewFillColor = "rgba(255, 0, 0, 0.2)";
  private static readonly _previewStrokeColor = "#FF0000";
  private static readonly _previewStrokeWidthFactor = 0.0005;
  private static readonly _minShapeSizeFactor = 0.001;
  private static readonly _closeShapeThresholdFactor = 0.01;
  private static readonly _vertexMarkerRadiusFactor = 0.0015;
  private _containerSize: Dims;
  private _viewport: Rect;
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore currently not used, but will be needed in the future
  private _interactionMode?: InteractionMode;

  private _rectangleDrawingState: RectangleDrawingState | null = null;

  private _polygonDrawingState: PolygonDrawingState | null = null;

  private _freehandDrawingState: FreehandDrawingState | null = null;

  /** Creates a positioned, full-size `<svg>` element for the SVG overlay */
  static createContainer(): SVGSVGElement {
    const container = document.createElementNS(SVG_NAMESPACE, "svg");
    container.style.position = "absolute";
    container.style.top = "0";
    container.style.left = "0";
    container.style.width = "100%";
    container.style.height = "100%";
    return container;
  }

  /**
   * Creates the transform node inside the given container and registers the
   * shape drawing event handlers
   *
   * The viewport starts out as the unit square; {@link setViewport} is the only
   * way to change it.
   *
   * @param container - The `<svg>` element to draw on (typically created by {@link createContainer})
   * @param options - Optional shape drawing event handlers
   */
  constructor(
    container: SVGSVGElement,
    options?: { onShapeComplete?: (shape: MultiPolygon) => void },
  ) {
    this.container = container;
    this.transformNode = document.createElementNS(SVG_NAMESPACE, "g");
    this.shapeCompleteHandler = options?.onShapeComplete;
    container.replaceChildren(this.transformNode);
    this._containerSize = container.getBoundingClientRect();
    this._viewport = { x: 0, y: 0, width: 1, height: 1 };
    this._updateTransformNode();

    this._registerEventHandlers();
  }

  /**
   * Updates the world-space viewport rectangle
   *
   * @param viewport - New viewport bounds in world coordinates
   * @returns `true` if the viewport actually changed, `false` otherwise
   */
  setViewport(viewport: Rect): boolean {
    if (!GeometryUtils.rectEquals(viewport, this._viewport)) {
      this._viewport = viewport;
      this._updateTransformNode();
      return true;
    }
    return false;
  }

  /**
   * Updates the interaction mode
   *
   * @param newInteractionMode - New interaction mode
   */
  setInteractionMode(newInteractionMode: InteractionMode) {
    if (
      this._freehandDrawingState &&
      newInteractionMode !== InteractionMode.drawFreehand
    ) {
      this._cancelFreehand();
    }
    if (
      this._polygonDrawingState &&
      newInteractionMode !== InteractionMode.drawPolygon
    ) {
      this._cancelPolygon();
    }
    this._interactionMode = newInteractionMode;
  }

  /**
   * Resizes the SVG container to match the given screen-space dimensions,
   * accounting for `devicePixelRatio`
   *
   * @param newContainerSize - Desired container size in screen-space pixels
   * @returns `true` if the container size actually changed, `false` otherwise
   */
  resizeContainer(newContainerSize: Dims): boolean {
    let { width, height } = newContainerSize;
    width *= window.devicePixelRatio;
    height *= window.devicePixelRatio;
    if (width <= 0 || height <= 0) {
      width = 1;
      height = 1;
    }
    if (
      this._containerSize.width !== width ||
      this._containerSize.height !== height
    ) {
      this._containerSize = { width, height };
      this.container.setAttribute("width", width.toString());
      this.container.setAttribute("height", height.toString());
      this._updateTransformNode();
      return true;
    }
    return false;
  }

  destroy(): void {
    this._cancelFreehand();
    this._cancelPolygon();
    this._unregisterEventHandlers();
  }

  // ─────────────────────────────────────────────────────────────
  // Event Handling
  // ─────────────────────────────────────────────────────────────

  private _registerEventHandlers(): void {
    this.container.addEventListener("pointerdown", this._handlePointerDown);
  }

  private _unregisterEventHandlers(): void {
    this.container.removeEventListener("pointerdown", this._handlePointerDown);
  }

  // Main handler for pointer down events; delegates to specific handlers based on the current interaction mode
  private _handlePointerDown = (event: PointerEvent): void => {
    if (event.shiftKey || event.button !== 0) return;

    switch (this._interactionMode) {
      case InteractionMode.drawRectangle:
        this._handleRectanglePointerDown(event);
        break;
      case InteractionMode.drawPolygon:
        this._handlePolygonPointerDown(event);
        break;
      case InteractionMode.drawFreehand:
        this._handleFreehandPointerDown(event);
        break;
    }
  };

  // ─────────────────────────────────────────────────────────────
  // Rectangle Drawing
  // ─────────────────────────────────────────────────────────────

  private _handleRectanglePointerDown = (event: PointerEvent): void => {
    document.addEventListener("pointermove", this._handleRectanglePointerMove);
    document.addEventListener("pointerup", this._handleRectanglePointerUp);

    event.preventDefault();
    event.stopPropagation();

    const worldPoint = this._screenToWorld(event.clientX, event.clientY);

    this._rectangleDrawingState = {
      startPoint: worldPoint,
      currentRect: this._createPreviewRect(worldPoint),
    };
  };

  private _handleRectanglePointerMove = (event: PointerEvent): void => {
    if (!this._rectangleDrawingState) return;

    const worldPoint = this._screenToWorld(event.clientX, event.clientY);
    this._updatePreviewRect(
      this._rectangleDrawingState.currentRect,
      this._rectangleDrawingState.startPoint,
      worldPoint,
    );
  };

  private _handleRectanglePointerUp = (event: PointerEvent): void => {
    if (!this._rectangleDrawingState || event.button !== 0) return;

    const worldPoint = this._screenToWorld(event.clientX, event.clientY);

    if (this._rectangleDrawingState.startPoint) {
      const multiPolygon = this._createRectangleMultiPolygon(
        this._rectangleDrawingState.startPoint,
        worldPoint,
      );

      if (
        this._hasMinimumSize(this._rectangleDrawingState.startPoint, worldPoint)
      ) {
        this.shapeCompleteHandler?.(multiPolygon);
      }
    }

    // Remove preview element - WebGL will render the actual shape
    this._rectangleDrawingState.currentRect.remove();
    this._rectangleDrawingState = null;

    document.removeEventListener(
      "pointermove",
      this._handleRectanglePointerMove,
    );
    document.removeEventListener("pointerup", this._handleRectanglePointerUp);
  };

  // ─────────────────────────────────────────────────────────────
  // Polygon drawing
  // ─────────────────────────────────────────────────────────────

  private _handlePolygonPointerDown = (event: PointerEvent): void => {
    event.preventDefault();
    event.stopPropagation();

    const worldPoint = this._screenToWorld(event.clientX, event.clientY);

    if (!this._polygonDrawingState) {
      document.addEventListener("pointermove", this._handlePolygonPointerMove);
      document.addEventListener("keydown", this._handlePolygonKeyDown);
      document.addEventListener(
        "pointercancel",
        this._handlePolygonPointerCancel,
      );

      this._polygonDrawingState = {
        pointerId: event.pointerId,
        vertices: [worldPoint],
        currentPolyline: this._createPreviewPolyline(worldPoint),
        vertexMarkers: [this._createVertexMarker(worldPoint)],
        isNearStart: false,
      };
    } else {
      const firstVertex = this._polygonDrawingState.vertices[0]!;
      if (
        this._polygonDrawingState.vertices.length >= 3 &&
        this._isNearPoint(worldPoint, firstVertex)
      ) {
        this._completePolygon();
        return;
      }

      this._polygonDrawingState.vertices.push(worldPoint);
      this._polygonDrawingState.vertexMarkers.push(
        this._createVertexMarker(worldPoint),
      );

      this._updatePolygonPreviewPolyline(this._polygonDrawingState.vertices);
    }
  };

  private _handlePolygonPointerMove = (event: PointerEvent): void => {
    if (!this._polygonDrawingState) return;

    const worldPoint = this._screenToWorld(event.clientX, event.clientY);
    const firstVertex = this._polygonDrawingState.vertices[0]!;

    const isNearStart =
      this._polygonDrawingState.vertices.length >= 3 &&
      this._isNearPoint(worldPoint, firstVertex);

    if (isNearStart !== this._polygonDrawingState.isNearStart) {
      this._polygonDrawingState.isNearStart = isNearStart;
      this._updateFirstVertexMarker(isNearStart);
    }

    this._updatePolygonPreviewPolyline([
      ...this._polygonDrawingState.vertices,
      worldPoint,
    ]);
  };

  private _handlePolygonKeyDown = (event: KeyboardEvent): void => {
    if (!this._polygonDrawingState) return;

    if (event.key !== "Escape" && event.key !== "Enter") return;

    event.preventDefault();
    event.stopPropagation();

    if (event.key === "Escape") {
      this._cancelPolygon();
    } else if (this._polygonDrawingState.vertices.length >= 3) {
      this._completePolygon();
    }
  };

  private _handlePolygonPointerCancel = (event: PointerEvent): void => {
    if (
      !this._polygonDrawingState ||
      event.pointerId !== this._polygonDrawingState.pointerId
    )
      return;
    this._cancelPolygon();
  };

  // ─────────────────────────────────────────────────────────────
  // Freehand Drawing
  // ─────────────────────────────────────────────────────────────

  private _handleFreehandPointerDown = (event: PointerEvent): void => {
    if (this._freehandDrawingState) return;

    event.preventDefault();
    event.stopPropagation();

    document.addEventListener("pointermove", this._handleFreehandPointerMove);
    document.addEventListener("pointerup", this._handleFreehandPointerUp);
    document.addEventListener(
      "pointercancel",
      this._handleFreehandPointerCancel,
    );

    const worldPoint = this._screenToWorld(event.clientX, event.clientY);

    this._freehandDrawingState = {
      pointerId: event.pointerId,
      points: [worldPoint],
      currentPolyline: this._createPreviewPolyline(worldPoint),
      startHandle: this._createVertexMarker(worldPoint),
      hasLeftStart: false,
      isNearStart: false,
    };
  };

  private _handleFreehandPointerMove = (event: PointerEvent): void => {
    if (
      !this._freehandDrawingState ||
      event.pointerId !== this._freehandDrawingState.pointerId
    ) {
      return;
    }

    const worldPoint = this._screenToWorld(event.clientX, event.clientY);
    this._freehandDrawingState.points.push(worldPoint);

    const firstPoint = this._freehandDrawingState.points[0]!;
    const nearStart = this._isNearPoint(worldPoint, firstPoint);

    if (!this._freehandDrawingState.hasLeftStart && !nearStart) {
      this._freehandDrawingState.hasLeftStart = true;
    }

    const isNearStart = this._freehandDrawingState.hasLeftStart && nearStart;
    if (isNearStart !== this._freehandDrawingState.isNearStart) {
      this._freehandDrawingState.isNearStart = isNearStart;
      this._setMarkerHighlighted(
        this._freehandDrawingState.startHandle,
        isNearStart,
      );
    }

    const existingPoints =
      this._freehandDrawingState.currentPolyline.getAttribute("points");
    this._freehandDrawingState.currentPolyline.setAttribute(
      "points",
      `${existingPoints} ${worldPoint.x},${worldPoint.y}`,
    );
  };

  private _handleFreehandPointerUp = (event: PointerEvent): void => {
    if (
      !this._freehandDrawingState ||
      event.button !== 0 ||
      event.pointerId !== this._freehandDrawingState.pointerId
    ) {
      return;
    }

    const releasePoint = this._screenToWorld(event.clientX, event.clientY);
    this._freehandDrawingState.points.push(releasePoint);

    const points = this._freehandDrawingState.points;
    const firstPoint = points[0]!;

    if (
      points.length >= 2 &&
      this._freehandDrawingState.hasLeftStart &&
      this._isNearPoint(releasePoint, firstPoint)
    ) {
      const multiPolygon = this._createMultiPolygon([...points, firstPoint]);
      this.shapeCompleteHandler?.(multiPolygon);
    }

    this._cancelFreehand();
  };

  private _handleFreehandPointerCancel = (event: PointerEvent): void => {
    if (
      !this._freehandDrawingState ||
      event.pointerId !== this._freehandDrawingState.pointerId
    ) {
      return;
    }
    this._cancelFreehand();
  };

  private _cancelFreehand(): void {
    if (!this._freehandDrawingState) return;

    this._freehandDrawingState.currentPolyline.remove();
    this._freehandDrawingState.startHandle.remove();
    this._freehandDrawingState = null;

    document.removeEventListener(
      "pointermove",
      this._handleFreehandPointerMove,
    );
    document.removeEventListener("pointerup", this._handleFreehandPointerUp);
    document.removeEventListener(
      "pointercancel",
      this._handleFreehandPointerCancel,
    );
  }

  // ─────────────────────────────────────────────────────────────
  // Rectangle Drawing Helpers
  // ─────────────────────────────────────────────────────────────

  private _createPreviewRect(startPoint: Vertex): SVGRectElement {
    const rect = document.createElementNS(SVG_NAMESPACE, "rect");
    rect.setAttribute("x", startPoint.x.toString());
    rect.setAttribute("y", startPoint.y.toString());
    rect.setAttribute("width", "0");
    rect.setAttribute("height", "0");
    rect.setAttribute("fill", SVGController._previewFillColor);
    rect.setAttribute("stroke", SVGController._previewStrokeColor);
    rect.setAttribute(
      "stroke-width",
      (
        this._viewport.width * SVGController._previewStrokeWidthFactor
      ).toString(),
    );

    this.transformNode.appendChild(rect);
    return rect;
  }

  private _updatePreviewRect(
    rect: SVGRectElement,
    start: Vertex,
    end: Vertex,
  ): void {
    const x = Math.min(start.x, end.x);
    const y = Math.min(start.y, end.y);
    const width = Math.abs(end.x - start.x);
    const height = Math.abs(end.y - start.y);

    rect.setAttribute("x", x.toString());
    rect.setAttribute("y", y.toString());
    rect.setAttribute("width", width.toString());
    rect.setAttribute("height", height.toString());
  }

  private _createRectangleMultiPolygon(
    start: Vertex,
    end: Vertex,
  ): MultiPolygon {
    const x0 = Math.min(start.x, end.x);
    const y0 = Math.min(start.y, end.y);
    const x1 = Math.max(start.x, end.x);
    const y1 = Math.max(start.y, end.y);

    return {
      polygons: [
        {
          shell: [
            { x: x0, y: y0 },
            { x: x1, y: y0 },
            { x: x1, y: y1 },
            { x: x0, y: y1 },
          ],
          holes: [],
        },
      ],
    };
  }

  private _hasMinimumSize(start: Vertex, end: Vertex): boolean {
    const minSize = this._viewport.width * SVGController._minShapeSizeFactor; // Adjust threshold as needed
    return (
      Math.abs(end.x - start.x) > minSize && Math.abs(end.y - start.y) > minSize
    );
  }

  // ─────────────────────────────────────────────────────────────
  // Polygon Drawing Helpers
  // ─────────────────────────────────────────────────────────────

  private _updatePolygonPreviewPolyline(points: Vertex[]): void {
    if (!this._polygonDrawingState) return;

    const pointsString = points.map((v) => `${v.x},${v.y}`).join(" ");
    this._polygonDrawingState.currentPolyline.setAttribute(
      "points",
      pointsString,
    );
  }

  private _updateFirstVertexMarker(isActive: boolean): void {
    if (!this._polygonDrawingState) return;

    const firstMarker = this._polygonDrawingState.vertexMarkers[0];
    if (!firstMarker) return;

    this._setMarkerHighlighted(firstMarker, isActive);
  }

  private _completePolygon(): void {
    if (!this._polygonDrawingState) return;

    const multiPolygon = this._createMultiPolygon(
      this._polygonDrawingState.vertices,
    );
    this.shapeCompleteHandler?.(multiPolygon);

    this._cancelPolygon();
  }

  private _cancelPolygon(): void {
    if (!this._polygonDrawingState) return;

    this._polygonDrawingState.currentPolyline.remove();
    for (const marker of this._polygonDrawingState.vertexMarkers) {
      marker.remove();
    }
    this._polygonDrawingState = null;

    document.removeEventListener("pointermove", this._handlePolygonPointerMove);
    document.removeEventListener("keydown", this._handlePolygonKeyDown);
    document.removeEventListener(
      "pointercancel",
      this._handlePolygonPointerCancel,
    );
  }

  // ─────────────────────────────────────────────────────────────
  // Freehand Drawing Helpers
  // ─────────────────────────────────────────────────────────────

  private _createPreviewPolyline(startPoint: Vertex): SVGPolylineElement {
    const polyline = document.createElementNS(SVG_NAMESPACE, "polyline");
    polyline.setAttribute("points", `${startPoint.x},${startPoint.y}`);
    polyline.setAttribute("fill", "none");
    polyline.setAttribute("stroke", SVGController._previewStrokeColor);
    polyline.setAttribute(
      "stroke-width",
      (
        this._viewport.width * SVGController._previewStrokeWidthFactor
      ).toString(),
    );

    this.transformNode.appendChild(polyline);
    return polyline;
  }

  private _createVertexMarker(point: Vertex): SVGCircleElement {
    const circle = document.createElementNS(SVG_NAMESPACE, "circle");
    const radius =
      this._viewport.width * SVGController._vertexMarkerRadiusFactor;
    circle.setAttribute("cx", point.x.toString());
    circle.setAttribute("cy", point.y.toString());
    circle.setAttribute("r", radius.toString());
    circle.setAttribute("fill", SVGController._previewStrokeColor);

    this.transformNode.appendChild(circle);
    return circle;
  }

  private _setMarkerHighlighted(
    marker: SVGCircleElement,
    isHighlighted: boolean,
  ): void {
    const radius =
      this._viewport.width * SVGController._vertexMarkerRadiusFactor;

    if (isHighlighted) {
      marker.setAttribute("fill", "none");
      marker.setAttribute("stroke", SVGController._previewStrokeColor);
      marker.setAttribute(
        "stroke-width",
        (
          this._viewport.width *
          SVGController._previewStrokeWidthFactor *
          2
        ).toString(),
      );
      marker.setAttribute("r", (radius * 1.5).toString());
    } else {
      marker.setAttribute("fill", SVGController._previewStrokeColor);
      marker.setAttribute("stroke", "none");
      marker.setAttribute("r", radius.toString());
    }
  }

  private _createMultiPolygon(vertices: Vertex[]): MultiPolygon {
    return {
      polygons: [
        {
          shell: vertices.map((v) => ({ x: v.x, y: v.y })),
          holes: [],
        },
      ],
    };
  }

  private _isNearPoint(p1: Vertex, p2: Vertex): boolean {
    const threshold =
      this._viewport.width * SVGController._closeShapeThresholdFactor;
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    return dx * dx + dy * dy < threshold * threshold;
  }

  // ─────────────────────────────────────────────────────────────
  // Coordinate Transformation
  // ─────────────────────────────────────────────────────────────

  private _screenToWorld(screenX: number, screenY: number): Vertex {
    const ctm = this.transformNode.getScreenCTM();
    if (!ctm) {
      throw new Error("Could not get screen CTM");
    }

    const inverse = ctm.inverse();
    const svgPoint = this.container.createSVGPoint();
    svgPoint.x = screenX;
    svgPoint.y = screenY;

    const worldPoint = svgPoint.matrixTransform(inverse);
    return { x: worldPoint.x, y: worldPoint.y };
  }

  /**
   * Updates the world-to-viewport transform based on the current container size
   * (in screen coordinates) and the current viewport (in world coordinates)
   */
  private _updateTransformNode() {
    const sx = this._containerSize.width / this._viewport.width;
    const sy = this._containerSize.height / this._viewport.height;
    const tx = -this._viewport.x * sx;
    const ty = -this._viewport.y * sy;
    this.transformNode.setAttribute(
      "transform",
      `translate(${tx} ${ty}) scale(${sx} ${sy})`,
    );
  }
}
