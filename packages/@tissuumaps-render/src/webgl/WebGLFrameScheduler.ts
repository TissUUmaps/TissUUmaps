import { type Dims, GeometryUtils, type Rect } from "@tissuumaps/core";

import type { WebGLContext } from "./WebGLContext";

/** A renderer that draws for a viewport, see {@link WebGLFrameScheduler} */
export interface WebGLRenderer {
  /**
   * The viewport to draw, set before every {@link draw}. Until the next redraw,
   * the viewport on screen can differ.
   */
  viewport: Rect;
  /** Draws onto the canvas, without clearing it */
  draw(): void;
}

/**
 * Redraws renderers into an overlay canvas at a pace the GPU can sustain
 *
 * Drawing a frame with many points or shapes can take longer on the GPU than a
 * display frame. Since the GPU executes commands in order, and the browser
 * composites the page only once all of them are done, redrawing on every
 * viewport change stalls everything else on screen - notably the OpenSeadragon
 * canvas underneath.
 *
 * Two viewports are therefore kept apart: the displayed viewport, i.e. the
 * latest one set by {@link setViewport}, and the rendered viewport, i.e. the
 * one the canvas pixels were drawn for. Between redraws, viewport changes only
 * move the existing pixels into place with a CSS transform on the canvas, which
 * costs nothing on the GPU. A redraw resets the transform in the same task, so
 * that both land in the same composited frame.
 *
 * Redraws are paced by the GPU: after a redraw, the scheduler waits for the GPU
 * to finish it (a fence), and then waits as long again before the next one, so
 * that the GPU is at most half busy with the overlay. The fence is checked once
 * per frame, so even a light overlay is only redrawn about every other frame,
 * which leaves OpenSeadragon a free frame in between.
 */
export class WebGLFrameScheduler {
  // waiting longer for a fence means that the page was hidden or the main
  // thread busy, rather than the GPU still drawing (ms)
  private static readonly _maxDrawDuration = 500;

  private readonly _context: WebGLContext;
  private readonly _canvas: HTMLCanvasElement;
  private readonly _renderers: WebGLRenderer[];
  private _displayedViewport: Rect | null = null;
  private _renderedViewport: Rect | null = null;
  private _containerSize: Dims | null = null;
  private _dirty = true;
  private _sync: WebGLSync | null = null;
  private _drawTime = -Infinity;
  private _drawDuration = 0;
  private _animationFrameId: number | null = null;

  /**
   * Creates a new WebGLFrameScheduler
   *
   * @param context - The WebGL context to draw with
   * @param canvas - The canvas of the context, resized and moved into place by
   * the scheduler
   * @param renderers - The renderers to draw, in order
   */
  constructor(
    context: WebGLContext,
    canvas: HTMLCanvasElement,
    renderers: WebGLRenderer[],
  ) {
    this._context = context;
    this._canvas = canvas;
    this._renderers = renderers;
    canvas.style.transformOrigin = "0 0";
  }

  /**
   * Sets the displayed viewport
   *
   * Redraws right away if the GPU is not busy with the overlay, and otherwise
   * moves the drawn pixels into place until the next redraw.
   *
   * @param viewport - The viewport to display, in world coordinates
   */
  setViewport(viewport: Rect): void {
    this._displayedViewport = viewport;
    this._updateTransform();
    this._drawWhenReady();
  }

  /**
   * Sets the size of the container that the canvas covers
   *
   * The canvas is resized right before the next redraw, since resizing clears
   * it. Until then, the drawn pixels are stretched into place.
   *
   * @param containerSize - The container size in screen-space pixels
   */
  setContainerSize(containerSize: Dims): void {
    this._containerSize = containerSize;
    this.invalidate();
  }

  /**
   * Schedules a redraw, for when the rendered content changed
   */
  invalidate(): void {
    this._dirty = true;
    this._schedule();
  }

  /**
   * Cancels the scheduled redraw, deletes the pending fence and resets the
   * canvas transform
   */
  destroy(): void {
    if (this._animationFrameId !== null) {
      cancelAnimationFrame(this._animationFrameId);
      this._animationFrameId = null;
    }
    if (this._sync !== null) {
      this._context.gl.deleteSync(this._sync);
      this._sync = null;
    }
    this._canvas.style.transform = "";
  }

  private _schedule(): void {
    if (this._animationFrameId === null) {
      this._animationFrameId = requestAnimationFrame(() => this._tick());
    }
  }

  private _tick(): void {
    this._animationFrameId = null;
    const gl = this._context.gl;
    if (this._sync !== null) {
      const signaled =
        gl.getSyncParameter(this._sync, WebGL2RenderingContext.SYNC_STATUS) ===
        WebGL2RenderingContext.SIGNALED;
      if (!signaled) {
        this._schedule();
        return;
      }
      gl.deleteSync(this._sync);
      this._sync = null;
      this._drawDuration = Math.min(
        performance.now() - this._drawTime,
        WebGLFrameScheduler._maxDrawDuration,
      );
    }
    this._drawWhenReady();
  }

  /**
   * Redraws if something changed and the GPU is ready, and otherwise tries
   * again in the next frame
   */
  private _drawWhenReady(): void {
    const viewport = this._displayedViewport;
    if (
      viewport === null ||
      (!this._dirty &&
        this._renderedViewport !== null &&
        GeometryUtils.rectEquals(viewport, this._renderedViewport))
    ) {
      return;
    }
    // with the fence polled once per frame, waiting as long again also keeps a
    // light overlay to about every other frame
    if (
      this._sync !== null ||
      performance.now() - this._drawTime < 2 * this._drawDuration
    ) {
      this._schedule();
      return;
    }
    this._draw(viewport);
  }

  private _draw(viewport: Rect): void {
    const gl = this._context.gl;
    if (this._containerSize !== null) {
      this._context.resizeCanvas(this._canvas, this._containerSize);
    }
    for (const renderer of this._renderers) {
      renderer.viewport = viewport;
    }
    this._context.clear();
    for (const renderer of this._renderers) {
      renderer.draw();
    }
    this._sync = gl.fenceSync(
      WebGL2RenderingContext.SYNC_GPU_COMMANDS_COMPLETE,
      0,
    );
    gl.flush();
    this._drawTime = performance.now();
    this._renderedViewport = viewport;
    this._dirty = false;
    this._updateTransform();
    this._schedule();
  }

  /**
   * Moves the rendered pixels to where the displayed viewport shows them
   *
   * The transform is expressed relative to the canvas size, so no layout is
   * read. The scale is non-uniform only after a container resize, until the
   * next redraw.
   */
  private _updateTransform(): void {
    const rendered = this._renderedViewport;
    const displayed = this._displayedViewport;
    if (
      rendered === null ||
      displayed === null ||
      GeometryUtils.rectEquals(rendered, displayed)
    ) {
      this._canvas.style.transform = "";
      return;
    }
    const sx = rendered.width / displayed.width;
    const sy = rendered.height / displayed.height;
    const tx = ((rendered.x - displayed.x) / displayed.width) * 100;
    const ty = ((rendered.y - displayed.y) / displayed.height) * 100;
    this._canvas.style.transform = `translate(${tx}%, ${ty}%) scale(${sx}, ${sy})`;
  }
}
