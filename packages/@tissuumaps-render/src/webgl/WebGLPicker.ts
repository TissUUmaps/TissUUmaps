import type { AnnotatedItemRef, Vertex } from "@tissuumaps/core";

import type { WebGLContext } from "./WebGLContext";
import type { WebGLPointsRenderer } from "./WebGLPointsRenderer";
import type { WebGLShapesRenderer } from "./WebGLShapesRenderer";

/**
 * Finds the point or shape that is visible at a pixel of the WebGL canvas
 *
 * Instead of hit-testing the items on the CPU, which would have to reproduce
 * what the shaders draw, the renderers draw them again into a small offscreen
 * framebuffer around the picked pixel, each item in the color of its pick ID
 * rather than in its own. The pixels are read back, and the ID nearest to the
 * picked pixel is resolved by the renderer that drew it. Reading back stalls
 * the GPU until everything queued before it has been drawn, which is why this
 * is for clicks, not for hovering.
 *
 * A pick ID is a 32-bit unsigned integer, written as the four bytes of an
 * RGBA8 pixel. The renderers number their items consecutively, starting at 1,
 * so that `0` means that nothing was drawn.
 */
export class WebGLPicker {
  /** Search radius around the picked pixel, in device pixels */
  private static readonly _radius = 4;
  private static readonly _size = 2 * WebGLPicker._radius + 1;

  private readonly _context: WebGLContext;
  private readonly _renderers: readonly (
    WebGLShapesRenderer | WebGLPointsRenderer
  )[];
  private readonly _framebuffer: WebGLFramebuffer;
  private readonly _texture: WebGLTexture;
  private readonly _pixels = new Uint8Array(4 * WebGLPicker._size ** 2);

  /**
   * Creates the offscreen framebuffer that the renderers draw their pick IDs into
   *
   * @param context - The WebGL context to pick in
   * @param renderers - The renderers to pick from, in the order they are drawn
   * in, so that the item on top is the one that is picked
   * @throws Error if the framebuffer is incomplete
   */
  constructor(
    context: WebGLContext,
    renderers: readonly (WebGLShapesRenderer | WebGLPointsRenderer)[],
  ) {
    this._context = context;
    this._renderers = renderers;
    this._texture = context.createDataTexture(
      WebGL2RenderingContext.RGBA8,
      WebGLPicker._size,
      WebGLPicker._size,
      WebGL2RenderingContext.RGBA,
      WebGL2RenderingContext.UNSIGNED_BYTE,
    );
    this._framebuffer = context.createFramebuffer();
    context.gl.bindFramebuffer(
      WebGL2RenderingContext.FRAMEBUFFER,
      this._framebuffer,
    );
    context.gl.framebufferTexture2D(
      WebGL2RenderingContext.FRAMEBUFFER,
      WebGL2RenderingContext.COLOR_ATTACHMENT0,
      WebGL2RenderingContext.TEXTURE_2D,
      this._texture,
      0,
    );
    const status = context.gl.checkFramebufferStatus(
      WebGL2RenderingContext.FRAMEBUFFER,
    );
    context.gl.bindFramebuffer(WebGL2RenderingContext.FRAMEBUFFER, null);
    if (status !== WebGL2RenderingContext.FRAMEBUFFER_COMPLETE) {
      context.gl.deleteFramebuffer(this._framebuffer);
      context.gl.deleteTexture(this._texture);
      throw new Error(`Pick framebuffer is incomplete (status ${status})`);
    }
  }

  /**
   * Finds the point or shape that is visible at a pixel of the canvas
   *
   * The renderers draw the viewport they were last drawn with, which is the
   * one on screen.
   *
   * @param devicePixel - The pixel to pick, in device pixels of the canvas,
   * measured from its top-left corner
   * @returns The picked item, or `null` if nothing is visible at the pixel
   */
  pick(devicePixel: Vertex): AnnotatedItemRef | null {
    const gl = this._context.gl;
    if (gl.isContextLost()) {
      return null;
    }
    const canvasWidth = gl.canvas.width;
    const canvasHeight = gl.canvas.height;
    if (
      devicePixel.x < 0 ||
      devicePixel.x >= canvasWidth ||
      devicePixel.y < 0 ||
      devicePixel.y >= canvasHeight
    ) {
      return null;
    }
    const radius = WebGLPicker._radius;
    const size = WebGLPicker._size;
    const firstPickIds: number[] = [];
    let nextPickId = 1;
    gl.bindFramebuffer(WebGL2RenderingContext.FRAMEBUFFER, this._framebuffer);
    try {
      // dithering is enabled by default, and may perturb the bytes written to a
      // fixed-point color buffer
      gl.disable(WebGL2RenderingContext.DITHER);
      // a negative viewport origin moves the picked pixel to the center of the
      // framebuffer, while every shader keeps computing with the full canvas
      // size, i.e. draws exactly what is on screen
      gl.viewport(
        -Math.floor(devicePixel.x) + radius,
        -(canvasHeight - 1 - Math.floor(devicePixel.y)) + radius,
        canvasWidth,
        canvasHeight,
      );
      this._context.clear();
      for (const renderer of this._renderers) {
        firstPickIds.push(nextPickId);
        nextPickId = renderer.drawPickIds(nextPickId);
      }
      gl.readPixels(
        0,
        0,
        size,
        size,
        WebGL2RenderingContext.RGBA,
        WebGL2RenderingContext.UNSIGNED_BYTE,
        this._pixels,
      );
    } finally {
      gl.bindFramebuffer(WebGL2RenderingContext.FRAMEBUFFER, null);
      gl.enable(WebGL2RenderingContext.DITHER);
      gl.viewport(0, 0, canvasWidth, canvasHeight);
    }
    const pickId = WebGLPicker.findNearestPickId(this._pixels, size);
    if (pickId === 0) {
      return null;
    }
    for (let i = this._renderers.length - 1; i >= 0; i--) {
      if (pickId >= firstPickIds[i]!) {
        return this._renderers[i]!.getPickedItem(pickId - firstPickIds[i]!);
      }
    }
    return null;
  }

  /** Releases the offscreen framebuffer and its texture */
  destroy(): void {
    this._context.gl.deleteFramebuffer(this._framebuffer);
    this._context.gl.deleteTexture(this._texture);
  }

  /**
   * Returns the pick ID nearest to the center of a read-back window
   *
   * Searching around the picked pixel, rather than reading that pixel alone,
   * is what makes items of one or two pixels clickable. Ties are resolved in
   * scan order.
   *
   * @param pixels - The read-back pixels, four bytes per pixel
   * @param size - The width and height of the window, in pixels
   * @returns The nearest pick ID, or `0` if no pixel was drawn to
   */
  static findNearestPickId(pixels: Uint8Array, size: number): number {
    const center = (size - 1) / 2;
    let nearestPickId = 0;
    let nearestSquaredDistance = Infinity;
    for (let row = 0; row < size; row++) {
      for (let col = 0; col < size; col++) {
        const offset = 4 * (row * size + col);
        const pickId =
          (pixels[offset]! |
            (pixels[offset + 1]! << 8) |
            (pixels[offset + 2]! << 16) |
            (pixels[offset + 3]! << 24)) >>>
          0;
        if (pickId === 0) {
          continue;
        }
        const dx = col - center;
        const dy = row - center;
        const squaredDistance = dx * dx + dy * dy;
        if (squaredDistance < nearestSquaredDistance) {
          nearestSquaredDistance = squaredDistance;
          nearestPickId = pickId;
        }
      }
    }
    return nearestPickId;
  }
}
