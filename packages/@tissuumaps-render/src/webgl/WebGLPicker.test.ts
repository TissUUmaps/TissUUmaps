import { describe, expect, it } from "vitest";

import { WebGLPicker } from "./WebGLPicker";

/**
 * Builds a read-back window with pick IDs at some pixels
 *
 * @param size - The width and height of the window, in pixels
 * @param pixels - The pick ID of every drawn pixel, by column and row
 * @returns The bytes of the window, four per pixel
 */
function createPixels(
  size: number,
  pixels: { col: number; row: number; pickId: number }[],
): Uint8Array {
  const bytes = new Uint8Array(4 * size * size);
  for (const { col, row, pickId } of pixels) {
    const offset = 4 * (row * size + col);
    bytes[offset] = pickId & 0xff;
    bytes[offset + 1] = (pickId >>> 8) & 0xff;
    bytes[offset + 2] = (pickId >>> 16) & 0xff;
    bytes[offset + 3] = (pickId >>> 24) & 0xff;
  }
  return bytes;
}

describe("WebGLPicker", () => {
  describe("findNearestPickId", () => {
    it("returns 0 if no pixel was drawn to", () => {
      expect(WebGLPicker.findNearestPickId(createPixels(3, []), 3)).toBe(0);
    });

    it("prefers the center pixel", () => {
      const pixels = createPixels(3, [
        { col: 0, row: 0, pickId: 7 },
        { col: 1, row: 1, pickId: 42 },
      ]);
      expect(WebGLPicker.findNearestPickId(pixels, 3)).toBe(42);
    });

    it("returns the pick ID nearest to the center", () => {
      const pixels = createPixels(9, [
        { col: 0, row: 0, pickId: 7 },
        { col: 6, row: 4, pickId: 8 },
      ]);
      expect(WebGLPicker.findNearestPickId(pixels, 9)).toBe(8);
    });

    it("resolves ties in scan order", () => {
      const pixels = createPixels(3, [
        { col: 1, row: 0, pickId: 1 },
        { col: 1, row: 2, pickId: 2 },
      ]);
      expect(WebGLPicker.findNearestPickId(pixels, 3)).toBe(1);
    });

    it("decodes all 32 bits of a pick ID", () => {
      const pixels = createPixels(1, [{ col: 0, row: 0, pickId: 0xfedcba98 }]);
      expect(WebGLPicker.findNearestPickId(pixels, 1)).toBe(0xfedcba98);
    });
  });
});
