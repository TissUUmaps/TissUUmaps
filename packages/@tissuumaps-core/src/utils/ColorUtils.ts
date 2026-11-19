import { rgb as parseCSSColor } from "d3-color";

import type { Color } from "../model/primitives";
import { BitUtils } from "./BitUtils";
import { MathUtils } from "./MathUtils";

/** Utility methods for color parsing, packing, and conversion */
export class ColorUtils {
  /**
   * Parses a text-based color palette into an array of colors
   *
   * Each non-empty line is expected to contain three numeric components
   * separated by `sep`, scaled from `[0, maxValue]` to `[0, 255]`.
   *
   * @param str - Multi-line palette string
   * @param options - Component separator (default `" "`) and maximum input
   * value per component (default `1`)
   * @returns The parsed colors, in the order of the palette's lines
   * @throws Error if a non-empty line does not hold exactly three components
   */
  static parsePalette(
    str: string,
    options?: { sep?: string; maxValue?: number },
  ): Color[] {
    const { sep = " ", maxValue = 1 } = options ?? {};
    return str
      .split("\n")
      .filter((line) => line.trim().length > 0)
      .map((line, i) => {
        const values = line.split(sep);
        if (values.length !== 3) {
          throw new Error(`Invalid color palette line '${line}' at index ${i}`);
        }
        return {
          r: (Number(values[0]) / maxValue) * 255,
          g: (Number(values[1]) / maxValue) * 255,
          b: (Number(values[2]) / maxValue) * 255,
        };
      });
  }

  /**
   * Samples a continuous color scheme into an array of colors
   *
   * The scheme is evaluated at `n` evenly spaced positions covering the whole
   * `[0, 1]` range, both ends included.
   *
   * @param colorScheme - Maps a position within `[0, 1]` to a CSS color string
   * @param n - Number of colors to sample, an integer of at least two
   * @returns The sampled colors, in the order of the scheme
   * @throws Error if `n` is not an integer of at least two, or if the scheme
   * yields a color string that cannot be parsed
   */
  static sampleColorScheme(
    colorScheme: (t: number) => string,
    n: number,
  ): Color[] {
    if (!Number.isInteger(n) || n < 2) {
      throw new Error(`Invalid color scheme size ${n}`);
    }
    return Array.from({ length: n }, (_, i) => {
      const spec = colorScheme(i / (n - 1));
      // d3-color reports a parse failure as NaN components, rather than by
      // throwing or returning null. It does not clamp, so this detects an
      // unparseable color only, not an out-of-range one.
      const { r, g, b } = parseCSSColor(spec);
      if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) {
        throw new Error(`Invalid color scheme color '${spec}'`);
      }
      return { r, g, b };
    });
  }

  /**
   * Packs an RGB color into a single 24-bit integer, `0xBBGGRR`
   *
   * The red component occupies the lowest byte, so that the integer, when
   * stored in host byte order on a little-endian host, matches the R, G, B byte
   * order of a canvas `ImageData` buffer.
   *
   * @param color - The color to pack
   * @returns The packed color, `0xBBGGRR`
   */
  static packColor(color: Color): number {
    // never exceeds 24 bits (0xFFFFFF), so no need for >>> 0
    return (color.b << 16) | (color.g << 8) | color.r;
  }

  /**
   * Packs a visibility into the form that {@link withAlpha} folds into a color
   *
   * @param visibility - The visibility
   * @returns `1` if visible, `0` if not
   */
  static packVisibility(visibility: boolean): number {
    return visibility ? 1 : 0;
  }

  /**
   * Packs an opacity into the form that {@link withAlpha} folds into a color
   *
   * @param opacity - The opacity, in the range [0, 1]; values outside are clamped
   * @returns The opacity as an integer in the range [0, 255]
   */
  static packOpacity(opacity: number): number {
    return MathUtils.clamp(Math.round(opacity * 255), 0, 255);
  }

  /**
   * Folds a visibility and an opacity into the alpha channel of a packed color
   *
   * Combines the lower 24 bits of `color` (see {@link packColor}) with
   * `opacity` as alpha if `visibility` is non-zero, and with zero alpha
   * otherwise, into a packed 32-bit RGBA color, `0xAABBGGRR`.
   *
   * @param packedColor - The packed color, `0xBBGGRR`; any higher bits are discarded
   * @param packedVisibility - The visibility, larger than `0` for visible
   * @param packedOpacity - The opacity, in the range [0, 255]
   * @returns The packed RGBA color, `0xAABBGGRR`, as an unsigned 32-bit integer
   */
  static withAlpha(
    packedColor: number,
    packedVisibility: number,
    packedOpacity: number,
  ): number {
    const rgb = packedColor & 0x00ffffff;
    if (packedVisibility > 0) {
      return BitUtils.safeOr(rgb, BitUtils.safeLeftShift(packedOpacity, 24));
    }
    return rgb;
  }

  /**
   * Parses a hex color string into a {@link Color}
   *
   * @param hex - A `#RRGGBB` hex string
   * @returns The parsed color
   * @throws Error if the string is not a valid 6-digit hex color
   */
  static fromHex(hex: string): Color {
    if (!/^#([0-9A-Fa-f]{6})$/.test(hex)) {
      throw new Error(`Invalid hex color '${hex}'`);
    }
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return { r, g, b };
  }

  /**
   * Formats a {@link Color} as a `#RRGGBB` hex string
   *
   * @param color - The color to format
   * @returns The formatted hex string
   */
  static toHex(color: Color): string {
    const rHex = Math.round(color.r).toString(16).padStart(2, "0");
    const gHex = Math.round(color.g).toString(16).padStart(2, "0");
    const bHex = Math.round(color.b).toString(16).padStart(2, "0");
    return `#${rHex}${gHex}${bHex}`;
  }

  /**
   * Checks whether two colors have identical RGB components
   *
   * Components are compared exactly, without rounding or tolerance.
   *
   * @param a - The first color
   * @param b - The second color
   * @returns `true` if all three components are equal, `false` otherwise
   */
  static equals(a: Color, b: Color): boolean {
    return a.r === b.r && a.g === b.g && a.b === b.b;
  }
  /**
   * Converts an HSB (HSV) color to RGB, following Java's `Color.HSBtoRGB`
   *
   * @param hue - The hue, in the range [0, 1) (values outside wrap around)
   * @param saturation - The saturation, in the range [0, 1]
   * @param brightness - The brightness, in the range [0, 1]
   * @returns The RGB color, with integer components in the range [0, 255]
   */
  static fromHSB(hue: number, saturation: number, brightness: number): Color {
    const scale = (x: number) => Math.round(x * 255);
    if (saturation === 0) {
      const v = scale(brightness);
      return { r: v, g: v, b: v };
    }
    const h = (hue - Math.floor(hue)) * 6;
    const f = h - Math.floor(h);
    const v = scale(brightness);
    const p = scale(brightness * (1 - saturation));
    const q = scale(brightness * (1 - saturation * f));
    const t = scale(brightness * (1 - saturation * (1 - f)));
    switch (Math.floor(h)) {
      case 0:
        return { r: v, g: t, b: p };
      case 1:
        return { r: q, g: v, b: p };
      case 2:
        return { r: p, g: v, b: t };
      case 3:
        return { r: p, g: q, b: v };
      case 4:
        return { r: t, g: p, b: v };
      default:
        return { r: v, g: p, b: q };
    }
  }
}
