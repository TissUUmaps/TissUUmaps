import { describe, expect, it } from "vitest";

import { type Image, type RawImage, createImage } from "../model/image";
import type { Color } from "../model/primitives";
import type { ImageChannelHistogram, ImageData } from "../storage/image";
import { ImageUtils } from "./ImageUtils";
import { MathUtils } from "./MathUtils";

function createImageData(
  channels: {
    color?: Color;
    contrastLimits?: [number, number];
    histogram?: ImageChannelHistogram;
    dataTypeRange?: [number, number];
    visibility?: boolean;
    opacity?: number;
  }[],
): ImageData {
  return {
    close: () => {},
    getSizeC: () => channels.length,
    getTileSource: () => "",
    getChannelColor: (c) => channels[c]?.color,
    getChannelContrastLimits: (c) => channels[c]?.contrastLimits,
    getChannelHistogram: (c) => channels[c]?.histogram,
    getChannelDataTypeRange: (c) => channels[c]?.dataTypeRange,
    getChannelVisibility: (c) => channels[c]?.visibility,
    getChannelOpacity: (c) => channels[c]?.opacity,
  };
}

function createTestImage(
  image?: Pick<RawImage, "channels" | "activeChannel">,
): Image {
  return createImage({
    id: "image",
    name: "image",
    layer: "layer",
    dataSource: { type: "test" },
    ...image,
  });
}

describe("ImageUtils", () => {
  describe("getChannelColor", () => {
    it("prefers the image's color over the one the data states", () => {
      const data = createImageData([{ color: { r: 1, g: 2, b: 3 } }, {}]);
      const image = createTestImage({
        channels: [{ color: { r: 4, g: 5, b: 6 } }],
      });
      expect(ImageUtils.getChannelColor(image, data, 0)).toEqual({
        r: 4,
        g: 5,
        b: 6,
      });
      expect(ImageUtils.getChannelColor(createTestImage(), data, 0)).toEqual({
        r: 1,
        g: 2,
        b: 3,
      });
    });

    it("falls back to the default color for the channel index", () => {
      const data = createImageData([{}, {}]);
      expect(ImageUtils.getChannelColor(createTestImage(), data, 1)).toEqual(
        ImageUtils.getDefaultChannelColor(1),
      );
    });

    it("returns white for the only channel of single-channel data", () => {
      const data = createImageData([{}]);
      expect(ImageUtils.getChannelColor(createTestImage(), data, 0)).toEqual({
        r: 255,
        g: 255,
        b: 255,
      });
    });
  });

  describe("getChannelContrastLimits", () => {
    it("prefers the image's limits over those the data states", () => {
      const data = createImageData([{ contrastLimits: [10, 20] }]);
      const image = createTestImage({
        channels: [{ contrastLimits: [30, 40] }],
      });
      expect(ImageUtils.getChannelContrastLimits(image, data, 0)).toEqual([
        30, 40,
      ]);
      expect(
        ImageUtils.getChannelContrastLimits(createTestImage(), data, 0),
      ).toEqual([10, 20]);
    });

    it("derives limits from the histogram before the data type range", () => {
      const histogram: ImageChannelHistogram = {
        hist: [100, 10, 10, 10, 100],
        range: [0, 4],
      };
      const data = createImageData([{ histogram, dataTypeRange: [0, 65535] }]);
      expect(
        ImageUtils.getChannelContrastLimits(createTestImage(), data, 0),
      ).toEqual(ImageUtils.getDefaultContrastLimits(histogram));
    });

    it("falls back to the data type range without a histogram", () => {
      const data = createImageData([{ dataTypeRange: [0, 65535] }]);
      expect(
        ImageUtils.getChannelContrastLimits(createTestImage(), data, 0),
      ).toEqual([0, 65535]);
    });

    it("returns undefined without limits, histogram and data type range", () => {
      const data = createImageData([{}]);
      expect(
        ImageUtils.getChannelContrastLimits(createTestImage(), data, 0),
      ).toBeUndefined();
    });
  });

  describe("getChannelVisibility", () => {
    it("prefers the image's visibility over the one the data states", () => {
      const data = createImageData([{ visibility: false }]);
      const image = createTestImage({ channels: [{ visibility: true }] });
      expect(ImageUtils.getChannelVisibility(image, data, 0)).toBe(true);
      expect(ImageUtils.getChannelVisibility(createTestImage(), data, 0)).toBe(
        false,
      );
    });

    it("returns true without a visibility", () => {
      const data = createImageData([{}]);
      expect(ImageUtils.getChannelVisibility(createTestImage(), data, 0)).toBe(
        true,
      );
    });
  });

  describe("getChannelOpacity", () => {
    it("prefers the image's opacity over the one the data states", () => {
      const data = createImageData([{ opacity: 0.2 }]);
      const image = createTestImage({ channels: [{ opacity: 0.5 }] });
      expect(ImageUtils.getChannelOpacity(image, data, 0)).toBe(0.5);
      expect(ImageUtils.getChannelOpacity(createTestImage(), data, 0)).toBe(
        0.2,
      );
    });

    it("returns 1 without an opacity", () => {
      const data = createImageData([{}]);
      expect(ImageUtils.getChannelOpacity(createTestImage(), data, 0)).toBe(1);
    });
  });

  describe("getActiveChannel", () => {
    it("bounds the image's active channel to the channels the data has", () => {
      expect(
        ImageUtils.getActiveChannel(createTestImage({ activeChannel: 7 }), 3),
      ).toBe(2);
      expect(
        ImageUtils.getActiveChannel(createTestImage({ activeChannel: 1 }), 3),
      ).toBe(1);
      expect(
        ImageUtils.getActiveChannel(createTestImage({ activeChannel: -1 }), 3),
      ).toBe(0);
    });
  });

  describe("getDefaultChannelColor", () => {
    it("returns fixed colors for the first six channels", () => {
      expect(ImageUtils.getDefaultChannelColor(0)).toEqual({
        r: 255,
        g: 0,
        b: 0,
      });
      expect(ImageUtils.getDefaultChannelColor(1)).toEqual({
        r: 0,
        g: 255,
        b: 0,
      });
      expect(ImageUtils.getDefaultChannelColor(2)).toEqual({
        r: 0,
        g: 0,
        b: 255,
      });
      expect(ImageUtils.getDefaultChannelColor(3)).toEqual({
        r: 255,
        g: 255,
        b: 0,
      });
      expect(ImageUtils.getDefaultChannelColor(4)).toEqual({
        r: 0,
        g: 255,
        b: 255,
      });
      expect(ImageUtils.getDefaultChannelColor(5)).toEqual({
        r: 255,
        g: 0,
        b: 255,
      });
    });

    it("derives further channels from HSB", () => {
      // channel 6: hue 48deg, full saturation and brightness
      expect(ImageUtils.getDefaultChannelColor(6)).toEqual({
        r: 255,
        g: 204,
        b: 0,
      });
      // channel 7: hue 176deg, full saturation and brightness
      expect(ImageUtils.getDefaultChannelColor(7)).toEqual({
        r: 0,
        g: 255,
        b: 238,
      });
      // channel 10: hue 200deg, saturation and brightness 0.95
      expect(ImageUtils.getDefaultChannelColor(10)).toEqual({
        r: 12,
        g: 166,
        b: 242,
      });
    });

    it("wraps around after 100 channels", () => {
      expect(ImageUtils.getDefaultChannelColor(100)).toEqual(
        ImageUtils.getDefaultChannelColor(0),
      );
      expect(ImageUtils.getDefaultChannelColor(107)).toEqual(
        ImageUtils.getDefaultChannelColor(7),
      );
    });

    it("keeps saturation and brightness above 0.5", () => {
      for (let c = 0; c < 100; c++) {
        const { r, g, b } = ImageUtils.getDefaultChannelColor(c);
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        const brightness = max / 255;
        const saturation = (max - min) / max;
        expect(brightness).toBeGreaterThan(0.5);
        expect(saturation).toBeGreaterThan(0.5);
      }
    });

    it("returns integer components within [0, 255] for all channels", () => {
      for (let c = 0; c < 100; c++) {
        const { r, g, b } = ImageUtils.getDefaultChannelColor(c);
        for (const v of [r, g, b]) {
          expect(Number.isInteger(v)).toBe(true);
          expect(v).toBeGreaterThanOrEqual(0);
          expect(v).toBeLessThanOrEqual(255);
        }
      }
    });
  });

  describe("getDefaultContrastLimits", () => {
    it("clips the quantiles at both ends of a uniform histogram", () => {
      // 100 values, one per bin: 5 values are clipped at each end
      const hist = new Array<number>(100).fill(1);
      const [low, high] = ImageUtils.getDefaultContrastLimits(
        { hist, range: [0, 99] },
        0.05,
        0.95,
      );
      expect(low).toBeCloseTo(4);
      expect(high).toBeCloseTo(95);
    });

    it("clips a histogram computed by MathUtils.computeHistogram", async () => {
      // 100 values, one per bin: 5 values are clipped at each end
      const data = new Uint16Array(100).map((_, i) => 1000 + i);
      const histogram = await MathUtils.computeHistogram(data, [1000, 1099], {
        bins: 100,
      });
      const [low, high] = ImageUtils.getDefaultContrastLimits(
        histogram,
        0.05,
        0.95,
      );
      expect(low).toBeCloseTo(1004);
      expect(high).toBeCloseTo(1095);
    });

    it("clips 1% at the bottom and 0.1% at the top by default", () => {
      // 1000 values, one per bin: the limits are the 10th value from the
      // bottom and the 1st value from the top
      const hist = new Array<number>(1000).fill(1);
      const [low, high] = ImageUtils.getDefaultContrastLimits({
        hist,
        range: [0, 999],
      });
      expect(low).toBeCloseTo(9);
      expect(high).toBeCloseTo(999);
    });

    it("maps quantiles of zero and one to the first and last non-empty bins", () => {
      const hist = [0, 1, 2, 3, 0];
      expect(
        ImageUtils.getDefaultContrastLimits({ hist, range: [10, 50] }, 0, 1),
      ).toEqual([20, 40]);
    });

    it("maps the first and last bins to the range bounds", () => {
      const hist = [5, 0, 0, 5];
      expect(
        ImageUtils.getDefaultContrastLimits({ hist, range: [-1, 1] }, 0.1, 0.9),
      ).toEqual([-1, 1]);
    });

    it("returns the full range when the limits would collapse", () => {
      // 98% of the values fall into bin 2: both limits would land there
      const hist = [1, 0, 98, 0, 1];
      expect(
        ImageUtils.getDefaultContrastLimits(
          { hist, range: [0, 4] },
          0.02,
          0.98,
        ),
      ).toEqual([0, 4]);
    });

    it("returns the full range when the upper quantile is below the lower", () => {
      const hist = [1, 1, 1, 1];
      expect(
        ImageUtils.getDefaultContrastLimits(
          { hist, range: [0, 3] },
          0.75,
          0.25,
        ),
      ).toEqual([0, 3]);
    });

    it("returns the full range for a lower quantile above one", () => {
      const hist = [1, 1, 1, 1];
      expect(
        ImageUtils.getDefaultContrastLimits({ hist, range: [0, 3] }, 1.5),
      ).toEqual([0, 3]);
    });

    it("returns the range as is for degenerate input", () => {
      expect(
        ImageUtils.getDefaultContrastLimits({ hist: [], range: [0, 255] }),
      ).toEqual([0, 255]);
      expect(
        ImageUtils.getDefaultContrastLimits({ hist: [7], range: [0, 255] }),
      ).toEqual([0, 255]);
      expect(
        ImageUtils.getDefaultContrastLimits({ hist: [0, 0], range: [0, 255] }),
      ).toEqual([0, 255]);
      expect(
        ImageUtils.getDefaultContrastLimits({ hist: [1, 1], range: [3, 3] }),
      ).toEqual([3, 3]);
    });
  });

  describe("getIntegerTypeRange", () => {
    it.each([
      [8, false, [0, 255]],
      [16, false, [0, 65535]],
      [32, false, [0, 4294967295]],
      [8, true, [-128, 127]],
      [16, true, [-32768, 32767]],
      [32, true, [-2147483648, 2147483647]],
    ])(
      "returns the range of %i-bit values (signed: %s)",
      (bits, signed, range) => {
        expect(ImageUtils.getIntegerTypeRange(bits, signed)).toEqual(range);
      },
    );
  });

  describe("getDataTypeRange", () => {
    it.each([
      [new Uint8Array(1), [0, 255]],
      [new Uint8ClampedArray(1), [0, 255]],
      [new Uint16Array(1), [0, 65535]],
      [new Uint32Array(1), [0, 4294967295]],
      [new Int8Array(1), [-128, 127]],
      [new Int16Array(1), [-32768, 32767]],
      [new Int32Array(1), [-2147483648, 2147483647]],
    ])("returns the full integer range of %o", (values, range) => {
      expect(ImageUtils.getDataTypeRange(values)).toEqual(range);
    });

    it.each([new Float32Array(1), new Float64Array(1)])(
      "returns [0, 1] for %o",
      (values) => {
        expect(ImageUtils.getDataTypeRange(values)).toEqual([0, 1]);
      },
    );
  });
});
