// @vitest-environment jsdom
// the provider module pulls in OpenSeadragon, which reads document on import
import type { GeoTIFFImage } from "geotiff";
import { describe, expect, it } from "vitest";

import {
  TIFFLabelsDataProvider,
  getLabelLevels,
} from "./TIFFLabelsDataProvider";

function fakeImage(options?: {
  sampleFormat?: number;
  bitsPerSample?: number;
}): GeoTIFFImage {
  const { sampleFormat = 1, bitsPerSample = 16 } = options ?? {};
  return {
    getSampleFormat: () => sampleFormat,
    getBitsPerSample: () => bitsPerSample,
  } as unknown as GeoTIFFImage;
}

const float = fakeImage({ sampleFormat: 3, bitsPerSample: 32 });
const wide = fakeImage({ bitsPerSample: 64 });

describe("getLabelLevels", () => {
  it("returns the levels of a single channel of signed or unsigned integers", () => {
    const unsigned = [fakeImage(), fakeImage()];
    const signed = [fakeImage({ sampleFormat: 2, bitsPerSample: 32 })];
    const untagged = [fakeImage({ bitsPerSample: 0 })]; // no BitsPerSample tag
    expect(getLabelLevels({ pyramids: [unsigned], channels: [{}] })).toBe(
      unsigned,
    );
    expect(getLabelLevels({ pyramids: [signed], channels: [{}] })).toBe(signed);
    expect(getLabelLevels({ pyramids: [untagged], channels: [{}] })).toBe(
      untagged,
    );
  });

  it("rejects a file drawn in its own colors and one with several channels", () => {
    expect(() =>
      getLabelLevels({ pyramids: [[fakeImage()]], channels: undefined }),
    ).toThrow(/own colors/);
    expect(() =>
      getLabelLevels({
        pyramids: [[fakeImage()], [fakeImage()]],
        channels: [{}, {}],
      }),
    ).toThrow(/2 channels/);
  });

  it("rejects samples that are not integers of at most 32 bits", () => {
    expect(() =>
      getLabelLevels({ pyramids: [[float]], channels: [{}] }),
    ).toThrow(/32-bit samples of TIFF sample format 3/);
    expect(() =>
      getLabelLevels({ pyramids: [[wide]], channels: [{}] }),
    ).toThrow(/64-bit samples/);
  });
});

describe("TIFFLabelsDataProvider", () => {
  describe("supports", () => {
    const provider = new TIFFLabelsDataProvider();

    it("supports TIFF file extensions, including OME-TIFF and QPTIFF", async () => {
      for (const source of [
        "/a.tif",
        "/a.TIFF",
        "/a.ome.tif",
        "https://data.example/a.qptiff",
      ]) {
        await expect(provider.supports(source)).resolves.toBe(true);
      }
    });

    it("does not support other sources", async () => {
      await expect(provider.supports("/a.png")).resolves.toBe(false);
    });
  });
});
