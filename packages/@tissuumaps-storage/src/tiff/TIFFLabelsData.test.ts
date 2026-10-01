// @vitest-environment jsdom
// the module pulls in OpenSeadragon, which reads document on import
import type { GeoTIFFImage } from "geotiff";
import type OpenSeadragon from "openseadragon";
import { describe, expect, it, vi } from "vitest";

import { TIFFLabelsData } from "./TIFFLabelsData";

vi.mock("./installTIFFTileSource", () => ({
  installTIFFTileSource: () => ({ pool: undefined }),
  tiffRasterType: "tiff-raster",
}));

function fakeImage(): GeoTIFFImage {
  // 4 x 3 pixels, each labelled with its row-major index
  return {
    getWidth: () => 4,
    getHeight: () => 3,
    readRasters: ({ window }: { window: number[] }) => {
      const [column, row] = window as [number, number];
      return Promise.resolve([Uint32Array.of(row * 4 + column)]);
    },
  } as unknown as GeoTIFFImage;
}

describe("TIFFLabelsData", () => {
  describe("loadLabelId", () => {
    const labelsData = new TIFFLabelsData(
      {} as OpenSeadragon.TileSource,
      fakeImage(),
    );

    it("loads the label ID of the pixel holding the position", async () => {
      await expect(labelsData.loadLabelId(2, 1)).resolves.toBe(6);
      await expect(labelsData.loadLabelId(2.9, 1.5)).resolves.toBe(6);
    });

    it("returns undefined outside of the label mask", async () => {
      await expect(labelsData.loadLabelId(-0.5, 0)).resolves.toBeUndefined();
      await expect(labelsData.loadLabelId(4, 0)).resolves.toBeUndefined();
      await expect(labelsData.loadLabelId(0, 3)).resolves.toBeUndefined();
    });
  });
});
