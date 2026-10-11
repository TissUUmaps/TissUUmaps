// @vitest-environment jsdom
import type { GeoTIFF } from "geotiff";
import { describe, expect, it } from "vitest";

import { OMETIFFParser } from "./OMETIFFParser";

/** A file whose first IFD has the given `ImageDescription` */
function fakeTIFF(description: string | undefined): GeoTIFF {
  const image = {
    getFileDirectory: () => ({
      loadValue: () => Promise.resolve(description),
    }),
  };
  return {
    getImageCount: () => Promise.resolve(1),
    getImage: () => Promise.resolve(image),
  } as unknown as GeoTIFF;
}

function omeXML(images: string): string {
  return `<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">${images}</OME>`;
}

describe("OMETIFFParser", () => {
  describe("readImageName", () => {
    it("reads the name of the largest image", async () => {
      const tiff = fakeTIFF(
        omeXML(
          `<Image Name="label"><Pixels SizeX="10" SizeY="10"/></Image>` +
            `<Image Name="slide"><Pixels SizeX="100" SizeY="100"/></Image>` +
            `<Image Name="overview"><Pixels SizeX="100" SizeY="100"/></Image>`,
        ),
      );
      await expect(OMETIFFParser.readImageName(tiff)).resolves.toBe("slide");
    });

    it("returns undefined for an image without a name", async () => {
      const tiff = fakeTIFF(
        omeXML(`<Image Name=" "><Pixels SizeX="10" SizeY="10"/></Image>`),
      );
      await expect(OMETIFFParser.readImageName(tiff)).resolves.toBeUndefined();
    });

    it("returns undefined for a file without OME-XML", async () => {
      await expect(
        OMETIFFParser.readImageName(fakeTIFF("ImageJ=1.54")),
      ).resolves.toBeUndefined();
      await expect(
        OMETIFFParser.readImageName(fakeTIFF(undefined)),
      ).resolves.toBeUndefined();
    });
  });
});
