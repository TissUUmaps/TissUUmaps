import type { GeoTIFF, GeoTIFFImage } from "geotiff";

import type { Color } from "@tissuumaps/core";

import type {
  TIFFChannelMetadata,
  TIFFParser,
  TIFFStructure,
} from "../TIFFParser";
import { TIFFUtils } from "../TIFFUtils";
import { XMLUtils } from "../XMLUtils";

/** The root element of the XML description written by PerkinElmer/Akoya scanners */
const rootElement = "PerkinElmer-QPI-ImageDescription";

/**
 * QPTIFF: one IFD per channel and pyramid level, each with an XML description
 * carrying the channel name and color and the image type. Thumbnail, overview
 * and label images are skipped.
 */
export class QPTIFFParser implements TIFFParser {
  async supports(
    tiff: GeoTIFF,
    options?: { signal?: AbortSignal },
  ): Promise<boolean> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const description = await TIFFUtils.readFirstDescription(tiff, {
      signal,
    });
    const root = XMLUtils.parse(description);
    return root?.localName === rootElement;
  }

  /** @throws Error if `z` or `t` is not 0, which only OME-TIFF has */
  async load(
    tiff: GeoTIFF,
    options?: { z?: number; t?: number; signal?: AbortSignal },
  ): Promise<TIFFStructure> {
    const { z = 0, t = 0, signal } = options ?? {};
    signal?.throwIfAborted();
    if (z !== 0 || t !== 0) {
      throw new Error("z and t require an OME-TIFF file.");
    }
    const images = await TIFFUtils.readImages(tiff, { signal });
    const descriptions = await Promise.all(
      images.map((image) => TIFFUtils.readDescription(image, { signal })),
    );
    const entries: { image: GeoTIFFImage; channel: TIFFChannelMetadata }[] = [];
    images.forEach((image, i) => {
      const root = XMLUtils.parse(descriptions[i]);
      if (root?.localName !== rootElement) {
        return;
      }
      const imageType = XMLUtils.getChildText(root, "ImageType");
      if (
        imageType !== undefined &&
        imageType !== "FullResolution" &&
        imageType !== "ReducedResolution"
      ) {
        return;
      }
      entries.push({
        image,
        channel: {
          name: XMLUtils.getChildText(root, "Name"),
          color: parseColor(XMLUtils.getChildText(root, "Color")),
        },
      });
    });
    const { full, reduced } = TIFFUtils.partitionBySize(
      entries.map((entry) => entry.image),
    );
    const channels = entries
      .filter((entry) => full.includes(entry.image))
      .map((entry) => entry.channel);
    const pyramids = TIFFUtils.groupByWidth(full, reduced);
    TIFFUtils.validatePyramids(pyramids);
    return {
      pyramids,
      channels: TIFFUtils.hasOwnColors(pyramids) ? undefined : channels,
    };
  }
}

/** Parses a QPTIFF color, `"r,g,b"` */
function parseColor(text: string | undefined): Color | undefined {
  const values = text?.split(",").map(Number);
  if (values?.length !== 3 || values.some((v) => !Number.isFinite(v))) {
    return undefined;
  }
  return { r: values[0]!, g: values[1]!, b: values[2]! };
}
