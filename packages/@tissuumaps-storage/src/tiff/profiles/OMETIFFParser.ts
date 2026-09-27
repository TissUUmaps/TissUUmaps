import type { GeoTIFF } from "geotiff";

import type { Color } from "@tissuumaps/core";

import type { TIFFParser, TIFFStructure } from "../TIFFParser";
import { TIFFUtils } from "../TIFFUtils";
import { XMLUtils } from "../XMLUtils";

type TiffData = {
  ifd: number;
  firstC: number;
  firstZ: number;
  firstT: number;
  planeCount: number;
};

type Pixels = {
  sizeX: number;
  sizeY: number;
  sizeC: number;
  sizeZ: number;
  sizeT: number;
  dimensionOrder: string;
  channels: {
    name: string | undefined;
    color: Color | undefined;
    samplesPerPixel: number;
  }[];
  tiffData: TiffData[];
};

type Dim = "C" | "Z" | "T";

/** The root element of the OME-XML description */
const rootElement = "OME";

/** The values OME-XML allows for `DimensionOrder` */
const dimensionOrders = ["XYZCT", "XYZTC", "XYCTZ", "XYCZT", "XYTCZ", "XYTZC"];

/**
 * OME-TIFF: one plane (channel, z, t) per IFD, mapped to IFDs by the OME-XML
 * of the first IFD; pyramid levels in SubIFDs. Files with several images show
 * the largest one.
 */
export class OMETIFFParser implements TIFFParser {
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

  /**
   * @throws Error if the OME-XML describes no image or an invalid
   * `DimensionOrder`, if `z` or `t` is out of bounds, if a plane is in another
   * file, or if a plane's IFD is missing
   */
  async load(
    tiff: GeoTIFF,
    options?: { z?: number; t?: number; signal?: AbortSignal },
  ): Promise<TIFFStructure> {
    const { z = 0, t = 0, signal } = options ?? {};
    signal?.throwIfAborted();
    const description = await TIFFUtils.readFirstDescription(tiff, {
      signal,
    });
    const root = XMLUtils.parse(description);
    if (root?.localName !== rootElement) {
      throw new Error("The file has no OME-XML.");
    }
    const images = await TIFFUtils.readImages(tiff, { signal });

    const fileUUID = root.getAttribute("UUID");
    const allPixels = XMLUtils.getChildren(root, "Image")
      .flatMap((image) => XMLUtils.getChildren(image, "Pixels").slice(0, 1))
      .map((pixels) => parsePixels(pixels, fileUUID));
    if (allPixels.length === 0) {
      throw new Error("The OME-XML describes no image.");
    }
    const pixels = allPixels.reduce((largest, candidate) =>
      candidate.sizeX * candidate.sizeY > largest.sizeX * largest.sizeY
        ? candidate
        : largest,
    );
    if (z < 0 || z >= pixels.sizeZ) {
      throw new Error(`z=${z} is out of bounds (SizeZ=${pixels.sizeZ}).`);
    }
    if (t < 0 || t >= pixels.sizeT) {
      throw new Error(`t=${t} is out of bounds (SizeT=${pixels.sizeT}).`);
    }

    const ifdByPlane = mapPlanesToIFDs(pixels, images.length);
    const planes = Array.from({ length: getPlaneCountC(pixels) }, (_, c) => {
      const ifd = ifdByPlane.get(planeKey(c, z, t));
      if (ifd === undefined) {
        throw new Error(`No IFD for plane c=${c}, z=${z}, t=${t}.`);
      }
      return images[ifd]!; // mapPlanesToIFDs() only maps IFDs the file has
    });
    // IFDs of any image (overviews, labels) are not reduced copies of this one
    const claimed = new Set(
      allPixels.flatMap((p) => [...mapPlanesToIFDs(p, images.length).values()]),
    );
    const fullWidth = planes[0]!.getWidth();
    const reduced = images.filter(
      (image, ifd) => !claimed.has(ifd) && image.getWidth() < fullWidth,
    );
    const pyramids =
      (await TIFFUtils.readSubIFDPyramids(tiff, planes, { signal })) ??
      TIFFUtils.groupByWidth(planes, reduced);
    TIFFUtils.validatePyramids(pyramids);
    const channels = planes.map((_, c) => ({
      name: pixels.channels[c]?.name,
      color: pixels.channels[c]?.color,
    }));
    return {
      pyramids,
      channels: TIFFUtils.hasOwnColors(pyramids) ? undefined : channels,
    };
  }
}

function parsePixels(pixels: Element, fileUUID: string | null): Pixels {
  const dimensionOrder = pixels.getAttribute("DimensionOrder") ?? "XYCZT";
  if (!dimensionOrders.includes(dimensionOrder)) {
    throw new Error(`Invalid DimensionOrder "${dimensionOrder}".`);
  }
  const tiffData = XMLUtils.getChildren(pixels, "TiffData").map((element) => {
    const uuid = XMLUtils.getChildren(element, "UUID")[0];
    if (uuid !== undefined && uuid.textContent?.trim() !== fileUUID) {
      const fileName =
        uuid.getAttribute("FileName") ?? uuid.textContent?.trim();
      throw new Error(
        fileUUID === null
          ? `The OME-XML has no UUID, so the planes in ${fileName} cannot be matched to this file.`
          : `Multi-file OME-TIFF is not supported (${fileName}).`,
      );
    }
    // PlaneCount defaults to 1 if IFD is given, else to all remaining planes
    const defaultPlaneCount = element.hasAttribute("IFD") ? 1 : Infinity;
    return {
      ifd: XMLUtils.getIntAttribute(element, "IFD", 0),
      firstC: XMLUtils.getIntAttribute(element, "FirstC", 0),
      firstZ: XMLUtils.getIntAttribute(element, "FirstZ", 0),
      firstT: XMLUtils.getIntAttribute(element, "FirstT", 0),
      planeCount: XMLUtils.getIntAttribute(
        element,
        "PlaneCount",
        defaultPlaneCount,
      ),
    };
  });
  return {
    sizeX: XMLUtils.getIntAttribute(pixels, "SizeX", 0),
    sizeY: XMLUtils.getIntAttribute(pixels, "SizeY", 0),
    sizeC: XMLUtils.getIntAttribute(pixels, "SizeC", 1),
    sizeZ: XMLUtils.getIntAttribute(pixels, "SizeZ", 1),
    sizeT: XMLUtils.getIntAttribute(pixels, "SizeT", 1),
    dimensionOrder,
    channels: XMLUtils.getChildren(pixels, "Channel").map((channel) => ({
      name: channel.getAttribute("Name") ?? undefined,
      color: parseColor(channel.getAttribute("Color")),
      samplesPerPixel: XMLUtils.getIntAttribute(channel, "SamplesPerPixel", 1),
    })),
    // without TiffData, the IFDs are the planes in dimension order
    tiffData:
      tiffData.length > 0
        ? tiffData
        : [{ ifd: 0, firstC: 0, firstZ: 0, firstT: 0, planeCount: Infinity }],
  };
}

/** SizeC counts samples, so an RGB channel spans three values of c but one plane */
function getPlaneCountC(pixels: Pixels): number {
  const samplesPerPixel = pixels.channels[0]?.samplesPerPixel ?? 1;
  return Math.max(1, Math.floor(pixels.sizeC / samplesPerPixel));
}

/**
 * Maps every plane to its IFD. Within a TiffData run, planes follow
 * `DimensionOrder`: of its last three letters, the first varies fastest. Runs
 * are cut at the last IFD the file has, so a header that declares more planes
 * than the file holds cannot make this loop for long.
 */
function mapPlanesToIFDs(
  pixels: Pixels,
  imageCount: number,
): Map<string, number> {
  const sizes: Record<Dim, number> = {
    C: getPlaneCountC(pixels),
    Z: pixels.sizeZ,
    T: pixels.sizeT,
  };
  const order = pixels.dimensionOrder.slice(2).split("") as Dim[];
  const totalPlanes = sizes.C * sizes.Z * sizes.T;
  const map = new Map<string, number>();
  for (const tiffData of pixels.tiffData) {
    const start: Record<Dim, number> = {
      C: tiffData.firstC,
      Z: tiffData.firstZ,
      T: tiffData.firstT,
    };
    let linear = 0;
    let stride = 1;
    for (const dim of order) {
      linear += start[dim] * stride;
      stride *= sizes[dim];
    }
    const count = Math.min(
      tiffData.planeCount,
      totalPlanes - linear,
      imageCount - tiffData.ifd,
    );
    for (let i = 0; i < count; i++) {
      let rest = linear + i;
      const coords: Record<Dim, number> = { C: 0, Z: 0, T: 0 };
      for (const dim of order) {
        coords[dim] = rest % sizes[dim];
        rest = Math.floor(rest / sizes[dim]);
      }
      map.set(planeKey(coords.C, coords.Z, coords.T), tiffData.ifd + i);
    }
  }
  return map;
}

function planeKey(c: number, z: number, t: number): string {
  return `${c},${z},${t}`;
}

/**
 * Parses an OME-XML color: a 32-bit RGBA integer with red in the top byte,
 * usually written signed. The schema default (opaque white) means no color.
 */
function parseColor(rawColor: string | null): Color | undefined {
  const n = Number.parseInt(rawColor ?? "", 10);
  if (!Number.isFinite(n)) {
    return undefined;
  }
  const rgba = n >>> 0;
  if (rgba === 0xffffffff) {
    return undefined;
  }
  return {
    r: (rgba >>> 24) & 0xff,
    g: (rgba >>> 16) & 0xff,
    b: (rgba >>> 8) & 0xff,
  };
}
