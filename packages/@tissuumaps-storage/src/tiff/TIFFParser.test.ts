// @vitest-environment jsdom
import {
  type GeoTIFF,
  GeoTIFFImage,
  fromArrayBuffer,
  writeArrayBuffer,
} from "geotiff";
import { afterEach, describe, expect, it } from "vitest";

import { type TIFFStructure, findTIFFParser } from "./TIFFParser";
import { OMETIFFParser } from "./profiles/OMETIFFParser";
import { PlainTIFFParser } from "./profiles/PlainTIFFParser";
import { QPTIFFParser } from "./profiles/QPTIFFParser";

// The three parsers are tested together, rather than one file each, because
// they share the fake-TIFF fixtures below: the same directories have to be
// read as OME-TIFF, QPTIFF and plain TIFF to tell the formats apart.

type FakeImageOptions = {
  width: number;
  height: number;
  samplesPerPixel?: number;
  photometricInterpretation?: number;
  description?: string;
  subIFDs?: FakeImageOptions[];
};

// SubIFD "offsets" index into this list, which the fake parser resolves
let directories: FakeImageOptions[] = [];

function fakeDirectory(options: FakeImageOptions) {
  const {
    width,
    height,
    samplesPerPixel = 1,
    photometricInterpretation,
    description,
    subIFDs,
  } = options;
  const values: Record<string, unknown> = {
    ImageWidth: width,
    ImageLength: height,
    SamplesPerPixel: samplesPerPixel,
    PhotometricInterpretation: photometricInterpretation,
    TileWidth: 512,
    TileLength: 512,
    PlanarConfiguration: 1,
    ImageDescription:
      description !== undefined ? `${description}\0` : undefined,
    SubIFDs: subIFDs?.map((sub) => directories.push(sub) - 1),
  };
  return {
    hasTag: (tag: string) => values[tag] !== undefined,
    getValue: (tag: string) => values[tag],
    loadValue: (tag: string) => Promise.resolve(values[tag]),
  };
}

function fakeImage(options: FakeImageOptions): GeoTIFFImage {
  const { width, height, samplesPerPixel = 1 } = options;
  return {
    getWidth: () => width,
    getHeight: () => height,
    getSamplesPerPixel: () => samplesPerPixel,
    getFileDirectory: () => fakeDirectory(options),
  } as unknown as GeoTIFFImage;
}

function fakeTIFF(images: GeoTIFFImage[]): GeoTIFF {
  return {
    getImageCount: () => Promise.resolve(images.length),
    getImage: (index: number) => Promise.resolve(images[index]),
    parser: {
      parseFileDirectoryAt: (offset: number) =>
        Promise.resolve(fakeDirectory(directories[offset]!)),
    },
    littleEndian: true,
    cache: false,
    source: {},
  } as unknown as GeoTIFF;
}

async function read(
  tiff: GeoTIFF,
  options?: { z?: number; t?: number },
): Promise<TIFFStructure> {
  const parser = await findTIFFParser(tiff);
  return parser.load(tiff, options);
}

function widths(structure: TIFFStructure): number[][] {
  return structure.pyramids.map((pyramid) =>
    pyramid.map((image) => image.getWidth()),
  );
}

function qpiDescription(
  imageType: string,
  name?: string,
  color?: string,
): string {
  return `<?xml version="1.0" encoding="utf-8"?>
    <PerkinElmer-QPI-ImageDescription>
      <DescriptionVersion>2</DescriptionVersion>
      <ImageType>${imageType}</ImageType>
      ${name !== undefined ? `<Name>${name}</Name>` : ""}
      ${color !== undefined ? `<Color>${color}</Color>` : ""}
    </PerkinElmer-QPI-ImageDescription>`;
}

function omeImage(options: {
  sizeX?: number;
  sizeY?: number;
  sizeC: number;
  sizeZ?: number;
  sizeT?: number;
  dimensionOrder?: string;
  channels?: { name?: string; color?: number; samplesPerPixel?: number }[];
  tiffData?: string;
}): string {
  const {
    sizeX = 100,
    sizeY = 100,
    sizeC,
    sizeZ = 1,
    sizeT = 1,
    dimensionOrder = "XYCZT",
    channels = [],
    tiffData = "",
  } = options;
  const channelXML = channels
    .map(
      (ch, i) =>
        `<Channel ID="Channel:0:${i}"` +
        (ch.name !== undefined ? ` Name="${ch.name}"` : "") +
        (ch.color !== undefined ? ` Color="${ch.color}"` : "") +
        ` SamplesPerPixel="${ch.samplesPerPixel ?? 1}"/>`,
    )
    .join("");
  return `<Image ID="Image:0" Name="test">
      <Pixels ID="Pixels:0" DimensionOrder="${dimensionOrder}" Type="uint16"
              SizeX="${sizeX}" SizeY="${sizeY}" SizeC="${sizeC}" SizeZ="${sizeZ}" SizeT="${sizeT}">
        ${channelXML}
        ${tiffData}
      </Pixels>
    </Image>`;
}

function omeDescription(...images: string[]): string {
  return `<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" UUID="urn:uuid:self">${images.join("")}</OME>`;
}

function omePlanes(description: string, count: number): GeoTIFFImage[] {
  return Array.from({ length: count }, (_, i) =>
    fakeImage({
      width: 100,
      height: 100,
      description: i === 0 ? description : undefined,
    }),
  );
}

afterEach(() => {
  directories = [];
});

describe("QPTIFFParser", () => {
  it("groups fluorescence IFDs into channels and levels", async () => {
    const images = [
      fakeImage({
        width: 1000,
        height: 800,
        description: qpiDescription("FullResolution", "DAPI", "0,0,255"),
      }),
      fakeImage({
        width: 1000,
        height: 800,
        description: qpiDescription("FullResolution", "Opal 520", "0,255,0"),
      }),
      fakeImage({
        width: 300,
        height: 200,
        description: qpiDescription("Thumbnail"),
      }),
      fakeImage({
        width: 500,
        height: 400,
        description: qpiDescription("ReducedResolution", "DAPI"),
      }),
      fakeImage({
        width: 500,
        height: 400,
        description: qpiDescription("ReducedResolution", "Opal 520"),
      }),
      fakeImage({
        width: 250,
        height: 200,
        description: qpiDescription("ReducedResolution", "DAPI"),
      }),
      fakeImage({
        width: 250,
        height: 200,
        description: qpiDescription("ReducedResolution", "Opal 520"),
      }),
      fakeImage({
        width: 600,
        height: 200,
        description: qpiDescription("Overview"),
      }),
      fakeImage({
        width: 200,
        height: 600,
        description: qpiDescription("Label"),
      }),
    ];
    const tiff = fakeTIFF(images);
    expect(await findTIFFParser(tiff)).toBeInstanceOf(QPTIFFParser);
    const structure = await read(tiff);
    expect(widths(structure)).toEqual([
      [1000, 500, 250],
      [1000, 500, 250],
    ]);
    expect(structure.channels?.map((ch) => ch.name)).toEqual([
      "DAPI",
      "Opal 520",
    ]);
    expect(structure.channels?.[0]?.color).toEqual({ r: 0, g: 0, b: 255 });
  });

  it("reads a brightfield RGB pyramid as an image without channels", async () => {
    const images = [
      fakeImage({
        width: 1000,
        height: 800,
        samplesPerPixel: 3,
        photometricInterpretation: 2,
        description: qpiDescription("FullResolution"),
      }),
      fakeImage({
        width: 500,
        height: 400,
        samplesPerPixel: 3,
        description: qpiDescription("ReducedResolution"),
      }),
    ];
    const structure = await read(fakeTIFF(images));
    expect(structure.channels).toBeUndefined();
    expect(widths(structure)).toEqual([[1000, 500]]);
  });
});

describe("OMETIFFParser", () => {
  it("maps planes to channels through TiffData and reads channel metadata", async () => {
    const description = omeDescription(
      omeImage({
        sizeC: 3,
        channels: [
          { name: "DAPI", color: 65535 }, // 0x0000FFFF: blue, opaque
          { name: "GFP", color: 16711935 }, // 0x00FF00FF: green, opaque
          { name: "Cy5", color: -16776961 }, // 0xFF0000FF: red, opaque
        ],
        tiffData: `<TiffData IFD="0" PlaneCount="3"><UUID FileName="self.ome.tiff">urn:uuid:self</UUID></TiffData>`,
      }),
    );
    const images = omePlanes(description, 3);
    const tiff = fakeTIFF(images);
    expect(await findTIFFParser(tiff)).toBeInstanceOf(OMETIFFParser);
    const structure = await read(tiff);
    expect(structure.pyramids).toEqual(images.map((image) => [image]));
    expect(structure.channels?.map((ch) => ch.name)).toEqual([
      "DAPI",
      "GFP",
      "Cy5",
    ]);
    expect(structure.channels?.map((ch) => ch.color)).toEqual([
      { r: 0, g: 0, b: 255 },
      { r: 0, g: 255, b: 0 },
      { r: 255, g: 0, b: 0 },
    ]);
  });

  it("selects the planes of the requested z and t in dimension order", async () => {
    // XYZCT: z varies fastest, then c, then t; 2 channels, 2 z, 2 t = 8 planes
    const description = omeDescription(
      omeImage({
        sizeC: 2,
        sizeZ: 2,
        sizeT: 2,
        dimensionOrder: "XYZCT",
        channels: [{ name: "A" }, { name: "B" }],
        tiffData: `<TiffData IFD="0" PlaneCount="8"/>`,
      }),
    );
    const images = omePlanes(description, 8);
    const structure = await read(fakeTIFF(images), { z: 1, t: 1 });
    // plane index = z + 2 * c + 4 * t
    expect(structure.pyramids).toEqual([[images[5]], [images[7]]]);
  });

  it("maps every IFD to a plane through a bare TiffData", async () => {
    const description = omeDescription(
      omeImage({
        sizeC: 3,
        channels: [{ name: "A" }, { name: "B" }, { name: "C" }],
        tiffData: `<TiffData/>`,
      }),
    );
    const images = omePlanes(description, 3);
    const structure = await read(fakeTIFF(images));
    expect(structure.pyramids).toEqual(images.map((image) => [image]));
  });

  it("maps planes described by several TiffData elements", async () => {
    // XYZCT, 2 channels, 3 z, 2 t = 12 planes, one TiffData per plane
    const description = omeDescription(
      omeImage({
        sizeC: 2,
        sizeZ: 3,
        sizeT: 2,
        dimensionOrder: "XYZCT",
        channels: [{ name: "A" }, { name: "B" }],
        tiffData: Array.from(
          { length: 12 },
          (_, i) =>
            `<TiffData IFD="${i}" FirstC="${Math.floor(i / 3) % 2}" FirstZ="${i % 3}" FirstT="${Math.floor(i / 6)}"/>`,
        ).join(""),
      }),
    );
    const images = omePlanes(description, 12);
    const structure = await read(fakeTIFF(images), { z: 1, t: 1 });
    // plane index = z + 3 * c + 6 * t
    expect(structure.pyramids).toEqual([[images[7]], [images[10]]]);
  });

  it("leaves all channels uncolored when the file colors none of them", async () => {
    const description = omeDescription(
      omeImage({
        sizeC: 3,
        channels: [{}, {}, {}],
        tiffData: `<TiffData IFD="0" PlaneCount="3"/>`,
      }),
    );
    const structure = await read(fakeTIFF(omePlanes(description, 3)));
    expect(structure.channels?.map((ch) => ch.color)).toEqual([
      undefined,
      undefined,
      undefined,
    ]);
  });

  it("leaves the uncolored channels of a partly colored file uncolored", async () => {
    const description = omeDescription(
      omeImage({
        sizeC: 3,
        channels: [{ color: -16776961 }, {}, {}], // red, uncolored, uncolored
        tiffData: `<TiffData IFD="0" PlaneCount="3"/>`,
      }),
    );
    const structure = await read(fakeTIFF(omePlanes(description, 3)));
    expect(structure.channels?.map((ch) => ch.color)).toEqual([
      { r: 255, g: 0, b: 0 },
      undefined,
      undefined,
    ]);
  });

  it("treats the default color as no color, signed or unsigned", async () => {
    const description = omeDescription(
      omeImage({
        sizeC: 3,
        // red, then the default (unset) color as -1 and as 4294967295
        channels: [{ color: -16776961 }, { color: -1 }, { color: 4294967295 }],
        tiffData: `<TiffData IFD="0" PlaneCount="3"/>`,
      }),
    );
    const structure = await read(fakeTIFF(omePlanes(description, 3)));
    expect(structure.channels?.map((ch) => ch.color)).toEqual([
      { r: 255, g: 0, b: 0 },
      undefined,
      undefined,
    ]);
  });

  it("reads a single channel with three samples as an RGB image", async () => {
    const description = omeDescription(
      omeImage({ sizeC: 3, channels: [{ samplesPerPixel: 3 }] }),
    );
    const structure = await read(
      fakeTIFF([
        fakeImage({
          width: 100,
          height: 100,
          samplesPerPixel: 3,
          photometricInterpretation: 2,
          description,
        }),
      ]),
    );
    expect(structure.channels).toBeUndefined();
    expect(structure.pyramids).toHaveLength(1);
  });

  it("shows the largest image of a multi-image file", async () => {
    // a small overview (IFD 0) and the scan (IFD 1)
    const description = omeDescription(
      omeImage({
        sizeX: 10,
        sizeY: 20,
        sizeC: 3,
        channels: [{ samplesPerPixel: 3 }],
        tiffData: `<TiffData IFD="0" PlaneCount="1"/>`,
      }),
      omeImage({
        sizeX: 1000,
        sizeY: 2000,
        sizeC: 3,
        channels: [{ samplesPerPixel: 3 }],
        tiffData: `<TiffData IFD="1" PlaneCount="1"/>`,
      }),
    );
    const images = [
      fakeImage({ width: 10, height: 20, samplesPerPixel: 3, description }),
      fakeImage({
        width: 1000,
        height: 2000,
        samplesPerPixel: 3,
        photometricInterpretation: 2,
      }),
    ];
    const structure = await read(fakeTIFF(images));
    expect(structure.pyramids).toEqual([[images[1]]]);
    expect(structure.channels).toBeUndefined();
  });

  it("rejects planes stored in another file", async () => {
    const description = omeDescription(
      omeImage({
        sizeC: 1,
        channels: [{}],
        tiffData: `<TiffData IFD="0" PlaneCount="1"><UUID FileName="other.ome.tiff">urn:uuid:other</UUID></TiffData>`,
      }),
    );
    await expect(read(fakeTIFF(omePlanes(description, 1)))).rejects.toThrow(
      /other\.ome\.tiff/,
    );
  });

  it("rejects TiffData naming a file when the root has no UUID", async () => {
    const description = `<?xml version="1.0" encoding="UTF-8"?>
      <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">${omeImage(
        {
          sizeC: 1,
          channels: [{}],
          tiffData: `<TiffData IFD="0" PlaneCount="1"><UUID FileName="self.ome.tiff">urn:uuid:self</UUID></TiffData>`,
        },
      )}</OME>`;
    await expect(read(fakeTIFF(omePlanes(description, 1)))).rejects.toThrow(
      /no UUID.*self\.ome\.tiff/,
    );
  });

  it("does not expand planes the file does not have", async () => {
    const description = omeDescription(
      omeImage({ sizeC: 1, sizeZ: 100000, sizeT: 100000, channels: [{}] }),
    );
    const images = omePlanes(description, 1);
    const structure = await read(fakeTIFF(images));
    expect(structure.pyramids).toEqual([[images[0]]]);
  });

  it("rejects z out of bounds", async () => {
    const description = omeDescription(omeImage({ sizeC: 1, channels: [{}] }));
    await expect(
      read(fakeTIFF(omePlanes(description, 1)), { z: 1 }),
    ).rejects.toThrow(/out of bounds/);
  });

  it("builds pyramid levels from SubIFDs", async () => {
    const description = omeDescription(
      omeImage({
        sizeC: 2,
        channels: [{ name: "A" }, { name: "B" }],
        tiffData: `<TiffData IFD="0" PlaneCount="2"/>`,
      }),
    );
    const subIFDs = [
      { width: 50, height: 50 },
      { width: 25, height: 25 },
    ];
    const images = [
      fakeImage({ width: 100, height: 100, description, subIFDs }),
      fakeImage({ width: 100, height: 100, subIFDs }),
    ];
    const structure = await read(fakeTIFF(images));
    expect(widths(structure)).toEqual([
      [100, 50, 25],
      [100, 50, 25],
    ]);
    expect(structure.pyramids[0]![1]).toBeInstanceOf(GeoTIFFImage);
  });

  it("drops SubIFDs whose aspect ratio does not match the planes", async () => {
    const description = omeDescription(
      omeImage({
        sizeC: 1,
        channels: [{ name: "A" }],
        tiffData: `<TiffData IFD="0" PlaneCount="1"/>`,
      }),
    );
    const subIFDs = [
      { width: 50, height: 50 },
      { width: 300, height: 50 }, // a thumbnail, not part of the pyramid
    ];
    const images = [
      fakeImage({ width: 100, height: 100, description, subIFDs }),
    ];
    const structure = await read(fakeTIFF(images));
    expect(widths(structure)).toEqual([[100, 50]]);
  });

  it("rejects planes with different numbers of SubIFDs", async () => {
    const description = omeDescription(
      omeImage({
        sizeC: 2,
        channels: [{ name: "A" }, { name: "B" }],
        tiffData: `<TiffData IFD="0" PlaneCount="2"/>`,
      }),
    );
    const images = [
      fakeImage({
        width: 100,
        height: 100,
        description,
        subIFDs: [{ width: 50, height: 50 }],
      }),
      fakeImage({ width: 100, height: 100 }), // no SubIFDs
    ];
    await expect(read(fakeTIFF(images))).rejects.toThrow(
      /different numbers of SubIFDs/,
    );
  });

  it("rejects a pyramid whose levels do not shrink", async () => {
    const description = omeDescription(
      omeImage({
        sizeC: 1,
        channels: [{ name: "A" }],
        tiffData: `<TiffData IFD="0" PlaneCount="1"/>`,
      }),
    );
    const images = [
      fakeImage({
        width: 100,
        height: 100,
        description,
        subIFDs: [
          { width: 50, height: 50 },
          { width: 50, height: 50 },
        ],
      }),
    ];
    await expect(read(fakeTIFF(images))).rejects.toThrow(/do not shrink/);
  });
});

describe("PlainTIFFParser", () => {
  it("groups same-sized IFDs into channels and smaller ones into levels", async () => {
    const images = [
      fakeImage({ width: 1000, height: 800 }),
      fakeImage({ width: 1000, height: 800 }),
      fakeImage({ width: 500, height: 400 }),
      fakeImage({ width: 500, height: 399 }), // heights are rounded independently
      fakeImage({ width: 600, height: 200 }), // a macro image
    ];
    const tiff = fakeTIFF(images);
    expect(await findTIFFParser(tiff)).toBeInstanceOf(PlainTIFFParser);
    const structure = await read(tiff);
    expect(widths(structure)).toEqual([
      [1000, 500],
      [1000, 500],
    ]);
    expect(structure.channels).toEqual([{}, {}]);
  });

  it("reads RGB and YCbCr images as images without channels", async () => {
    const rgb = await read(
      fakeTIFF([
        fakeImage({
          width: 1000,
          height: 800,
          samplesPerPixel: 3,
          photometricInterpretation: 2,
        }),
        fakeImage({ width: 250, height: 200, samplesPerPixel: 3 }),
        fakeImage({ width: 500, height: 400, samplesPerPixel: 3 }),
      ]),
    );
    expect(rgb.channels).toBeUndefined();
    expect(widths(rgb)).toEqual([[1000, 500, 250]]);

    const rgba = await read(
      fakeTIFF([
        fakeImage({
          width: 100,
          height: 100,
          samplesPerPixel: 4,
          photometricInterpretation: 2,
        }),
      ]),
    );
    expect(rgba.channels).toBeUndefined();

    const ycbcr = await read(
      fakeTIFF([
        fakeImage({
          width: 100,
          height: 100,
          samplesPerPixel: 3,
          photometricInterpretation: 6,
        }),
      ]),
    );
    expect(ycbcr.channels).toBeUndefined();
  });

  it("leaves palette and white-is-zero images to the tile source", async () => {
    for (const photometricInterpretation of [3, 0]) {
      const structure = await read(
        fakeTIFF([
          fakeImage({ width: 100, height: 100, photometricInterpretation }),
        ]),
      );
      expect(structure.channels).toBeUndefined();
    }
    await expect(
      read(
        fakeTIFF([
          fakeImage({
            width: 100,
            height: 100,
            photometricInterpretation: 3,
          }),
          fakeImage({
            width: 100,
            height: 100,
            photometricInterpretation: 3,
          }),
        ]),
      ),
    ).rejects.toThrow(/palette/);
  });

  it("rejects multi-sample planes that are not an RGB image", async () => {
    await expect(
      read(
        fakeTIFF([
          fakeImage({
            width: 100,
            height: 100,
            samplesPerPixel: 3,
            photometricInterpretation: 1, // BlackIsZero
          }),
        ]),
      ),
    ).rejects.toThrow(/Channel 0 has 3 samples per pixel/);
    await expect(
      read(
        fakeTIFF([
          fakeImage({ width: 100, height: 100, samplesPerPixel: 3 }),
          fakeImage({ width: 100, height: 100, samplesPerPixel: 3 }),
        ]),
      ),
    ).rejects.toThrow(/Channel 0 has 3 samples per pixel/);
  });

  it("tells planes from a label image of the same width", async () => {
    const images = [
      fakeImage({ width: 1000, height: 800 }),
      fakeImage({ width: 1000, height: 800 }),
      fakeImage({ width: 1000, height: 300 }), // a label image
    ];
    const structure = await read(fakeTIFF(images));
    expect(structure.pyramids).toEqual([[images[0]], [images[1]]]);
  });

  it("rejects z and t for files without OME-XML", async () => {
    const images = [fakeImage({ width: 100, height: 100 })];
    await expect(read(fakeTIFF(images), { z: 1 })).rejects.toThrow(/OME-TIFF/);
    await expect(read(fakeTIFF(images), { t: 1 })).rejects.toThrow(/OME-TIFF/);
  });

  it("rejects empty files", async () => {
    await expect(read(fakeTIFF([]))).rejects.toThrow(/no image/);
  });

  describe("on a written file", () => {
    function openTIFF(
      values: Uint16Array | Uint8Array,
      width: number,
      height: number,
      samplesPerPixel = 1,
    ): Promise<GeoTIFF> {
      const buffer = writeArrayBuffer(values, {
        width,
        height,
        SamplesPerPixel: [samplesPerPixel],
        BitsPerSample: Array.from(
          { length: samplesPerPixel },
          () => values.BYTES_PER_ELEMENT * 8,
        ),
        PhotometricInterpretation: samplesPerPixel === 3 ? 2 : 1,
      });
      return fromArrayBuffer(buffer);
    }

    it("reads a single 16-bit plane as one channel", async () => {
      const values = Uint16Array.from({ length: 64 }, (_, i) => i * 100);
      const tiff = await openTIFF(values, 8, 8);
      expect(await findTIFFParser(tiff)).toBeInstanceOf(PlainTIFFParser);
      const structure = await read(tiff);
      expect(structure.pyramids).toEqual([[expect.anything()]]);
      expect(structure.channels).toEqual([{}]);
      expect(structure.pyramids[0]![0]!.getBitsPerSample(0)).toBe(16);
    });

    it("reads a three-sample 8-bit plane as an RGB image without channels", async () => {
      const values = Uint8Array.from({ length: 8 * 8 * 3 }, (_, i) => i % 256);
      const tiff = await openTIFF(values, 8, 8, 3);
      const structure = await read(tiff);
      expect(structure.channels).toBeUndefined();
      expect(structure.pyramids).toEqual([[expect.anything()]]);
    });
  });
});
