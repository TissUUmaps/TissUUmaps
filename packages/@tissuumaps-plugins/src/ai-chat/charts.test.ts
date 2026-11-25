import { describe, expect, it } from "vitest";

import { chartFileName, checkPng, maxChartBytes } from "./charts";

/** A PNG image of this size: its signature, IHDR and IEND, without pixels */
function pngImage(width: number, height: number, extra = 0): ArrayBuffer {
  const bytes = new Uint8Array(8 + 25 + extra + 12);
  bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const view = new DataView(bytes.buffer);
  view.setUint32(8, 13);
  bytes.set([0x49, 0x48, 0x44, 0x52], 12);
  view.setUint32(16, width);
  view.setUint32(20, height);
  bytes.set([8, 6, 0, 0, 0], 24);
  bytes.set(
    [0, 0, 0, 0, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82],
    33 + extra,
  );
  return bytes.buffer;
}

describe("checkPng", () => {
  it("gives the size of a PNG image", () => {
    expect(checkPng(pngImage(640, 480))).toEqual({ width: 640, height: 480 });
  });

  it("refuses what is not a whole PNG image", () => {
    const text = (value: string) => new TextEncoder().encode(value).buffer;
    const notPng = { error: "it is not a PNG image" };
    expect(checkPng(text("<svg onload='alert(1)'></svg>"))).toEqual(notPng);
    expect(checkPng(new ArrayBuffer(0))).toEqual(notPng);
    // cut before its end
    expect(checkPng(pngImage(640, 480).slice(0, 40))).toEqual(notPng);
    // another first chunk than IHDR
    const other = new Uint8Array(pngImage(640, 480));
    other.set([0x74, 0x45, 0x58, 0x74], 12);
    expect(checkPng(other.buffer)).toEqual(notPng);
    // a JPEG
    const jpeg = new Uint8Array(pngImage(640, 480));
    jpeg.set([0xff, 0xd8, 0xff, 0xe0]);
    expect(checkPng(jpeg.buffer)).toEqual(notPng);
  });

  it("refuses images too big to show", () => {
    expect(checkPng(pngImage(0, 480))).toEqual({
      error: "its size is not one to show (0 × 480 pixels)",
    });
    expect(checkPng(pngImage(20_000, 480))).toEqual({
      error: "its size is not one to show (20000 × 480 pixels)",
    });
    expect(checkPng(pngImage(640, 480, maxChartBytes))).toEqual({
      error: "it is too big (5.0 MB, at most 5 MB)",
    });
  });
});

describe("chartFileName", () => {
  it("names the file after what the code computed", () => {
    expect(chartFileName("Cells per cluster", 1, 1)).toBe(
      "cells-per-cluster.png",
    );
    expect(chartFileName("Z-scores of the leiden clusters!", 2, 3)).toBe(
      "z-scores-of-the-leiden-clusters-2.png",
    );
    expect(chartFileName("Expression über Zeit", 1, 1)).toBe(
      "expression-uber-zeit.png",
    );
    expect(chartFileName("???", 1, 1)).toBe("chart.png");
    expect(chartFileName("a".repeat(100), 1, 1)).toBe(`${"a".repeat(60)}.png`);
  });
});
