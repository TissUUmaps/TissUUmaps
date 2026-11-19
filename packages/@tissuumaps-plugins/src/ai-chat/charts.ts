// Python code can draw charts with matplotlib: the sandbox gives back each
// figure as a PNG image, with what it shows in words. The image is for the
// user only, who can copy and save it; the model gets the words. The panel
// shows an image only once it has checked that it is a PNG: the sandbox runs
// code that the panel does not trust.

/** The most charts that one run of Python code shows */
export const maxCharts = 10;

/** The largest chart that the panel shows, in bytes */
export const maxChartBytes = 5_000_000;

/** The largest width or height of a chart, in pixels */
const maxChartSide = 10_000;

/** The eight bytes that every PNG file starts with */
const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/** The last chunk of every PNG file: its length (0), "IEND", and its CRC */
const endChunk = [0, 0, 0, 0, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82];

/**
 * Checks that an image is a PNG: its signature, then the chunk with its size
 * (IHDR), and the chunk that ends it (IEND)
 *
 * @param buffer - The image
 * @returns Its width and height in pixels, or why it is not a PNG that the
 * panel shows
 */
export function checkPng(
  buffer: ArrayBuffer,
): { width: number; height: number } | { error: string } {
  const bytes = new Uint8Array(buffer);
  if (bytes.length > maxChartBytes) {
    return {
      error: `it is too big (${(bytes.length / 1e6).toFixed(1)} MB, at most ${maxChartBytes / 1e6} MB)`,
    };
  }
  const startsRight =
    bytes.length >= signature.length + 25 + endChunk.length &&
    signature.every((byte, index) => bytes[index] === byte);
  const view = new DataView(buffer);
  // the first chunk: 13 bytes of IHDR, which starts with the size
  const header =
    startsRight &&
    view.getUint32(8) === 13 &&
    String.fromCharCode(...bytes.subarray(12, 16)) === "IHDR";
  const endsRight = endChunk.every(
    (byte, index) => bytes[bytes.length - endChunk.length + index] === byte,
  );
  if (!header || !endsRight) {
    return { error: "it is not a PNG image" };
  }
  const width = view.getUint32(16);
  const height = view.getUint32(20);
  if (
    width === 0 ||
    height === 0 ||
    width > maxChartSide ||
    height > maxChartSide
  ) {
    return {
      error: `its size is not one to show (${width} × ${height} pixels)`,
    };
  }
  return { width, height };
}

/**
 * The name of the file of a chart, for Save
 *
 * @param title - What the code computed, e.g. "cells per cluster"
 * @param number - The number of the chart in its run, from 1
 * @param count - How many charts the run drew
 * @returns e.g. "cells-per-cluster.png" or "cells-per-cluster-2.png"
 */
export function chartFileName(
  title: string,
  number: number,
  count: number,
): string {
  const base =
    title
      .normalize("NFKD")
      .replace(/[̀-ͯ]/g, "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 60)
      .replace(/-+$/, "") || "chart";
  return `${base}${count > 1 ? `-${number}` : ""}.png`;
}

/**
 * Writes an image in base64, as the services want the images that the model
 * sees
 *
 * @param buffer - The image
 * @returns Its bytes in base64
 */
export function toBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let text = "";
  // in pieces, since a call takes only so many arguments
  for (let start = 0; start < bytes.length; start += 0x8000) {
    text += String.fromCharCode(...bytes.subarray(start, start + 0x8000));
  }
  return btoa(text);
}
