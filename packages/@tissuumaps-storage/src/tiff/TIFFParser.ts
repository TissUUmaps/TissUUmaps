import type { GeoTIFF, GeoTIFFImage } from "geotiff";

import type { Color } from "@tissuumaps/core";

import { OMETIFFParser } from "./profiles/OMETIFFParser";
import { PlainTIFFParser } from "./profiles/PlainTIFFParser";
import { QPTIFFParser } from "./profiles/QPTIFFParser";

/** What a parser reads about a channel of a multi-channel TIFF file */
export type TIFFChannelMetadata = {
  name?: string;
  color?: Color;
};

/** The pyramid and channel metadata of a TIFF file */
export type TIFFStructure = {
  /**
   * One pyramid per channel, largest image first; a single one for files that
   * are drawn in their own colors
   */
  pyramids: GeoTIFFImage[][];

  /**
   * One entry per pyramid, empty if the file has no name or color for it.
   * `undefined` for files that are drawn in their own colors.
   */
  channels: TIFFChannelMetadata[] | undefined;
};

/** Reads the structure of one TIFF profile */
export interface TIFFParser {
  /**
   * Whether this parser reads the file
   *
   * @param tiff - The opened file
   * @param options - An abort signal
   * @returns Whether the file is of this parser's format
   */
  supports(tiff: GeoTIFF, options?: { signal?: AbortSignal }): Promise<boolean>;

  /**
   * Reads the structure of the file
   *
   * @param tiff - The opened file
   * @param options - The z-slice and timepoint (default `0`, only OME-TIFF
   * has them), and an abort signal
   * @returns The structure
   */
  load(
    tiff: GeoTIFF,
    options?: { z?: number; t?: number; signal?: AbortSignal },
  ): Promise<TIFFStructure>;
}

/**
 * The parsers in detection order
 *
 * The plain parser accepts any file, so it has to stay last: ahead of another
 * parser it would claim that parser's files and drop their channel names and
 * colors, without failing.
 */
export const tiffParsers: TIFFParser[] = [
  new OMETIFFParser(),
  new QPTIFFParser(),
  new PlainTIFFParser(),
];

/**
 * Finds the parser for a file
 *
 * @param tiff - The opened file
 * @param options - An abort signal
 * @returns The first parser that supports the file
 */
export async function findTIFFParser(
  tiff: GeoTIFF,
  options?: { signal?: AbortSignal },
): Promise<TIFFParser> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  for (const parser of tiffParsers) {
    if (await parser.supports(tiff, { signal })) {
      return parser;
    }
  }
  throw new Error("No parser supports the TIFF file.");
}
