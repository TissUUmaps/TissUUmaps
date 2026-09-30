import {
  type BlockedSourceOptions,
  type GeoTIFF,
  type RemoteSourceOptions,
  fromBlob,
  fromUrl,
} from "geotiff";

import { type DataProviderLoadOptions, SourceUtils } from "@tissuumaps/core";

import { type TIFFStructure, findTIFFParser } from "./TIFFParser";

/**
 * Remote files are read in 64 KiB blocks, of which 256 (16 MiB) are cached per
 * open file. Without blocks, geotiff.js sends one range request per tag value
 * and per strip.
 */
const remoteSourceOptions: RemoteSourceOptions & BlockedSourceOptions = {
  blockSize: 65536,
  cacheSize: 256,
};

/**
 * Opens the TIFF file a data source points to and reads its structure
 *
 * The normalized source is resolved with `SourceUtils.openSourceFile`:
 * files in the open workspace are read through their file handle, remote files
 * over HTTP range requests (see {@link remoteSourceOptions}).
 *
 * Opening a file reads nothing but its header. Its structure is then read by
 * the parser of its format (see `findTIFFParser`), with `z` and `t` selecting
 * the plane; its pixels are read on demand.
 *
 * @param normalizedSource - The normalized source of the data source to open
 * @param options - The plane to read (`z` and `t`, default `0`, only OME-TIFF
 * has them), and `DataProviderLoadOptions`; `workspace` is required for
 * workspace-relative sources
 * @returns The opened file and its structure
 * @throws Error if the source is workspace-relative while no workspace is
 * open, or if the file holds a TIFF no parser recognizes
 */
export async function openTIFF(
  normalizedSource: string,
  options?: DataProviderLoadOptions & { z?: number; t?: number },
): Promise<TIFFStructure & { tiff: GeoTIFF }> {
  const { z, t, signal } = options ?? {};
  const tiff = await openFile(normalizedSource, options);
  const parser = await findTIFFParser(tiff, { signal });
  const structure = await parser.load(tiff, { z, t, signal });
  return { ...structure, tiff };
}

/**
 * Opens the TIFF file a data source points to
 *
 * @param normalizedSource - The normalized source of the data source to open
 * @param options - See `DataProviderLoadOptions`; `workspace` is required for
 * workspace-relative sources
 * @returns The opened file
 * @throws Error if the source is workspace-relative while no workspace is open
 */
async function openFile(
  normalizedSource: string,
  options?: DataProviderLoadOptions,
): Promise<GeoTIFF> {
  const { signal, workspace = null } = options ?? {};
  signal?.throwIfAborted();
  const source = await SourceUtils.openSourceFile(normalizedSource, workspace, {
    signal,
  });
  if (source.url !== undefined) {
    return await fromUrl(source.url, remoteSourceOptions, signal);
  }
  return await fromBlob(source.file, signal);
}
