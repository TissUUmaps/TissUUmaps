import {
  type BlockedSourceOptions,
  type GeoTIFF,
  type RemoteSourceOptions,
  fromBlob,
  fromUrl,
} from "geotiff";

import { SourceUtils } from "@tissuumaps/core";

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
 * Opens the TIFF file a data source points to
 *
 * The normalized source is resolved with `SourceUtils.openSourceFile`:
 * files in the open workspace are read through their file handle, remote files
 * over HTTP range requests (see {@link remoteSourceOptions}). Opening a file
 * reads nothing but its header.
 *
 * @param normalizedSource - The normalized source of the data source to open
 * @param options - `signal` aborts the load; `workspace` is the directory
 * handle of the open workspace, required for workspace-relative sources
 * @returns The opened file
 * @throws Error if the source is workspace-relative while no workspace is open
 */
export async function openTIFF(
  normalizedSource: string,
  options?: {
    signal?: AbortSignal;
    workspace?: FileSystemDirectoryHandle | null;
  },
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
