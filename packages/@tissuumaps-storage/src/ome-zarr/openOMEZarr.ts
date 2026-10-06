// only this subpath: the fork ships TypeScript sources, and those of the other
// stores fail our type-checking
import FileSystemHandleStore from "@zarrita/storage/fs-handle";
import { type OMEZarr, OMEZarrTileSource } from "omezarr-tilesource";

import { SourceUtils } from "@tissuumaps/core";

import { OMEZarrUtils } from "./OMEZarrUtils";

/**
 * Opens the OME-Zarr image a data source points to, loading its metadata and
 * the arrays of all its resolution levels with `OMEZarrTileSource.loadOMEZarr`
 *
 * The normalized source is resolved with `SourceUtils.resolveSource`: a URL is
 * opened as a remote zipped OME-Zarr file if it has the zipped OME-Zarr
 * extension in any case (see `OMEZarrUtils.isZipSource`), and as a remote
 * OME-Zarr store otherwise; a file in the open workspace as a zipped OME-Zarr
 * file, and a directory in the open workspace as an OME-Zarr store (see
 * `FileSystemHandleStore`).
 *
 * The tile sources opened for the image share the loaded image instead of
 * loading it from its URL (see `OMEZarrTileSource.open`), but need an absolute
 * URL for their tile cache keys: sources within the workspace get a `urn:uuid:`
 * URL that is unique to the load.
 *
 * @param normalizedSource - The normalized source of the data source to open
 * @param options - `signal` aborts the load; `workspace` is the directory
 * handle of the open workspace, required for workspace-relative sources
 * @returns A promise that resolves to the loaded image and arrays, and to the
 * absolute URL and the `zip` flag to open its tile sources with
 * @throws Error if the source is workspace-relative while no workspace is open
 */
export async function openOMEZarr(
  normalizedSource: string,
  options?: {
    signal?: AbortSignal;
    workspace?: FileSystemDirectoryHandle | null;
  },
): Promise<{ loaded: OMEZarr; url: string; zip: boolean }> {
  const { signal, workspace = null } = options ?? {};
  signal?.throwIfAborted();
  const resolvedSource = await SourceUtils.resolveSource(
    normalizedSource,
    workspace,
    { signal },
  );
  if (typeof resolvedSource === "string") {
    const zip = OMEZarrUtils.isZipSource(normalizedSource);
    const loaded = await OMEZarrTileSource.loadOMEZarr(resolvedSource, zip, {
      signal,
    });
    return { loaded, url: resolvedSource, zip };
  }
  const url = `urn:uuid:${crypto.randomUUID()}`;
  if (resolvedSource.kind === "file") {
    const file = await resolvedSource.getFile();
    signal?.throwIfAborted(); // getFile() does not throw on abort
    const loaded = await OMEZarrTileSource.loadOMEZarr(file, true, { signal });
    return { loaded, url, zip: true };
  }
  const loaded = await OMEZarrTileSource.loadOMEZarr(
    new FileSystemHandleStore(resolvedSource),
    false,
    { signal },
  );
  return { loaded, url, zip: false };
}
