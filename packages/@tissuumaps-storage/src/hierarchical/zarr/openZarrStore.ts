// only this subpath: the fork ships TypeScript sources, and those of the other
// stores fail our type-checking
import FileSystemHandleStore from "@zarrita/storage/fs-handle";
import * as zarr from "zarrita";

import { SourceUtils } from "@tissuumaps/core";

import { ZarrStore } from "./ZarrStore";

/**
 * Opens the Zarr store a data source points to
 *
 * The source may point at a group inside a store, such as the `tables/<name>`
 * of a SpatialData store, whose consolidated metadata lives at the store root.
 * The source and then each of its ancestors are tried as the store root; the
 * path below the first one that opens becomes the root of the table.
 *
 * A URL is read with fetch requests, a directory in the open workspace through
 * its handle (see `FileSystemHandleStore`).
 *
 * @param normalizedSource - The normalized source of the data source to open
 * @param options - `signal` aborts the load; `workspace` is the directory
 * handle of the open workspace, required for workspace-relative sources
 * @returns The store, rooted at the group the source points to
 * @throws Error if neither the source nor any of its ancestors opens as a
 * Zarr store with consolidated metadata; the last failure is the cause
 */
export async function openZarrStore(
  normalizedSource: string,
  options?: {
    signal?: AbortSignal;
    workspace?: FileSystemDirectoryHandle | null;
  },
): Promise<ZarrStore> {
  const { signal, workspace = null } = options ?? {};
  signal?.throwIfAborted();
  let rootSource = normalizedSource;
  const groupSegments: string[] = [];
  let lastError: unknown;
  for (;;) {
    try {
      const resolvedRootSource = await SourceUtils.resolveSourceDirectory(
        rootSource,
        workspace,
        { signal },
      );
      const store =
        typeof resolvedRootSource === "string"
          ? new zarr.FetchStore(resolvedRootSource)
          : new FileSystemHandleStore(resolvedRootSource);
      return await ZarrStore.open(store, groupSegments.join("/"), { signal });
    } catch (error) {
      signal?.throwIfAborted();
      lastError = error;
    }
    const parent = SourceUtils.getParentSource(rootSource);
    if (parent === null) {
      break;
    }
    rootSource = parent.parentSource;
    groupSegments.unshift(parent.name);
  }
  throw new Error(
    `No Zarr store with consolidated metadata found at "${normalizedSource}" or above it.`,
    { cause: lastError },
  );
}
