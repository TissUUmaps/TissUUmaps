import * as zarr from "zarrita";

/**
 * Reads the consolidated metadata of a Zarr store
 *
 * Zarr v2 consolidates the metadata under a dotted key and Zarr v3 in the
 * root document, which zarrita finds on its own. spatialdata writes the Zarr
 * v2 metadata without the leading dot the spec prescribes, and so do the
 * Vitessce fixtures, so that key is tried as well.
 */
export class ConsolidatedMetadataUtils {
  /** The key spatialdata writes the Zarr v2 consolidated metadata under */
  private static readonly _undottedMetadataKey = "zmetadata";

  /**
   * @param store - Any asynchronous store zarrita can read
   * @param options - Optional abort signal
   * @returns The store with its nodes listed, or `null` if it has no
   * consolidated metadata
   */
  static async open(
    store: zarr.AsyncReadable,
    options?: { signal?: AbortSignal },
  ): Promise<zarr.Listable<zarr.AsyncReadable> | null> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const consolidatedStore = await zarr.withMaybeConsolidatedMetadata(store);
    signal?.throwIfAborted(); // withMaybeConsolidatedMetadata() does not throw on abort
    if ("contents" in consolidatedStore) {
      return consolidatedStore;
    }
    const undottedStore = await zarr.withMaybeConsolidatedMetadata(store, {
      format: "v2",
      metadataKey: ConsolidatedMetadataUtils._undottedMetadataKey,
    });
    signal?.throwIfAborted(); // withMaybeConsolidatedMetadata() does not throw on abort
    return "contents" in undottedStore ? undottedStore : null;
  }
}
