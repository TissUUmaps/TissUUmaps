import FileSystemHandleStore from "@zarrita/storage/fs-handle";
import * as zarr from "zarrita";

import { SourceUtils } from "@tissuumaps/core";

/**
 * Reads what OME-Zarr data sources are and what their images are named,
 * without opening them
 *
 * The OME-NGFF attributes of a Zarr group are stored under `ome` in its
 * attributes (OME-Zarr 0.5, Zarr v3), or at the top level of its attributes
 * (OME-Zarr 0.4 and earlier, Zarr v2).
 */
export class OMEZarrUtils {
  /** The extension of zipped OME-Zarr files */
  private static readonly _zipExtension = ".ozx";

  /** The extension of Zarr stores, of which any segment of a source may be */
  private static readonly _storeExtension = ".zarr";

  /**
   * Returns whether a normalized source is a zipped OME-Zarr file, judged by
   * its extension
   *
   * @param normalizedSource - The normalized source
   * @returns `true` if its last segment has the zipped OME-Zarr extension
   *   (see {@link OMEZarrUtils._zipExtension})
   */
  static isZipSource(normalizedSource: string): boolean {
    return (
      SourceUtils.getExtension(normalizedSource) === OMEZarrUtils._zipExtension
    );
  }

  /**
   * Returns whether a normalized source lies within a Zarr store, judged by
   * the extensions of its segments
   *
   * A source may point at a group below the store root, such as
   * `x.zarr/labels/cells` or `x.zarr/images/a` of a SpatialData store.
   *
   * @param normalizedSource - The normalized source
   * @returns `true` if any of its segments has the Zarr store extension (see
   *   {@link OMEZarrUtils._storeExtension})
   */
  static isStoreSource(normalizedSource: string): boolean {
    return SourceUtils.getPathSegments(normalizedSource).some((segment) =>
      segment.toLowerCase().endsWith(OMEZarrUtils._storeExtension),
    );
  }

  /**
   * Reads the name of the OME-Zarr image a normalized source points to, from
   * its first multiscales
   *
   * Zipped OME-Zarr files (see {@link OMEZarrUtils.isZipSource}) and sources
   * outside Zarr stores (see {@link OMEZarrUtils.isStoreSource}) are not read.
   * Blank names are ignored, and names are trimmed.
   *
   * @param normalizedSource - The normalized source
   * @param workspace - The directory handle of the open workspace, if any
   * @param options - Optional abort signal
   * @returns A promise that resolves to the name, or to `undefined` if the
   *   image has none
   * @throws See {@link OMEZarrUtils.readAttributes}
   */
  static async readImageName(
    normalizedSource: string,
    workspace: FileSystemDirectoryHandle | null,
    options?: { signal?: AbortSignal },
  ): Promise<string | undefined> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    if (
      OMEZarrUtils.isZipSource(normalizedSource) ||
      !OMEZarrUtils.isStoreSource(normalizedSource)
    ) {
      return undefined;
    }
    const attributes = await OMEZarrUtils.readAttributes(
      normalizedSource,
      workspace,
      options,
    );
    const multiscales = attributes?.multiscales;
    const name: unknown = Array.isArray(multiscales)
      ? (multiscales[0] as { name?: unknown } | undefined)?.name
      : undefined;
    if (typeof name !== "string") {
      return undefined;
    }
    return name.trim() || undefined;
  }

  /**
   * Reads the OME-NGFF attributes of the Zarr group a normalized source
   * points to
   *
   * Zarr v2 and v3 groups are both read, through zarrita. Only the group
   * itself is read, none of its children.
   *
   * @param normalizedSource - The normalized source of a Zarr group: a URL or
   *   a directory in the workspace
   * @param workspace - The directory handle of the open workspace, if any
   * @param options - Optional abort signal
   * @returns A promise that resolves to the OME-NGFF attributes, or to
   *   `undefined` if the source is no Zarr group
   * @throws Error if the source cannot be resolved to a directory or read, or
   *   if the operation is aborted
   */
  static async readAttributes(
    normalizedSource: string,
    workspace: FileSystemDirectoryHandle | null,
    options?: { signal?: AbortSignal },
  ): Promise<Record<string, unknown> | undefined> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const directory = await SourceUtils.resolveSourceDirectory(
      normalizedSource,
      workspace,
      { signal },
    );
    const store =
      typeof directory === "string"
        ? new zarr.FetchStore(directory)
        : new FileSystemHandleStore(directory);
    let attributes: Record<string, unknown>;
    try {
      attributes = (await zarr.open(store, { kind: "group", signal })).attrs;
    } catch (error) {
      if (zarr.isZarritaError(error, "NotFoundError")) {
        return undefined;
      }
      throw error;
    }
    const { ome } = attributes;
    return typeof ome === "object" && ome !== null && !Array.isArray(ome)
      ? (ome as Record<string, unknown>)
      : attributes;
  }

  /**
   * Inspects whether a normalized source is an OME-Zarr image, and whether it
   * is a label image
   *
   * A zipped OME-Zarr file is judged by its extension only (see
   * {@link OMEZarrUtils.isZipSource}), as it is not read, and is taken for no
   * label image. A source within a Zarr store (see
   * {@link OMEZarrUtils.isStoreSource}) has to be a group with multiscales (see
   * {@link OMEZarrUtils.readAttributes}); it is a label image if it has
   * `image-label` attributes, which OME-NGFF recommends but does not require
   * for label images.
   *
   * @param normalizedSource - The normalized source
   * @param workspace - The directory handle of the open workspace, if any
   * @param options - Optional abort signal
   * @returns A promise that resolves to `undefined` if the source is no
   *   OME-Zarr image, and otherwise to whether it is a label image
   * @throws See {@link OMEZarrUtils.readAttributes}; Error if the operation is
   *   aborted
   */
  static async inspectImage(
    normalizedSource: string,
    workspace: FileSystemDirectoryHandle | null,
    options?: { signal?: AbortSignal },
  ): Promise<{ labels: boolean } | undefined> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    if (OMEZarrUtils.isZipSource(normalizedSource)) {
      return { labels: false };
    }
    if (!OMEZarrUtils.isStoreSource(normalizedSource)) {
      return undefined;
    }
    const attributes = await OMEZarrUtils.readAttributes(
      normalizedSource,
      workspace,
      options,
    );
    if (attributes === undefined || !("multiscales" in attributes)) {
      return undefined;
    }
    return { labels: "image-label" in attributes };
  }
}
