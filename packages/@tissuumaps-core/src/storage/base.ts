import type { JsonSchema, UISchemaElement } from "@jsonforms/core";
import type OpenSeadragon from "openseadragon";

import type { AnnotatedDataSource, DataSource } from "../model/base";
import type { IDArray, TypedArray } from "../types/arrays";
import type { ProgressCallback } from "../types/callbacks";
import type {
  CustomTileSource,
  TileSourceConfig,
} from "../types/openseadragon";
import type { TableData } from "./table";

/**
 * Base interface for loaded data objects
 */
export interface Data {
  /** Releases all resources held by this data object */
  close(): void;
}

/**
 * A {@link Data} object that contains an indexed collection of items
 *
 * Extended by data types whose storage is addressable by item IDs
 * (e.g. points, shapes, tables).
 *
 * Implementations have to be immutable: the values returned by their accessors
 * must not change over the lifetime of the object, and the arrays they return
 * have to keep their identity across calls (i.e. be memoized rather than built
 * anew on every call). Consumers rely on that identity to detect change - a
 * renderer that compares the previous `getIds()` result against the current one
 * re-uploads its GPU buffers whenever the two differ - so an implementation
 * returning a fresh array each time is correct but defeats every such cache.
 */
export interface ItemsData extends Data {
  /**
   * Returns an array of item IDs
   *
   * IDs are either integers, held by an integer typed array or by 64-bit
   * floats (safe integers only), or strings, held by a plain array (see
   * `IDArray`). The returned array is owned by this data object: callers must
   * not modify it, and every call returns the very same array (see
   * {@link ItemsData}). IDs are expected to be unique, which is not checked.
   *
   * @returns The item IDs
   */
  getIds(): IDArray;

  /**
   * Returns the total number of items
   *
   * @returns The item count
   */
  getSize(): number;

  /**
   * Returns the item names if available, otherwise undefined
   *
   * Optional: data types that never carry item names need not implement it.
   *
   * The returned array is owned by this data object: callers must not modify it,
   * and every call returns the very same array (see {@link ItemsData}).
   *
   * @returns The item names, or `undefined` if not available
   */
  getNames?: () => string[] | undefined;
}

/**
 * A {@link Data} object that provides a tiled, multi-resolution raster
 *
 * Extended by data types that are drawn as OpenSeadragon tiled images (e.g.
 * images, labels).
 */
export interface RasterData extends Data {
  /**
   * Returns the tile source of this raster
   *
   * Data types providing more than one tile source (see `ImageData`) extend
   * this to address them.
   *
   * @returns The tile source, which can be a URL string, a TileSourceConfig
   * object, or a CustomTileSource object
   */
  getTileSource(): string | TileSourceConfig | CustomTileSource;

  /**
   * Extracts the raw values of an invalidated tile
   *
   * Provided by raster data whose tiles carry values rather than colors (e.g.
   * label IDs, or one sample per pixel of an image channel), which the renderer
   * recolors through a data transfer; raster data whose tiles are drawn as they
   * are does not provide it. The tile in `event` already belongs to one of this
   * raster's tile sources, which identifies the channel for multi-channel data.
   *
   * The raster does not need to be cropped to the tile's source bounds: it may
   * cover the full tile size, as the renderer crops it when drawing.
   *
   * @param event - The tile invalidation event
   * @returns The values of the invalidated tile, one per raster pixel in
   * row-major order, along with the width and height of the raster in pixels
   * @throws Error if the event does not contain raster data
   */
  getTileData?: (
    event: OpenSeadragon.TileInvalidatedEvent,
  ) => Promise<{ values: TypedArray; width: number; height: number }>;
}

/**
 * Options accepted by {@link DataProvider.load}
 */
export type DataProviderLoadOptions = {
  /** Signal for aborting the load operation */
  signal?: AbortSignal;

  /** Directory handle of the open workspace, if any */
  workspace?: FileSystemDirectoryHandle | null;

  /** Callback for reporting the progress of the load operation */
  onProgress?: ProgressCallback;
};

/**
 * Base interface for data providers
 *
 * @typeParam TDataSource - The data source type this data provider opens
 * @typeParam TData - The {@link Data} type produced by this data provider
 * @typeParam TNormalizedDataSource - The data source type produced by
 * {@link DataProvider.normalize} and accepted by {@link DataProvider.load}
 */
export interface DataProvider<
  TDataSource extends DataSource,
  TData extends Data,
  TNormalizedDataSource extends TDataSource = TDataSource,
> {
  /** The name of the data provider */
  readonly name: string;

  /** The JSON schema for the data source */
  readonly schema: JsonSchema;

  /** The JSON Forms UI schema for the data source */
  readonly uischema: UISchemaElement;

  /**
   * Returns the data source with all of this data provider's defaults applied
   * and its source normalized
   *
   * Data sources that normalize to the same value are considered equal, and
   * their data is loaded only once and shared between all referencing objects.
   * Normalization has to be idempotent: normalizing an already normalized data
   * source must return the same value again.
   *
   * The source is normalized with `SourceUtils.normalizeSource`, which resolves
   * it against the project source and the open workspace, so the result has to
   * be recomputed when either changes.
   *
   * Normalizing may throw, e.g. for a source that cannot be resolved against
   * the given workspace and project source. The data cache catches it, so only
   * the objects referencing that data source fail to load; the data of all
   * other objects is unaffected.
   *
   * @param dataSource - The data source to normalize
   * @param workspace - The directory handle of the open workspace, if any
   * @param projectSource - Where the project was loaded from: its absolute URL,
   * the workspace-relative path of the project file, or `null` for projects
   * that were loaded from neither
   * @returns The normalized data source
   */
  normalize(
    dataSource: TDataSource,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
  ): TNormalizedDataSource;

  /**
   * Opens a data source and returns the loaded data accessor
   *
   * The data source has to have been normalized by
   * {@link DataProvider.normalize} beforehand.
   *
   * @param normalizedDataSource - The normalized data source to open
   * @param options - See {@link DataProviderLoadOptions}
   * @returns A promise that resolves to the loaded data accessor
   */
  load(
    normalizedDataSource: TNormalizedDataSource,
    options?: DataProviderLoadOptions,
  ): Promise<TData>;
}

/**
 * Options accepted by {@link AnnotatedDataProvider.load}
 */
export type AnnotatedDataProviderLoadOptions = DataProviderLoadOptions & {
  /** The data of the table referenced by the data source, if any */
  tableDataPromise: Promise<TableData> | undefined;
};

/**
 * Base interface for data providers opening {@link AnnotatedDataSource}s
 *
 * In addition to a {@link DataProvider}, an annotated data provider receives
 * the data of the table referenced by the data source, if any, when loading
 * (see {@link AnnotatedDataProviderLoadOptions}).
 *
 * @typeParam TAnnotatedDataSource - The data source type this data provider
 * opens
 * @typeParam TData - The {@link Data} type produced by this data provider
 * @typeParam TNormalizedAnnotatedDataSource - The data source type produced by
 * {@link DataProvider.normalize} and accepted by
 * {@link AnnotatedDataProvider.load}
 */
export interface AnnotatedDataProvider<
  TAnnotatedDataSource extends AnnotatedDataSource,
  TData extends Data,
  TNormalizedAnnotatedDataSource extends TAnnotatedDataSource =
    TAnnotatedDataSource,
> extends DataProvider<
  TAnnotatedDataSource,
  TData,
  TNormalizedAnnotatedDataSource
> {
  /**
   * Opens a data source and returns the loaded data accessor
   *
   * The data source has to have been normalized by
   * {@link DataProvider.normalize} beforehand.
   *
   * @param normalizedDataSource - The normalized data source to open
   * @param options - See {@link AnnotatedDataProviderLoadOptions}
   * @returns A promise that resolves to the loaded data accessor
   */
  load(
    normalizedDataSource: TNormalizedAnnotatedDataSource,
    options?: AnnotatedDataProviderLoadOptions,
  ): Promise<TData>;
}
