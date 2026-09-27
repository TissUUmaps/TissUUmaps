import {
  ConfigUtils,
  type ConstantConfig,
  type FromConfig,
  type GroupByConfig,
  type GroupValueMap,
  type IDArray,
  MathUtils,
  NumberUtils,
  type SizeConfig,
  type TableData,
  TableUtils,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
} from "@tissuumaps/core";

/**
 * Resolves the size of every item
 */
export class SizeResolver {
  /**
   * Loads size data for a set of IDs based on the active size configuration source
   *
   * Dispatches to the appropriate loader (constant, from, or groupBy) depending on which
   * configuration source is active.
   *
   * @param ids - Ordered list of item IDs
   * @param config - Size configuration specifying the data source
   * @param sizeMaps - Available size maps for groupBy lookups
   * @param defaultSize - Fallback size when no valid config or value is found
   * @param options - Optional abort signal, buffer alignment, and table loader
   * @returns A `Float32Array` of packed size values, one per ID
   */
  static async resolveSizes(
    ids: IDArray,
    config: SizeConfig,
    sizeMaps: GroupValueMap<number>[],
    defaultSize: number,
    options?: {
      signal?: AbortSignal;
      align?: number;
      loadTable?: (options?: { signal?: AbortSignal }) => Promise<TableData>;
    },
  ): Promise<Float32Array> {
    const { signal, align = 1, loadTable } = options ?? {};
    signal?.throwIfAborted();
    const activeConfigSource = getActiveConfigSource(config);
    if (activeConfigSource === "constant" && isConstantConfig(config)) {
      return SizeResolver.resolveUniformSizes(ids, config, {
        align,
      });
    }
    if (
      activeConfigSource === "from" &&
      isFromConfig(config) &&
      loadTable !== undefined
    ) {
      return SizeResolver.resolveSizesFromTableValues(
        ids,
        config,
        defaultSize,
        loadTable,
        { signal, align },
      );
    }
    if (
      activeConfigSource === "groupBy" &&
      isGroupByConfig(config) &&
      loadTable !== undefined
    ) {
      return SizeResolver.resolveSizesFromTableGroups(
        ids,
        config,
        sizeMaps,
        defaultSize,
        loadTable,
        { signal, align },
      );
    }
    console.warn("No valid size config found, using default size");
    return SizeResolver.createUniformSizes(ids.length, defaultSize, {
      align,
    });
  }

  /**
   * Resolves the size all items share if the configuration is a constant
   *
   * The counterpart of {@link resolveSizes} for a constant source, which needs
   * no items: the one packed size applies to every item, so a consumer can
   * supply it once instead of once per item.
   *
   * @param config - Size configuration specifying the data source
   * @returns The packed size, or `undefined` if the active source is not a
   * constant
   */
  static resolveConstantSize(config: SizeConfig): number | undefined {
    const activeConfigSource = getActiveConfigSource(config);
    if (activeConfigSource === "constant" && isConstantConfig(config)) {
      return SizeResolver.packSize(config.constant.value);
    }
    return undefined;
  }

  /**
   * Resolves the size of a single item without loading any table data
   *
   * Synchronous counterpart to {@link resolveSizes} for items that are not
   * known up front, in the way the labels renderer resolves label IDs as they
   * are first drawn (see `ColorResolver.resolveColorWithoutTable`). Labels have
   * no size, so this has no caller yet and exists for symmetry with the other
   * resolvers. A constant source resolves exactly, whereas from and groupBy
   * sources depend on table values and fall back to `defaultSize`.
   *
   * @param _id - The item ID (unused, kept for symmetry with the other resolvers)
   * @param config - Size configuration specifying the data source
   * @param defaultSize - Fallback size when the source cannot be resolved without a table
   * @returns The packed size value
   */
  static resolveSizeWithoutTable(
    _id: number | string,
    config: SizeConfig,
    defaultSize: number,
  ): number {
    return (
      SizeResolver.resolveConstantSize(config) ??
      SizeResolver.packSize(defaultSize)
    );
  }

  /**
   * Creates a uniform size data buffer filled with the configured constant size
   *
   * @param ids - Ordered list of item IDs (only the length is used)
   * @param config - Constant size configuration containing the size value
   * @param options - Optional buffer alignment
   * @returns A `Float32Array` filled with the packed constant size
   */
  static resolveUniformSizes(
    ids: IDArray,
    config: Extract<SizeConfig, ConstantConfig<number>>,
    options?: { align?: number },
  ): Float32Array {
    const { align = 1 } = options ?? {};
    return SizeResolver.createUniformSizes(ids.length, config.constant.value, {
      align,
    });
  }

  /**
   * Loads size data by reading numeric values from a table column
   *
   * @param ids - Ordered list of item IDs
   * @param config - From configuration specifying the source column
   * @param defaultSize - Fallback size when a value is missing or invalid
   * @param loadTable - Async function that loads the {@link TableData}
   * @param options - Optional abort signal and buffer alignment
   * @returns A `Float32Array` of packed size values
   */
  static async resolveSizesFromTableValues(
    ids: IDArray,
    config: Extract<SizeConfig, FromConfig>,
    defaultSize: number,
    loadTable: (options?: { signal?: AbortSignal }) => Promise<TableData>,
    options?: { signal?: AbortSignal; align?: number },
  ): Promise<Float32Array> {
    const { signal, align = 1 } = options ?? {};
    signal?.throwIfAborted();
    const data = await loadTable({ signal });
    const packedSizes = SizeResolver.createSizeBuffer(ids.length, { align });
    await TableUtils.fillFromTableValues(
      packedSizes,
      data,
      ids,
      config.from.column,
      defaultSize,
      (value) => SizeResolver.parseSize(value),
      (size) => SizeResolver.packSize(size),
      { signal },
    );
    return packedSizes;
  }

  /**
   * Loads size data by grouping IDs via a table column and mapping each group
   * to a size value using a size map.
   *
   * @param ids - Ordered list of item IDs
   * @param config - GroupBy configuration specifying the source column and map
   * @param sizeMaps - Available size maps for group-to-size lookups
   * @param defaultSize - Fallback size when the map is not found or a group is unmapped
   * @param loadTable - Async function that loads the {@link TableData}
   * @param options - Optional abort signal and buffer alignment
   * @returns A `Float32Array` of packed size values
   */
  static async resolveSizesFromTableGroups(
    ids: IDArray,
    config: Extract<SizeConfig, GroupByConfig<true>>,
    sizeMaps: GroupValueMap<number>[],
    defaultSize: number,
    loadTable: (options?: { signal?: AbortSignal }) => Promise<TableData>,
    options?: { signal?: AbortSignal; align?: number },
  ): Promise<Float32Array> {
    const { signal, align = 1 } = options ?? {};
    signal?.throwIfAborted();
    const sizeMap = ConfigUtils.findGroupByMap(config, sizeMaps);
    if (sizeMap === undefined) {
      console.warn(
        `Size map ${config.groupBy.map} not found, using default size`,
      );
      return SizeResolver.createUniformSizes(ids.length, defaultSize, {
        align,
      });
    }
    const data = await loadTable({ signal });
    const packedSizes = SizeResolver.createSizeBuffer(ids.length, { align });
    await TableUtils.fillFromTableGroups(
      packedSizes,
      data,
      ids,
      config.groupBy.column,
      sizeMap.default ?? defaultSize,
      ConfigUtils.createGroupValueGetter(config, sizeMap, defaultSize),
      (size) => SizeResolver.packSize(size),
      { signal },
    );
    return packedSizes;
  }

  /**
   * Creates a size data buffer of the given size filled with a single size value
   *
   * @param n - Number of elements
   * @param size - The size value to fill with
   * @param options - Optional buffer alignment
   * @returns A `Float32Array` filled with the packed size value
   */
  static createUniformSizes(
    n: number,
    size: number,
    options?: { align?: number },
  ): Float32Array {
    const { align = 1 } = options ?? {};
    const packedSizes = SizeResolver.createSizeBuffer(n, { align });
    const packedSize = SizeResolver.packSize(size);
    packedSizes.fill(packedSize, 0, n);
    return packedSizes;
  }

  /**
   * Creates a buffer of the given size for storing packed size values, aligned to the specified byte boundary
   *
   * @param n - The number of elements in the buffer
   * @param options - Optional buffer alignment
   * @returns A `Float32Array` of length `n`, aligned to the given byte boundary
   */
  static createSizeBuffer(
    n: number,
    options?: { align?: number },
  ): Float32Array {
    const { align = 1 } = options ?? {};
    const alignedN = MathUtils.align(n, align);
    return new Float32Array(alignedN);
  }

  /**
   * Parses a raw value as a size number
   *
   * @param value - The raw value to parse (must be a number)
   * @returns The numeric size value, or `undefined` if `value` is not a number
   */
  static parseSize(value: unknown): number | undefined {
    return NumberUtils.tryParseFinite(value, { requireSafeBigInt: true });
  }

  /**
   * Packs a size value
   *
   * @param size - The size value to pack
   * @returns The packed size value
   */
  static packSize(size: number): number {
    return size;
  }
}
