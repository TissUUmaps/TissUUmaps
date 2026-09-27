import {
  ColorUtils,
  ConfigUtils,
  type ConstantConfig,
  type FromConfig,
  type GroupByConfig,
  type GroupValueMap,
  type IDArray,
  MathUtils,
  NumberUtils,
  type OpacityConfig,
  type TableData,
  TableUtils,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
} from "@tissuumaps/core";

/**
 * Resolves the opacity of every item, packed as a byte in `[0, 255]`
 */
export class OpacityResolver {
  /**
   * Loads opacity data for a set of IDs based on the active opacity configuration source
   *
   * Dispatches to the appropriate loader (constant, from, or groupBy) depending on which
   * configuration source is active.
   *
   * @param ids - Ordered list of item IDs
   * @param config - Opacity configuration specifying the data source
   * @param opacityMaps - Available opacity maps for groupBy lookups
   * @param defaultOpacity - Fallback opacity value (0–1) when no valid config or value is found
   * @param options - Optional abort signal, buffer alignment, and table loader
   * @returns A `Uint8Array` of packed opacity values (0–255), one per ID
   */
  static async resolveOpacities(
    ids: IDArray,
    config: OpacityConfig,
    opacityMaps: GroupValueMap<number>[],
    defaultOpacity: number,
    options?: {
      signal?: AbortSignal;
      align?: number;
      loadTable?: (options?: { signal?: AbortSignal }) => Promise<TableData>;
    },
  ): Promise<Uint8Array> {
    const { signal, align = 1, loadTable } = options ?? {};
    signal?.throwIfAborted();
    const activeConfigSource = getActiveConfigSource(config);
    if (activeConfigSource === "constant" && isConstantConfig(config)) {
      return OpacityResolver.resolveUniformOpacities(ids, config, {
        align,
      });
    }
    if (
      activeConfigSource === "from" &&
      isFromConfig(config) &&
      loadTable !== undefined
    ) {
      return OpacityResolver.resolveOpacitiesFromTableValues(
        ids,
        config,
        defaultOpacity,
        loadTable,
        { signal, align },
      );
    }
    if (
      activeConfigSource === "groupBy" &&
      isGroupByConfig(config) &&
      loadTable !== undefined
    ) {
      return OpacityResolver.resolveOpacitiesFromTableGroups(
        ids,
        config,
        opacityMaps,
        defaultOpacity,
        loadTable,
        { signal, align },
      );
    }
    console.warn("No valid opacity config found, using default opacity");
    return OpacityResolver.createUniformOpacities(ids.length, defaultOpacity, {
      align,
    });
  }

  /**
   * Resolves the opacity all items share if the configuration is a constant
   *
   * The counterpart of {@link resolveOpacities} for a constant source, which needs
   * no items: the one packed opacity applies to every item, so a consumer can
   * supply it once instead of once per item.
   *
   * @param config - Opacity configuration specifying the data source
   * @returns The packed opacity, or `undefined` if the active source is not a
   * constant
   */
  static resolveConstantOpacity(config: OpacityConfig): number | undefined {
    const activeConfigSource = getActiveConfigSource(config);
    if (activeConfigSource === "constant" && isConstantConfig(config)) {
      return OpacityResolver.packOpacity(config.constant.value);
    }
    return undefined;
  }

  /**
   * Resolves the opacity of a single item without loading any table data
   *
   * Synchronous counterpart to {@link resolveOpacities} for items that are
   * not known up front: the labels renderer uses it for label IDs as they are
   * first drawn, since a label image does not enumerate its labels. A constant
   * source resolves exactly, whereas from and groupBy sources depend on table
   * values and fall back to `defaultOpacity`.
   *
   * @param _id - The item ID (unused, kept for symmetry with the other resolvers)
   * @param config - Opacity configuration specifying the data source
   * @param defaultOpacity - Fallback opacity when the source cannot be resolved without a table
   * @returns The packed opacity as an integer in the range [0, 255]
   */
  static resolveOpacityWithoutTable(
    _id: number | string,
    config: OpacityConfig,
    defaultOpacity: number,
  ): number {
    return (
      OpacityResolver.resolveConstantOpacity(config) ??
      OpacityResolver.packOpacity(defaultOpacity)
    );
  }

  /**
   * Creates a uniform opacity data buffer filled with the configured constant opacity
   *
   * @param ids - Ordered list of item IDs (only the length is used)
   * @param config - Constant opacity configuration containing the opacity value
   * @param options - Optional buffer alignment
   * @returns A `Uint8Array` filled with the packed constant opacity
   */
  static resolveUniformOpacities(
    ids: IDArray,
    config: Extract<OpacityConfig, ConstantConfig<number>>,
    options?: { align?: number },
  ): Uint8Array {
    const { align = 1 } = options ?? {};
    return OpacityResolver.createUniformOpacities(
      ids.length,
      config.constant.value,
      { align },
    );
  }

  /**
   * Loads opacity data by reading numeric values from a table column
   *
   * @param ids - Ordered list of item IDs
   * @param config - From configuration specifying the source column
   * @param defaultOpacity - Fallback opacity when a value is missing or invalid
   * @param loadTable - Async function that loads the {@link TableData}
   * @param options - Optional abort signal and buffer alignment
   * @returns A `Uint8Array` of packed opacity values
   */
  static async resolveOpacitiesFromTableValues(
    ids: IDArray,
    config: Extract<OpacityConfig, FromConfig>,
    defaultOpacity: number,
    loadTable: (options?: { signal?: AbortSignal }) => Promise<TableData>,
    options?: { signal?: AbortSignal; align?: number },
  ): Promise<Uint8Array> {
    const { signal, align = 1 } = options ?? {};
    signal?.throwIfAborted();
    const data = await loadTable({ signal });
    const packedOpacities = OpacityResolver.createOpacityBuffer(ids.length, {
      align,
    });
    await TableUtils.fillFromTableValues(
      packedOpacities,
      data,
      ids,
      config.from.column,
      defaultOpacity,
      (value) => OpacityResolver.parseOpacity(value),
      (opacity) => OpacityResolver.packOpacity(opacity),
      { signal },
    );
    return packedOpacities;
  }

  /**
   * Loads opacity data by grouping IDs via a table column and mapping each group
   * to an opacity value using an opacity map.
   *
   * @param ids - Ordered list of item IDs
   * @param config - GroupBy configuration specifying the source column and map
   * @param opacityMaps - Available opacity maps for group-to-opacity lookups
   * @param defaultOpacity - Fallback opacity when the map is not found or a group is unmapped
   * @param loadTable - Async function that loads the {@link TableData}
   * @param options - Optional abort signal and buffer alignment
   * @returns A `Uint8Array` of packed opacity values
   */
  static async resolveOpacitiesFromTableGroups(
    ids: IDArray,
    config: Extract<OpacityConfig, GroupByConfig<true>>,
    opacityMaps: GroupValueMap<number>[],
    defaultOpacity: number,
    loadTable: (options?: { signal?: AbortSignal }) => Promise<TableData>,
    options?: { signal?: AbortSignal; align?: number },
  ): Promise<Uint8Array> {
    const { signal, align = 1 } = options ?? {};
    signal?.throwIfAborted();
    const opacityMap = ConfigUtils.findGroupByMap(config, opacityMaps);
    if (opacityMap === undefined) {
      console.warn(
        `Opacity map ${config.groupBy.map} not found, using default opacity`,
      );
      return OpacityResolver.createUniformOpacities(
        ids.length,
        defaultOpacity,
        { align },
      );
    }
    const data = await loadTable({ signal });
    const packedOpacities = OpacityResolver.createOpacityBuffer(ids.length, {
      align,
    });
    await TableUtils.fillFromTableGroups(
      packedOpacities,
      data,
      ids,
      config.groupBy.column,
      opacityMap.default ?? defaultOpacity,
      ConfigUtils.createGroupValueGetter(config, opacityMap, defaultOpacity),
      (opacity) => OpacityResolver.packOpacity(opacity),
      { signal },
    );
    return packedOpacities;
  }

  /**
   * Creates an opacity data buffer of the given size filled with a single opacity value
   *
   * @param n - Number of elements
   * @param opacity - The opacity value (0–1) to fill with
   * @param options - Optional buffer alignment
   * @returns A `Uint8Array` filled with the packed opacity
   */
  static createUniformOpacities(
    n: number,
    opacity: number,
    options?: { align?: number },
  ): Uint8Array {
    const { align = 1 } = options ?? {};
    const packedOpacities = OpacityResolver.createOpacityBuffer(n, { align });
    const packedOpacity = OpacityResolver.packOpacity(opacity);
    packedOpacities.fill(packedOpacity, 0, n);
    return packedOpacities;
  }

  /**
   * Creates a buffer of the given size for storing packed opacity values, aligned to the specified byte boundary
   *
   * @param n - The number of elements in the buffer
   * @param options - Optional buffer alignment
   * @returns A `Uint8Array` of length `n`, aligned to the given byte boundary
   */
  static createOpacityBuffer(
    n: number,
    options?: { align?: number },
  ): Uint8Array {
    const { align = 1 } = options ?? {};
    const alignedN = MathUtils.align(n, align);
    return new Uint8Array(alignedN);
  }

  /**
   * Parses a raw value as an opacity number, clamped to the range [0, 1]
   *
   * @param value - The raw value to parse (must be a number)
   * @returns The clamped opacity, or `undefined` if `value` is not a number
   */
  static parseOpacity(value: unknown): number | undefined {
    const opacity = NumberUtils.tryParseFinite(value, {
      requireSafeBigInt: true,
    });
    return opacity !== undefined ? MathUtils.clamp(opacity, 0, 1) : undefined;
  }

  /**
   * Packs an opacity value (0–1) into a `Uint8Array`-compatible integer (0–255)
   *
   * @param opacity - The opacity value (0–1)
   * @returns The packed opacity as an integer in the range [0, 255]
   */
  static packOpacity(opacity: number): number {
    return ColorUtils.packOpacity(opacity);
  }
}
