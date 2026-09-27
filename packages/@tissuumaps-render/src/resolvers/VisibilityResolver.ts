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
  type TableData,
  TableUtils,
  type VisibilityConfig,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
} from "@tissuumaps/core";

/**
 * Resolves the visibility of every item, packed as `0` or `1`
 */
export class VisibilityResolver {
  /**
   * Loads visibility data for a set of IDs based on the active visibility configuration source
   *
   * Dispatches to the appropriate loader (constant, from, or groupBy) depending on which
   * configuration source is active.
   *
   * @param ids - Ordered list of item IDs
   * @param config - Visibility configuration specifying the data source
   * @param visibilityMaps - Available visibility maps for groupBy lookups
   * @param defaultVisibility - Fallback visibility when no valid config or value is found
   * @param options - Optional abort signal, buffer alignment, and table loader
   * @returns A `Uint8Array` of packed visibility values (0 or 1), one per ID
   */
  static async resolveVisibilities(
    ids: IDArray,
    config: VisibilityConfig,
    visibilityMaps: GroupValueMap<boolean>[],
    defaultVisibility: boolean,
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
      return VisibilityResolver.resolveUniformVisibilities(ids, config, {
        align,
      });
    }
    if (
      activeConfigSource === "from" &&
      isFromConfig(config) &&
      loadTable !== undefined
    ) {
      return VisibilityResolver.resolveVisibilitiesFromTableValues(
        ids,
        config,
        defaultVisibility,
        loadTable,
        { signal, align },
      );
    }
    if (
      activeConfigSource === "groupBy" &&
      isGroupByConfig(config) &&
      loadTable !== undefined
    ) {
      return VisibilityResolver.resolveVisibilitiesFromTableGroups(
        ids,
        config,
        visibilityMaps,
        defaultVisibility,
        loadTable,
        { signal, align },
      );
    }
    console.warn("No valid visibility config found, using default visibility");
    return VisibilityResolver.createUniformVisibilities(
      ids.length,
      defaultVisibility,
      { align },
    );
  }

  /**
   * Resolves the visibility all items share if the configuration is a constant
   *
   * The counterpart of {@link resolveVisibilities} for a constant source, which needs
   * no items: the one packed visibility applies to every item, so a consumer can
   * supply it once instead of once per item.
   *
   * @param config - Visibility configuration specifying the data source
   * @returns The packed visibility, or `undefined` if the active source is not a
   * constant
   */
  static resolveConstantVisibility(
    config: VisibilityConfig,
  ): number | undefined {
    const activeConfigSource = getActiveConfigSource(config);
    if (activeConfigSource === "constant" && isConstantConfig(config)) {
      return VisibilityResolver.packVisibility(config.constant.value);
    }
    return undefined;
  }

  /**
   * Resolves the visibility of a single item without loading any table data
   *
   * Synchronous counterpart to {@link resolveVisibilities} for items that are
   * not known up front: the labels renderer uses it for label IDs as they are
   * first drawn, since a label image does not enumerate its labels. A constant
   * source resolves exactly, whereas from and groupBy sources depend on table
   * values and fall back to `defaultVisibility`.
   *
   * @param _id - The item ID (unused, kept for symmetry with the other resolvers)
   * @param config - Visibility configuration specifying the data source
   * @param defaultVisibility - Fallback visibility when the source cannot be resolved without a table
   * @returns The packed visibility (`0` or `1`)
   */
  static resolveVisibilityWithoutTable(
    _id: number | string,
    config: VisibilityConfig,
    defaultVisibility: boolean,
  ): number {
    return (
      VisibilityResolver.resolveConstantVisibility(config) ??
      VisibilityResolver.packVisibility(defaultVisibility)
    );
  }

  /**
   * Creates a uniform visibility data buffer filled with the configured constant visibility
   *
   * @param ids - Ordered list of item IDs (only the length is used)
   * @param config - Constant visibility configuration containing the boolean value
   * @param options - Optional buffer alignment
   * @returns A `Uint8Array` filled with the packed constant visibility
   */
  static resolveUniformVisibilities(
    ids: IDArray,
    config: Extract<VisibilityConfig, ConstantConfig<boolean>>,
    options?: { align?: number },
  ): Uint8Array {
    const { align = 1 } = options ?? {};
    return VisibilityResolver.createUniformVisibilities(
      ids.length,
      config.constant.value,
      { align },
    );
  }

  /**
   * Loads visibility data by reading values from a table column and parsing them as booleans
   *
   * @param ids - Ordered list of item IDs
   * @param config - From configuration specifying the source column
   * @param defaultVisibility - Fallback visibility when a value is missing or invalid
   * @param loadTable - Async function that loads the {@link TableData}
   * @param options - Optional abort signal and buffer alignment
   * @returns A `Uint8Array` of packed visibility values
   */
  static async resolveVisibilitiesFromTableValues(
    ids: IDArray,
    config: Extract<VisibilityConfig, FromConfig>,
    defaultVisibility: boolean,
    loadTable: (options?: { signal?: AbortSignal }) => Promise<TableData>,
    options?: { signal?: AbortSignal; align?: number },
  ): Promise<Uint8Array> {
    const { signal, align = 1 } = options ?? {};
    signal?.throwIfAborted();
    const data = await loadTable({ signal });
    const packedVisibilities = VisibilityResolver.createVisibilityBuffer(
      ids.length,
      { align },
    );
    await TableUtils.fillFromTableValues(
      packedVisibilities,
      data,
      ids,
      config.from.column,
      defaultVisibility,
      (value) => VisibilityResolver.parseVisibility(value),
      (visibility) => VisibilityResolver.packVisibility(visibility),
      { signal },
    );
    return packedVisibilities;
  }

  /**
   * Loads visibility data by grouping IDs via a table column and mapping each group
   * to a boolean visibility value using a visibility map.
   *
   * @param ids - Ordered list of item IDs
   * @param config - GroupBy configuration specifying the source column and map
   * @param visibilityMaps - Available visibility maps for group-to-boolean lookups
   * @param defaultVisibility - Fallback visibility when the map is not found or a group is unmapped
   * @param loadTable - Async function that loads the {@link TableData}
   * @param options - Optional abort signal and buffer alignment
   * @returns A `Uint8Array` of packed visibility values
   */
  static async resolveVisibilitiesFromTableGroups(
    ids: IDArray,
    config: Extract<VisibilityConfig, GroupByConfig<true>>,
    visibilityMaps: GroupValueMap<boolean>[],
    defaultVisibility: boolean,
    loadTable: (options?: { signal?: AbortSignal }) => Promise<TableData>,
    options?: { signal?: AbortSignal; align?: number },
  ): Promise<Uint8Array> {
    const { signal, align = 1 } = options ?? {};
    signal?.throwIfAborted();
    const visibilityMap = ConfigUtils.findGroupByMap(config, visibilityMaps);
    if (visibilityMap === undefined) {
      console.warn(
        `Visibility map ${config.groupBy.map} not found, using default visibility`,
      );
      return VisibilityResolver.createUniformVisibilities(
        ids.length,
        defaultVisibility,
        { align },
      );
    }
    const data = await loadTable({ signal });
    const packedVisibilities = VisibilityResolver.createVisibilityBuffer(
      ids.length,
      { align },
    );
    await TableUtils.fillFromTableGroups(
      packedVisibilities,
      data,
      ids,
      config.groupBy.column,
      visibilityMap.default ?? defaultVisibility,
      ConfigUtils.createGroupValueGetter(
        config,
        visibilityMap,
        defaultVisibility,
      ),
      (visibility) => VisibilityResolver.packVisibility(visibility),
      { signal },
    );
    return packedVisibilities;
  }

  /**
   * Creates a visibility data buffer of the given size filled with a single visibility value
   *
   * @param n - Number of elements
   * @param visibility - The boolean visibility to fill with
   * @param options - Optional buffer alignment
   * @returns A `Uint8Array` filled with the packed visibility
   */
  static createUniformVisibilities(
    n: number,
    visibility: boolean,
    options?: { align?: number },
  ): Uint8Array {
    const { align = 1 } = options ?? {};
    const packedVisibilities = VisibilityResolver.createVisibilityBuffer(n, {
      align,
    });
    const packedVisibility = VisibilityResolver.packVisibility(visibility);
    packedVisibilities.fill(packedVisibility, 0, n);
    return packedVisibilities;
  }

  /**
   * Creates a buffer of the given size for storing packed visibility values, aligned to the specified byte boundary
   *
   * @param n - The number of elements in the buffer
   * @param options - Optional buffer alignment
   * @returns A `Uint8Array` of length `n`, aligned to the given byte boundary
   */
  static createVisibilityBuffer(
    n: number,
    options?: { align?: number },
  ): Uint8Array {
    const { align = 1 } = options ?? {};
    const alignedN = MathUtils.align(n, align);
    return new Uint8Array(alignedN);
  }

  /**
   * Parses a raw value as a boolean visibility (truthy if greater than 0)
   *
   * @param value - The raw value to parse (must be a number)
   * @returns `true` if the value is greater than 0, `false` if 0 or negative, or `undefined` if not a number
   */
  static parseVisibility(value: unknown): boolean | undefined {
    if (typeof value === "boolean") {
      return value;
    }
    const visibility = NumberUtils.tryParseFinite(value, {
      requireSafeBigInt: true,
    });
    return visibility !== undefined ? visibility > 0 : undefined;
  }

  /**
   * Packs a boolean visibility into a numeric representation
   *
   * @param visibility - The boolean visibility to pack
   * @returns `1` if visible, `0` if not
   */
  static packVisibility(visibility: boolean): number {
    return ColorUtils.packVisibility(visibility);
  }
}
