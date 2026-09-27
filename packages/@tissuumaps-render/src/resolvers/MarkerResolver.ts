import {
  ConfigUtils,
  type ConstantConfig,
  type FromConfig,
  type GroupByConfig,
  type GroupValueMap,
  type IDArray,
  type Marker,
  type MarkerConfig,
  MathUtils,
  NumberUtils,
  type TableData,
  TableUtils,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
  markerPalette,
} from "@tissuumaps/core";

/**
 * Resolves the marker of every item, packed as a {@link Marker} index
 */
export class MarkerResolver {
  /**
   * Loads marker data for a set of IDs based on the active marker configuration source
   *
   * Dispatches to the appropriate loader (constant, from, or groupBy) depending on which
   * configuration source is active.
   *
   * @param ids - Ordered list of item IDs
   * @param config - Marker configuration specifying the data source
   * @param markerMaps - Available marker maps for groupBy lookups
   * @param defaultMarker - Fallback marker when no valid config or value is found
   * @param options - Optional abort signal, buffer alignment, and table loader
   * @returns A `Uint8Array` of packed marker values, one per ID
   */
  static async resolveMarkers(
    ids: IDArray,
    config: MarkerConfig,
    markerMaps: GroupValueMap<Marker>[],
    defaultMarker: Marker,
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
      return MarkerResolver.resolveUniformMarkers(ids, config, { align });
    }
    if (
      activeConfigSource === "from" &&
      isFromConfig(config) &&
      loadTable !== undefined
    ) {
      return MarkerResolver.resolveMarkersFromTableValues(
        ids,
        config,
        defaultMarker,
        loadTable,
        { signal, align },
      );
    }
    if (
      activeConfigSource === "groupBy" &&
      isGroupByConfig(config) &&
      loadTable !== undefined
    ) {
      return MarkerResolver.resolveMarkersFromTableGroups(
        ids,
        config,
        markerMaps,
        defaultMarker,
        loadTable,
        { signal, align },
      );
    }
    console.warn("No valid marker config found, using default marker");
    return MarkerResolver.createUniformMarkers(ids.length, defaultMarker, {
      align,
    });
  }

  /**
   * Resolves the marker all items share if the configuration is a constant
   *
   * The counterpart of {@link resolveMarkers} for a constant source, which needs
   * no items: the one packed marker applies to every item, so a consumer can
   * supply it once instead of once per item.
   *
   * @param config - Marker configuration specifying the data source
   * @returns The packed marker, or `undefined` if the active source is not a
   * constant
   */
  static resolveConstantMarker(config: MarkerConfig): number | undefined {
    const activeConfigSource = getActiveConfigSource(config);
    if (activeConfigSource === "constant" && isConstantConfig(config)) {
      return MarkerResolver.packMarker(config.constant.value);
    }
    return undefined;
  }

  /**
   * Resolves the marker of a single item without loading any table data
   *
   * Synchronous counterpart to {@link resolveMarkers} for items that are not
   * known up front, in the way the labels renderer resolves label IDs as they
   * are first drawn (see `ColorResolver.resolveColorWithoutTable`). Labels have
   * no marker, so this has no caller yet and exists for symmetry with the
   * other resolvers. A constant source resolves exactly, whereas from and
   * groupBy sources depend on table values and fall back to `defaultMarker`.
   *
   * @param _id - The item ID (unused, kept for symmetry with the other resolvers)
   * @param config - Marker configuration specifying the data source
   * @param defaultMarker - Fallback marker when the source cannot be resolved without a table
   * @returns The packed marker index
   */
  static resolveMarkerWithoutTable(
    _id: number | string,
    config: MarkerConfig,
    defaultMarker: Marker,
  ): number {
    return (
      MarkerResolver.resolveConstantMarker(config) ??
      MarkerResolver.packMarker(defaultMarker)
    );
  }

  /**
   * Creates a uniform marker data buffer filled with the configured constant marker
   *
   * @param ids - Ordered list of item IDs (only the length is used)
   * @param config - Constant marker configuration containing the marker value
   * @param options - Optional buffer alignment
   * @returns A `Uint8Array` filled with the packed constant marker
   */
  static resolveUniformMarkers(
    ids: IDArray,
    config: Extract<MarkerConfig, ConstantConfig<Marker>>,
    options?: { align?: number },
  ): Uint8Array {
    const { align = 1 } = options ?? {};
    return MarkerResolver.createUniformMarkers(
      ids.length,
      config.constant.value,
      { align },
    );
  }

  /**
   * Loads marker data by reading values from a table column and parsing them as markers
   *
   * @param ids - Ordered list of item IDs
   * @param config - From configuration specifying the source column
   * @param defaultMarker - Fallback marker when a value is missing or invalid
   * @param loadTable - Async function that loads the {@link TableData}
   * @param options - Optional abort signal and buffer alignment
   * @returns A `Uint8Array` of packed marker values
   */
  static async resolveMarkersFromTableValues(
    ids: IDArray,
    config: Extract<MarkerConfig, FromConfig>,
    defaultMarker: Marker,
    loadTable: (options?: { signal?: AbortSignal }) => Promise<TableData>,
    options?: { signal?: AbortSignal; align?: number },
  ): Promise<Uint8Array> {
    const { signal, align = 1 } = options ?? {};
    signal?.throwIfAborted();
    const data = await loadTable({ signal });
    const packedMarkers = MarkerResolver.createMarkerBuffer(ids.length, {
      align,
    });
    await TableUtils.fillFromTableValues(
      packedMarkers,
      data,
      ids,
      config.from.column,
      defaultMarker,
      (value) => MarkerResolver.parseMarker(value),
      (marker) => MarkerResolver.packMarker(marker),
      { signal },
    );
    return packedMarkers;
  }

  /**
   * Loads marker data by grouping IDs via a table column and mapping each group
   * to a marker using either a marker map or the default marker palette.
   *
   * @param ids - Ordered list of item IDs
   * @param config - GroupBy configuration specifying the source column and optional map
   * @param markerMaps - Available marker maps for group-to-marker lookups
   * @param defaultMarker - Fallback marker when the map is not found or a group is unmapped
   * @param loadTable - Async function that loads the {@link TableData}
   * @param options - Optional abort signal and buffer alignment
   * @returns A `Uint8Array` of packed marker values
   */
  static async resolveMarkersFromTableGroups(
    ids: IDArray,
    config: Extract<MarkerConfig, GroupByConfig<false>>,
    markerMaps: GroupValueMap<Marker>[],
    defaultMarker: Marker,
    loadTable: (options?: { signal?: AbortSignal }) => Promise<TableData>,
    options?: { signal?: AbortSignal; align?: number },
  ) {
    const { signal, align = 1 } = options ?? {};
    signal?.throwIfAborted();
    const markerMap = ConfigUtils.findGroupByMap(config, markerMaps);
    if (config.groupBy.map !== undefined && markerMap === undefined) {
      console.warn(
        `Marker map ${config.groupBy.map} not found, using default marker`,
      );
      return MarkerResolver.createUniformMarkers(ids.length, defaultMarker, {
        align,
      });
    }
    const data = await loadTable({ signal });
    const packedMarkers = MarkerResolver.createMarkerBuffer(ids.length, {
      align,
    });
    await TableUtils.fillFromTableGroups(
      packedMarkers,
      data,
      ids,
      config.groupBy.column,
      markerMap?.default ?? defaultMarker,
      ConfigUtils.createGroupValueGetter(
        config,
        markerMap,
        defaultMarker,
        markerPalette,
      ),
      (marker) => MarkerResolver.packMarker(marker),
      { signal },
    );
    return packedMarkers;
  }

  /**
   * Creates a marker data buffer of the given size filled with a single marker
   *
   * @param n - Number of elements
   * @param marker - The marker to fill with
   * @param options - Optional buffer alignment
   * @returns A `Uint8Array` filled with the packed marker
   */
  static createUniformMarkers(
    n: number,
    marker: Marker,
    options?: { align?: number },
  ): Uint8Array {
    const { align = 1 } = options ?? {};
    const packedMarkers = MarkerResolver.createMarkerBuffer(n, { align });
    const packedMarker = MarkerResolver.packMarker(marker);
    packedMarkers.fill(packedMarker, 0, n);
    return packedMarkers;
  }

  /**
   * Creates a buffer of the given size for storing packed marker values, aligned to the specified byte boundary
   *
   * @param n - The number of elements in the buffer
   * @param options - Optional buffer alignment
   * @returns A `Uint8Array` of length `n`, aligned to the given byte boundary
   */
  static createMarkerBuffer(
    n: number,
    options?: { align?: number },
  ): Uint8Array {
    const { align = 1 } = options ?? {};
    const alignedN = MathUtils.align(n, align);
    return new Uint8Array(alignedN);
  }

  /**
   * Parses a raw value as a {@link Marker}
   *
   * @param value - The raw value to parse (must be a number)
   * @returns The value cast to a {@link Marker}, or `undefined` if not a number
   */
  static parseMarker(value: unknown): Marker | undefined {
    return NumberUtils.tryParseSafeInt(value) as Marker | undefined;
  }

  /**
   * Packs a {@link Marker} into its numeric representation
   *
   * @param marker - The marker to pack
   * @returns The marker index as a number
   */
  static packMarker(marker: Marker): number {
    return marker;
  }
}
