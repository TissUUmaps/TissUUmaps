import {
  AsyncUtils,
  type Color,
  type ColorConfig,
  type ColorPalette,
  ColorUtils,
  ConfigUtils,
  type ConstantConfig,
  type FromConfig,
  type GroupByConfig,
  type GroupValueMap,
  HashUtils,
  type IDArray,
  MathUtils,
  NumberUtils,
  type RandomConfig,
  type TableData,
  TableUtils,
  defaultRandomSeed,
  findColorPalette,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
  isRandomConfig,
} from "@tissuumaps/core";

/**
 * Resolves the color of every item, as a packed RGB value
 *
 * The alpha channel is left to the caller, which resolves visibilities and
 * opacities separately and folds them in afterwards.
 */
export class ColorResolver {
  /**
   * Loads color data for a set of IDs based on the active color configuration source
   *
   * Dispatches to the appropriate loader (constant, from, groupBy, or random).
   * The returned colors carry no alpha; the caller folds the separately
   * resolved visibilities and opacities into it.
   *
   * @param ids - Ordered list of item IDs
   * @param config - Color configuration specifying the data source
   * @param colorMaps - Available color maps for groupBy lookups
   * @param defaultColor - Fallback color when no valid config or value is found
   * @param options - Optional abort signal, buffer alignment, and table loader
   * @returns A `Uint32Array` of packed RGB color values, one per ID
   */
  static async resolveColors(
    ids: IDArray,
    config: ColorConfig,
    colorMaps: GroupValueMap<Color>[],
    defaultColor: Color,
    options?: {
      signal?: AbortSignal;
      align?: number;
      loadTable?: (options?: { signal?: AbortSignal }) => Promise<TableData>;
    },
  ): Promise<Uint32Array> {
    const { signal, align = 1, loadTable } = options ?? {};
    signal?.throwIfAborted();
    let packedColors: Uint32Array;
    const activeConfigSource = getActiveConfigSource(config);
    if (activeConfigSource === "constant" && isConstantConfig(config)) {
      packedColors = ColorResolver.resolveUniformColors(ids, config, { align });
    } else if (
      activeConfigSource === "from" &&
      isFromConfig(config) &&
      loadTable !== undefined
    ) {
      packedColors = await ColorResolver.resolveColorsFromTableValues(
        ids,
        config,
        defaultColor,
        loadTable,
        { signal, align },
      );
    } else if (
      activeConfigSource === "groupBy" &&
      isGroupByConfig(config) &&
      loadTable !== undefined
    ) {
      packedColors = await ColorResolver.resolveColorsFromTableGroups(
        ids,
        config,
        colorMaps,
        defaultColor,
        loadTable,
        { signal, align },
      );
    } else if (activeConfigSource === "random" && isRandomConfig(config)) {
      packedColors = await ColorResolver.resolveRandomColors(
        ids,
        config,
        defaultColor,
        { signal, align },
      );
    } else {
      console.warn("No valid color config found, using default color");
      packedColors = ColorResolver.createUniformColors(
        ids.length,
        defaultColor,
        {
          align,
        },
      );
    }
    return packedColors;
  }

  /**
   * Resolves the color all items share if the configuration is a constant
   *
   * The counterpart of {@link resolveColors} for a constant source, which needs
   * no items: the one packed color applies to every item, so a consumer can
   * supply it once instead of once per item.
   *
   * @param config - Color configuration specifying the data source
   * @returns The packed color, or `undefined` if the active source is not a
   * constant
   */
  static resolveConstantColor(config: ColorConfig): number | undefined {
    const activeConfigSource = getActiveConfigSource(config);
    if (activeConfigSource === "constant" && isConstantConfig(config)) {
      return ColorResolver.packColor(config.constant.value);
    }
    return undefined;
  }

  /**
   * Resolves the color of a single item without loading any table data
   *
   * Synchronous counterpart to {@link resolveColors} for items that are not
   * known up front: the labels renderer uses it to color label IDs as they are
   * first drawn, since a label image does not enumerate its labels. Constant
   * and random sources resolve exactly, whereas from and groupBy sources
   * depend on table values and fall back to `defaultColor`.
   *
   * @param id - The item ID
   * @param config - Color configuration specifying the data source
   * @param defaultColor - Fallback color when the source cannot be resolved without a table
   * @returns The packed RGB color value, without alpha
   */
  static resolveColorWithoutTable(
    id: number | string,
    config: ColorConfig,
    defaultColor: Color,
  ): number {
    const constantColor = ColorResolver.resolveConstantColor(config);
    if (constantColor !== undefined) {
      return constantColor;
    }
    if (getActiveConfigSource(config) === "random" && isRandomConfig(config)) {
      const colorPalette = findColorPalette(config.random.palette);
      if (colorPalette !== undefined) {
        const color = ColorResolver.pickRandomColor(
          id,
          config.random.seed ?? defaultRandomSeed,
          colorPalette,
        );
        return ColorResolver.packColor(color);
      }
    }
    return ColorResolver.packColor(defaultColor);
  }

  /**
   * Creates a uniform color data buffer filled with the configured constant color
   *
   * @param ids - Ordered list of item IDs (only the length is used)
   * @param config - Constant color configuration containing the color value
   * @param options - Optional buffer alignment
   * @returns A `Uint32Array` filled with the packed constant color, without alpha
   */
  static resolveUniformColors(
    ids: IDArray,
    config: Extract<ColorConfig, ConstantConfig<Color>>,
    options?: { align?: number },
  ): Uint32Array {
    const { align = 1 } = options ?? {};
    return ColorResolver.createUniformColors(
      ids.length,
      config.constant.value,
      { align },
    );
  }

  /**
   * Loads color data by reading numeric values from a table column and mapping them
   * through a color palette.
   *
   * @param ids - Ordered list of item IDs
   * @param config - From configuration specifying the source column, palette, and range
   * @param defaultColor - Fallback color when the palette is not found or a value is invalid
   * @param loadTable - Async function that loads the {@link TableData}
   * @param options - Optional abort signal and buffer alignment
   * @returns A `Uint32Array` of packed color values
   */
  static async resolveColorsFromTableValues(
    ids: IDArray,
    config: Extract<ColorConfig, FromConfig>,
    defaultColor: Color,
    loadTable: (options?: { signal?: AbortSignal }) => Promise<TableData>,
    options?: { signal?: AbortSignal; align?: number },
  ): Promise<Uint32Array> {
    const { signal, align = 1 } = options ?? {};
    signal?.throwIfAborted();
    const colorPalette = findColorPalette(config.from.palette);
    if (colorPalette === undefined) {
      console.warn(
        `Color palette ${config.from.palette} not found, using default color`,
      );
      return ColorResolver.createUniformColors(ids.length, defaultColor, {
        align,
      });
    }
    const data = await loadTable({ signal });
    const valueRange =
      config.from.range ??
      (await data.loadValueRange(config.from.column, { signal }));
    const packedColors = ColorResolver.createColorBuffer(ids.length, { align });
    await TableUtils.fillFromTableValues(
      packedColors,
      data,
      ids,
      config.from.column,
      defaultColor,
      (value) => ColorResolver.parseColor(value, valueRange, colorPalette),
      (color) => ColorResolver.packColor(color),
      { signal },
    );
    return packedColors;
  }

  /**
   * Loads color data by grouping IDs via a table column and mapping each group
   * to a color using either a color map or a color palette.
   *
   * @param ids - Ordered list of item IDs
   * @param config - GroupBy configuration specifying the source column and map/palette
   * @param colorMaps - Available color maps for group-to-color lookups
   * @param defaultColor - Fallback color when the map/palette is not found or a group is unmapped
   * @param loadTable - Async function that loads the {@link TableData}
   * @param options - Optional abort signal and buffer alignment
   * @returns A `Uint32Array` of packed color values
   */
  static async resolveColorsFromTableGroups(
    ids: IDArray,
    config: Extract<ColorConfig, GroupByConfig<false>>,
    colorMaps: GroupValueMap<Color>[],
    defaultColor: Color,
    loadTable: (options?: { signal?: AbortSignal }) => Promise<TableData>,
    options?: { signal?: AbortSignal; align?: number },
  ): Promise<Uint32Array> {
    const { signal, align = 1 } = options ?? {};
    signal?.throwIfAborted();
    const colorMap = ConfigUtils.findGroupByMap(config, colorMaps);
    const colorPalette = findColorPalette(config.groupBy.palette);
    if (config.groupBy.map !== undefined && colorMap === undefined) {
      console.warn(
        `Color map ${config.groupBy.map} not found, using default color`,
      );
      return ColorResolver.createUniformColors(ids.length, defaultColor, {
        align,
      });
    }
    if (config.groupBy.map === undefined && colorPalette === undefined) {
      console.warn(
        config.groupBy.palette !== undefined
          ? `Color palette ${config.groupBy.palette} not found, using default color`
          : `No color map or color palette specified, using default color`,
      );
      return ColorResolver.createUniformColors(ids.length, defaultColor, {
        align,
      });
    }
    const data = await loadTable({ signal });
    const packedColors = ColorResolver.createColorBuffer(ids.length, {
      align,
    });
    await TableUtils.fillFromTableGroups(
      packedColors,
      data,
      ids,
      config.groupBy.column,
      colorMap?.default ?? defaultColor,
      ConfigUtils.createGroupValueGetter(
        config,
        colorMap,
        defaultColor,
        colorPalette?.colors,
      ),
      (color) => ColorResolver.packColor(color),
      { signal },
    );
    return packedColors;
  }

  /**
   * Loads color data by assigning each ID a random color from the configured palette
   *
   * The colors are drawn per ID (see {@link pickRandomColor}) and packed
   * without alpha, which {@link resolveColors} adds.
   *
   * @param ids - Ordered list of item IDs
   * @param config - Random configuration specifying the palette to sample from
   * @param defaultColor - Fallback color when the palette is not found
   * @param options - Optional abort signal and buffer alignment
   * @returns A `Uint32Array` of packed random color values
   */
  static async resolveRandomColors(
    ids: IDArray,
    config: Extract<ColorConfig, RandomConfig<unknown>>,
    defaultColor: Color,
    options?: { signal?: AbortSignal; align?: number },
  ): Promise<Uint32Array> {
    const { signal, align = 1 } = options ?? {};
    signal?.throwIfAborted();
    const colorPalette = findColorPalette(config.random.palette);
    if (colorPalette === undefined) {
      console.warn(
        `Color palette ${config.random.palette} not found, using default color`,
      );
      return ColorResolver.createUniformColors(ids.length, defaultColor, {
        align,
      });
    }
    const packedColors = ColorResolver.createColorBuffer(ids.length, { align });
    await AsyncUtils.forEach<number | string>(
      ids,
      (id, i) => {
        const color = ColorResolver.pickRandomColor(
          id,
          config.random.seed ?? defaultRandomSeed,
          colorPalette,
        );
        packedColors[i] = ColorResolver.packColor(color);
      },
      { signal },
    );
    return packedColors;
  }

  /**
   * Deterministically picks a random color for an item from a palette
   *
   * The pick is a seeded hash of the ID (see {@link HashUtils.mix} for
   * integer IDs and {@link HashUtils.hash} for string IDs), so it is stable
   * across resolutions and scatters consecutive IDs over the palette.
   *
   * @param id - The item ID
   * @param seed - The seed of the random configuration
   * @param colorPalette - The palette to pick from (must not be empty)
   * @returns The picked {@link Color}
   */
  static pickRandomColor(
    id: number | string,
    seed: number,
    colorPalette: ColorPalette,
  ): Color {
    const hash =
      typeof id === "string"
        ? HashUtils.hash(id, seed)
        : HashUtils.mix(id, seed);
    return colorPalette.colors[hash % colorPalette.colors.length]!;
  }

  /**
   * Creates a color data buffer of the given size filled with a single color
   *
   * @param n - Number of elements
   * @param color - The color to fill with
   * @param options - Optional buffer alignment
   * @returns A `Uint32Array` filled with the packed color
   */
  static createUniformColors(
    n: number,
    color: Color,
    options?: { align?: number },
  ): Uint32Array {
    const { align = 1 } = options ?? {};
    const packedColors = ColorResolver.createColorBuffer(n, { align });
    const packedColor = ColorResolver.packColor(color);
    packedColors.fill(packedColor, 0, n);
    return packedColors;
  }

  /**
   * Creates a buffer of the given size for storing packed color values, aligned to the specified byte boundary
   *
   * @param n - The number of elements in the buffer
   * @param options - Optional buffer alignment
   * @returns A `Uint32Array` of length `n`, aligned to the given byte boundary
   */
  static createColorBuffer(
    n: number,
    options?: { align?: number },
  ): Uint32Array {
    const { align = 1 } = options ?? {};
    const alignedN = MathUtils.align(n, align);
    return new Uint32Array(alignedN);
  }

  /**
   * Parses a raw numeric value into a {@link Color} by normalizing it within the
   * given range and sampling a color palette.
   *
   * The normalized value is clamped to `[0, 1]` and spread over the palette, so
   * that `0` yields its first color and `1` its last. Values falling between two
   * colors are interpolated linearly, which keeps the palette continuous instead
   * of quantizing it to its number of colors.
   *
   * @param value - The raw value to parse (must be a finite number)
   * @param valueRange - The value range `[min, max]` to normalize within,
   * `[0, 1]` if `undefined`
   * @param colorPalette - The palette to sample
   * @returns The corresponding {@link Color}, or `undefined` if `value` is not a
   * finite number, or if the palette is empty
   */
  static parseColor(
    value: unknown,
    valueRange: [number, number] | undefined,
    colorPalette: ColorPalette,
  ): Color | undefined {
    const v = NumberUtils.tryParseFinite(value, { requireSafeBigInt: true });
    if (v === undefined) {
      return undefined;
    }
    const { colors } = colorPalette;
    const [vmin, vmax] = valueRange ?? [0, 1];
    const vnorm = vmax > vmin ? (v - vmin) / (vmax - vmin) : 0;
    const position = MathUtils.clamp(vnorm, 0, 1) * (colors.length - 1);
    const index = Math.floor(position);
    const color = colors[index];
    const nextColor = colors[index + 1];
    if (color === undefined) {
      return undefined;
    }
    if (nextColor === undefined) {
      return color;
    }
    const t = position - index;
    return {
      r: color.r + t * (nextColor.r - color.r),
      g: color.g + t * (nextColor.g - color.g),
      b: color.b + t * (nextColor.b - color.b),
    };
  }

  /**
   * Packs a {@link Color} into a packed numeric representation
   *
   * @param color - The color to pack
   * @returns The color packed into the lower 24 bits, without alpha
   */
  static packColor(color: Color): number {
    return ColorUtils.packColor(color);
  }
}
