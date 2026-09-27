import type { Color, CoordinateSpace, Marker } from "./primitives";

/**
 * Base type for property configurations that can be sourced from different providers
 *
 * Concrete configuration types (e.g. {@link ColorConfig}, {@link SizeConfig}) are unions
 * of this base with one or more source-specific types such as {@link ConstantConfig},
 * {@link FromConfig}, {@link GroupByConfig}, and {@link RandomConfig}.
 */
export type Config<TSource extends string> = {
  /**
   * Explicitly prioritized configuration source
   *
   * When set, this overrides the automatic source detection in {@link getActiveConfigSource}.
   */
  source?: TSource;
};

/**
 * Determines the active source of the given {@link Config}
 *
 * If a source is explicitly prioritized using {@link Config.source}, that source is returned.
 *
 * Otherwise, the active source is determined by checking for the presence of configuration-specific fields in the following order:
 * - {@link ConstantConfig}
 * - {@link FromConfig}
 * - {@link GroupByConfig}
 * - {@link RandomConfig}
 *
 * @param config - The configuration
 * @returns The active configuration source, or `undefined` if none is active
 */
export function getActiveConfigSource<TSource extends string>(
  config: Config<TSource>,
): TSource | undefined {
  if (config.source !== undefined) {
    return config.source;
  }
  if (isConstantConfig(config)) {
    return "constant" as TSource;
  }
  if (isFromConfig(config)) {
    return "from" as TSource;
  }
  if (isGroupByConfig(config)) {
    return "groupBy" as TSource;
  }
  if (isRandomConfig(config)) {
    return "random" as TSource;
  }
  return undefined;
}

/** Configuration to use a constant value */
export type ConstantConfig<
  TValue,
  TConstantExtra = unknown,
> = Config<"constant"> & {
  /** Specification of a constant value */
  constant: { value: NonNullable<TValue> } & TConstantExtra;
};

/**
 * Determines whether the given object is a {@link ConstantConfig}
 *
 * @param obj - The object to check
 * @returns Whether the object is a {@link ConstantConfig}
 */
export function isConstantConfig<TValue, TConstantExtra = unknown>(
  obj: unknown,
): obj is ConstantConfig<TValue, TConstantExtra> {
  return (obj as ConstantConfig<TValue, TConstantExtra>).constant !== undefined;
}

/** Configuration to load values from a table column */
export type FromConfig<TFromExtra = unknown> = Config<"from"> & {
  /** Specification of what table column to load */
  from: {
    /** Name of the table column */
    column: string;
  } & TFromExtra;
};

/**
 * Determines whether the given object is a {@link FromConfig}
 *
 * @param obj - The object to check
 * @returns Whether the object is a {@link FromConfig}
 */
export function isFromConfig<TFromExtra = unknown>(
  obj: unknown,
): obj is FromConfig<TFromExtra> {
  return (obj as FromConfig<TFromExtra>).from !== undefined;
}

/** Configuration to map a categorical table column to values */
export type GroupByConfig<
  TMapRequired extends boolean,
  TGroupByExtra = unknown,
> = Config<"groupBy"> & {
  /** Specification of what categorical table column to load and how to map groups to values */
  groupBy: {
    /** Name of the categorical table column */
    column: string;

    /** Project-global group-to-value map ID */
    map: TMapRequired extends true ? string : string | undefined;
  } & TGroupByExtra;
};

/**
 * Determines whether the given object is a {@link GroupByConfig}
 *
 * @param obj - The object to check
 * @returns Whether the object is a {@link GroupByConfig}
 */
export function isGroupByConfig<
  TMapRequired extends boolean,
  TGroupByExtra = unknown,
>(obj: unknown): obj is GroupByConfig<TMapRequired, TGroupByExtra> {
  return (
    (obj as GroupByConfig<TMapRequired, TGroupByExtra>).groupBy !== undefined
  );
}

/** Configuration to use random values */
export type RandomConfig<TRandomExtra = unknown> = Config<"random"> & {
  /** Specification of random value generation */
  random: {
    /**
     * The seed for the random number generator
     *
     * Values are drawn deterministically from the item ID and the seed, so
     * different seeds yield different assignments for the same items.
     *
     * @defaultValue {@link defaultRandomSeed}
     */
    seed?: number;
  } & TRandomExtra;
};

/**
 * Determines whether the given object is a {@link RandomConfig}
 *
 * @param obj - The object to check
 * @returns Whether the object is a {@link RandomConfig}
 */
export function isRandomConfig<TRandomExtra = unknown>(
  obj: unknown,
): obj is RandomConfig<TRandomExtra> {
  return (obj as RandomConfig<TRandomExtra>).random !== undefined;
}

/**
 * Marker configuration
 *
 * When sourced from a table column, numerical values are interpreted as
 * {@link Marker} indices (e.g. `0` = Cross, `6` = Disc).
 */
export type MarkerConfig =
  ConstantConfig<Marker> | FromConfig | GroupByConfig<false>;

/**
 * Size configuration
 *
 * When sourced from a table column, numerical values are interpreted as sizes
 * in the specified {@link CoordinateSpace} unit.
 */
export type SizeConfig =
  | ConstantConfig<
      number,
      {
        /** Coordinate space in which the size values are specified */
        unit?: CoordinateSpace;
      }
    >
  | FromConfig<{
      /** Coordinate space in which the size values are specified */
      unit?: CoordinateSpace;
    }>
  | GroupByConfig<
      true,
      {
        /** Coordinate space in which the size values are specified */
        unit?: CoordinateSpace;
      }
    >;

/**
 * Color configuration
 *
 * When sourced from a numerical table column, values are clipped to the
 * configured range - or min-max-scaled, if none is given - and then mapped to
 * the configured color palette.
 */
export type ColorConfig =
  | ConstantConfig<Color>
  | FromConfig<{
      /**
       * Value range that is linearly mapped to the color palette
       *
       * Values are clipped to this range before mapping them to colors.
       *
       * If not specified, min-max-scaling is used.
       */
      range?: [number, number];

      /** ID of the color palette to which clipped and rescaled numerical values are mapped */
      palette: string;
    }>
  | GroupByConfig<
      false,
      {
        /**
         * ID of the color palette for mapping hashed group names to colors
         *
         * Only used when no project-global colormap is specified
         */
        palette?: string;
      }
    >
  | RandomConfig<{
      /** ID of the color palette from which colors are randomly drawn */
      palette: string;
    }>;

/**
 * Visibility configuration
 *
 * Numerical table values are interpreted as booleans, where `0` is `false` and any other value is `true`.
 */
export type VisibilityConfig =
  ConstantConfig<boolean> | FromConfig | GroupByConfig<true>;

/**
 * Opacity configuration
 *
 * Numerical table values are interpreted as opacities between `0` (fully transparent) and `1` (fully opaque).
 */
export type OpacityConfig =
  ConstantConfig<number> | FromConfig | GroupByConfig<true>;
