import {
  type AnnotatedDataSource,
  type RawAnnotatedDataSource,
  type RawRenderedItemsDataObject,
  type RenderedItemsDataObject,
  createAnnotatedDataSource,
  createRenderedItemsDataObject,
  omitUndefined,
} from "./base";
import type {
  ColorConfig,
  MarkerConfig,
  OpacityConfig,
  OrderConfig,
  SizeConfig,
  VisibilityConfig,
} from "./configs";
import {
  defaultPointColor,
  defaultPointMarker,
  defaultPointOpacity,
  defaultPointSize,
  defaultPointVisibility,
} from "./constants";

/**
 * Default values for {@link RawPoints}
 */
export const pointsDefaults = {
  pointMarker: { constant: { value: defaultPointMarker } },
  pointSize: { constant: { value: defaultPointSize } },
  pointColor: { constant: { value: defaultPointColor } },
  pointVisibility: { constant: { value: defaultPointVisibility } },
  pointOpacity: { constant: { value: defaultPointOpacity } },
  pointSizeFactor: 1,
} as const satisfies Partial<RawPoints>;

/**
 * A two-dimensional point cloud
 */
export interface RawPoints extends RawRenderedItemsDataObject<
  RawPointsDataSource<string>
> {
  /**
   * Point marker
   *
   * @defaultValue {@link pointsDefaults.pointMarker}
   */
  pointMarker?: MarkerConfig;

  /**
   * Point size
   *
   * @defaultValue {@link pointsDefaults.pointSize}
   */
  pointSize?: SizeConfig;

  /**
   * Point color
   *
   * @defaultValue {@link pointsDefaults.pointColor}
   */
  pointColor?: ColorConfig;

  /**
   * Point visibility
   *
   * @defaultValue {@link pointsDefaults.pointVisibility}
   */
  pointVisibility?: VisibilityConfig;

  /**
   * Point opacity
   *
   * @defaultValue {@link pointsDefaults.pointOpacity}
   */
  pointOpacity?: OpacityConfig;

  /**
   * Point draw order
   *
   * If not specified, points are drawn in item order.
   */
  pointOrder?: OrderConfig;

  /**
   * Object-level point size scaling factor
   *
   * A unitless scaling factor by which the size of every point of this object is
   * multiplied, for adjusting point sizes without touching the individual point
   * sizes or their unit.
   *
   * Point sizes are also affected by {@link RawLayer.pointSizeFactor} and by the
   * project-global {@link WebGLPointsRenderOptions.globalPointSizeFactor}.
   *
   * @defaultValue {@link pointsDefaults.pointSizeFactor}
   */
  pointSizeFactor?: number;
}

/**
 * A {@link RawPoints} object with {@link pointsDefaults} applied
 */
export type Points = RenderedItemsDataObject<PointsDataSource<string>> &
  Required<Pick<RawPoints, keyof typeof pointsDefaults>> &
  Omit<RawPoints, keyof typeof pointsDefaults>;

/**
 * Creates a {@link Points} from a {@link RawPoints} by applying {@link pointsDefaults}
 *
 * @param rawPoints - The raw points
 * @returns The complete points with the default values of missing or `undefined` properties
 */
export function createPoints(rawPoints: RawPoints): Points {
  return {
    ...createRenderedItemsDataObject(rawPoints),
    ...structuredClone(pointsDefaults),
    ...structuredClone(omitUndefined(rawPoints)),
    dataSource: createPointsDataSource(rawPoints.dataSource),
  };
}

/**
 * Default values for {@link RawPointsDataSource}
 */
export const pointsDataSourceDefaults = {} as const satisfies Partial<
  RawPointsDataSource<string>
>;

/**
 * A data source for two-dimensional point clouds
 */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface RawPointsDataSource<
  TType extends string = string,
> extends RawAnnotatedDataSource<TType> {}

/**
 * A {@link RawPointsDataSource} with {@link pointsDataSourceDefaults} applied
 */
export type PointsDataSource<TType extends string = string> =
  AnnotatedDataSource<TType> &
    Required<
      Pick<RawPointsDataSource<TType>, keyof typeof pointsDataSourceDefaults>
    > &
    Omit<RawPointsDataSource<TType>, keyof typeof pointsDataSourceDefaults>;

/**
 * Creates a {@link PointsDataSource} from a {@link RawPointsDataSource} by applying {@link pointsDataSourceDefaults}
 *
 * @param rawPointsDataSource - The raw points data source
 * @returns The complete points data source with the default values of missing or `undefined` properties
 */
export function createPointsDataSource<TType extends string>(
  rawPointsDataSource: RawPointsDataSource<TType>,
): PointsDataSource<TType> {
  return {
    ...createAnnotatedDataSource(rawPointsDataSource),
    ...structuredClone(pointsDataSourceDefaults),
    ...structuredClone(omitUndefined(rawPointsDataSource)),
  };
}
