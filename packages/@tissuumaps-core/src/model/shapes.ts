import {
  type AnnotatedDataSource,
  type RawAnnotatedDataSource,
  type RawRenderedItemsDataObject,
  type RenderedItemsDataObject,
  createAnnotatedDataSource,
  createRenderedItemsDataObject,
} from "./base";
import type { ColorConfig, OpacityConfig, VisibilityConfig } from "./configs";
import {
  defaultShapeFillColor,
  defaultShapeFillOpacity,
  defaultShapeFillVisibility,
  defaultShapeOpacity,
  defaultShapeStrokeColor,
  defaultShapeStrokeOpacity,
  defaultShapeStrokeVisibility,
  defaultShapeVisibility,
} from "./constants";

/**
 * Default values for {@link RawShapes}
 */
export const shapesDefaults = {
  shapeVisibility: { constant: { value: defaultShapeVisibility } },
  shapeOpacity: { constant: { value: defaultShapeOpacity } },
  shapeFillColor: { constant: { value: defaultShapeFillColor } },
  shapeFillVisibility: { constant: { value: defaultShapeFillVisibility } },
  shapeFillOpacity: { constant: { value: defaultShapeFillOpacity } },
  shapeStrokeColor: { constant: { value: defaultShapeStrokeColor } },
  shapeStrokeVisibility: { constant: { value: defaultShapeStrokeVisibility } },
  shapeStrokeOpacity: { constant: { value: defaultShapeStrokeOpacity } },
} as const satisfies Partial<RawShapes>;

/**
 * A two-dimensional shape cloud
 */
export interface RawShapes extends RawRenderedItemsDataObject<
  RawShapesDataSource<string>
> {
  /**
   * Shape visibility
   *
   * Applies to both the fill and the stroke of a shape, on top of
   * {@link shapeFillVisibility} and {@link shapeStrokeVisibility}.
   *
   * @defaultValue {@link shapesDefaults.shapeVisibility}
   */
  shapeVisibility?: VisibilityConfig;

  /**
   * Shape opacity
   *
   * Multiplies {@link shapeFillOpacity} and {@link shapeStrokeOpacity}.
   *
   * @defaultValue {@link shapesDefaults.shapeOpacity}
   */
  shapeOpacity?: OpacityConfig;

  /**
   * Shape fill color
   *
   * @defaultValue {@link shapesDefaults.shapeFillColor}
   */
  shapeFillColor?: ColorConfig;

  /**
   * Shape fill visibility
   *
   * @defaultValue {@link shapesDefaults.shapeFillVisibility}
   */
  shapeFillVisibility?: VisibilityConfig;

  /**
   * Shape fill opacity
   *
   * @defaultValue {@link shapesDefaults.shapeFillOpacity}
   */
  shapeFillOpacity?: OpacityConfig;

  /**
   * Shape stroke color
   *
   * @defaultValue {@link shapesDefaults.shapeStrokeColor}
   */
  shapeStrokeColor?: ColorConfig;

  /**
   * Shape stroke visibility
   *
   * @defaultValue {@link shapesDefaults.shapeStrokeVisibility}
   */
  shapeStrokeVisibility?: VisibilityConfig;

  /**
   * Shape stroke opacity
   *
   * @defaultValue {@link shapesDefaults.shapeStrokeOpacity}
   */
  shapeStrokeOpacity?: OpacityConfig;
}

/**
 * A {@link RawShapes} object with {@link shapesDefaults} applied
 */
export type Shapes = RenderedItemsDataObject<ShapesDataSource<string>> &
  Required<Pick<RawShapes, keyof typeof shapesDefaults>> &
  Omit<RawShapes, keyof typeof shapesDefaults>;

/**
 * Creates a {@link Shapes} from a {@link RawShapes} by applying {@link shapesDefaults}
 *
 * @param rawShapes - The raw shapes
 * @returns The complete shapes with default values applied
 */
export function createShapes(rawShapes: RawShapes): Shapes {
  return {
    ...createRenderedItemsDataObject(rawShapes),
    ...structuredClone(shapesDefaults),
    ...structuredClone(rawShapes),
    dataSource: createShapesDataSource(rawShapes.dataSource),
  };
}

/**
 * Default values for {@link RawShapesDataSource}
 */
export const shapesDataSourceDefaults = {} as const satisfies Partial<
  RawShapesDataSource<string>
>;

/**
 * A data source for two-dimensional shape clouds
 */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface RawShapesDataSource<
  TType extends string = string,
> extends RawAnnotatedDataSource<TType> {}

/**
 * A {@link RawShapesDataSource} with {@link shapesDataSourceDefaults} applied
 */
export type ShapesDataSource<TType extends string = string> =
  AnnotatedDataSource<TType> &
    Required<
      Pick<RawShapesDataSource<TType>, keyof typeof shapesDataSourceDefaults>
    > &
    Omit<RawShapesDataSource<TType>, keyof typeof shapesDataSourceDefaults>;

/**
 * Creates a {@link ShapesDataSource} from a {@link RawShapesDataSource} by applying {@link shapesDataSourceDefaults}
 *
 * @param rawShapesDataSource - The raw shapes data source
 * @returns The complete shapes data source with default values applied
 */
export function createShapesDataSource<TType extends string>(
  rawShapesDataSource: RawShapesDataSource<TType>,
): ShapesDataSource<TType> {
  return {
    ...createAnnotatedDataSource(rawShapesDataSource),
    ...structuredClone(shapesDataSourceDefaults),
    ...structuredClone(rawShapesDataSource),
  };
}
