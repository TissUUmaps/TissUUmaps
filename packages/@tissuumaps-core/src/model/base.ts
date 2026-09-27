import type { TableColumnRef } from "./configs";
import { identityTransform } from "./constants";
import type { SimilarityTransform } from "./primitives";

/**
 * Default values for {@link RawModel}
 */
export const modelDefaults = {} as const satisfies Partial<RawModel>;

/**
 * Base interface for all model types in a TissUUmaps project
 */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface RawModel {}

/**
 * A {@link RawModel} with {@link modelDefaults} applied
 */
export type Model = object &
  Required<Pick<RawModel, keyof typeof modelDefaults>> &
  Omit<RawModel, keyof typeof modelDefaults>;

/**
 * Creates a {@link Model} from a {@link RawModel} by applying {@link modelDefaults}
 *
 * @param rawModel - The raw model
 * @returns The complete model with default values applied
 */
export function createModel(rawModel: RawModel): Model {
  return { ...structuredClone(modelDefaults), ...structuredClone(rawModel) };
}

/**
 * Default values for {@link RawDataObject}
 */
export const dataObjectDefaults = {} as const satisfies Partial<
  RawDataObject<RawDataSource<string>>
>;

/**
 * A named, identifiable data object backed by a data source
 */
export interface RawDataObject<
  TRawDataSource extends RawDataSource<string>,
> extends RawModel {
  /** Data object ID */
  id: string;

  /** Human-readable data object name */
  name: string;

  /** Data source */
  dataSource: TRawDataSource;
}

/**
 * A {@link RawDataObject} with {@link dataObjectDefaults} applied
 */
export type DataObject<TDataSource extends DataSource<string>> = Model &
  Required<Pick<RawDataObject<TDataSource>, keyof typeof dataObjectDefaults>> &
  Omit<RawDataObject<TDataSource>, keyof typeof dataObjectDefaults>;

/**
 * Creates a {@link DataObject} from a {@link RawDataObject} by applying {@link dataObjectDefaults}
 *
 * @param rawDataObject - The raw data object
 * @returns The complete data object with default values applied
 */
export function createDataObject<
  TType extends string,
  TRawDataSource extends RawDataSource<TType>,
>(rawDataObject: RawDataObject<TRawDataSource>): DataObject<DataSource<TType>> {
  return {
    ...createModel(rawDataObject),
    ...structuredClone(dataObjectDefaults),
    ...structuredClone(rawDataObject),
    dataSource: createDataSource(rawDataObject.dataSource),
  };
}

/**
 * Default values for {@link RawRenderedDataObject}
 */
export const renderedDataObjectDefaults = {
  visibility: true,
  opacity: 1,
  transform: identityTransform,
} as const satisfies Partial<RawRenderedDataObject<RawDataSource<string>>>;

/**
 * A data object that is rendered
 *
 * Which layer(s) a rendered data object is drawn on depends on its kind of
 * data (see {@link RawRenderedRasterDataObject} and
 * {@link RawRenderedItemsDataObject}).
 */
export interface RawRenderedDataObject<
  TRawDataSource extends RawDataSource<string>,
> extends RawDataObject<TRawDataSource> {
  /**
   * Data object visibility
   *
   * @defaultValue {@link renderedDataObjectDefaults.visibility}
   */
  visibility?: boolean;

  /**
   * Data object opacity, in the range [0, 1]
   *
   * @defaultValue {@link renderedDataObjectDefaults.opacity}
   */
  opacity?: number;

  /**
   * Transformation from data object space to layer space
   *
   * @defaultValue {@link renderedDataObjectDefaults.transform}
   */
  transform?: SimilarityTransform;
}

/**
 * A {@link RawRenderedDataObject} with {@link renderedDataObjectDefaults} applied
 */
export type RenderedDataObject<TDataSource extends DataSource<string>> =
  DataObject<TDataSource> &
    Required<
      Pick<
        RawRenderedDataObject<TDataSource>,
        keyof typeof renderedDataObjectDefaults
      >
    > &
    Omit<
      RawRenderedDataObject<TDataSource>,
      keyof typeof renderedDataObjectDefaults
    >;

/**
 * Creates a {@link RenderedDataObject} from a {@link RawRenderedDataObject} by applying {@link renderedDataObjectDefaults}
 *
 * @param rawRenderedDataObject - The raw rendered data object
 * @returns The complete rendered data object with default values applied
 */
export function createRenderedDataObject<
  TType extends string,
  TRawDataSource extends RawDataSource<TType>,
>(
  rawRenderedDataObject: RawRenderedDataObject<TRawDataSource>,
): RenderedDataObject<DataSource<TType>> {
  return {
    ...createDataObject(rawRenderedDataObject),
    ...structuredClone(renderedDataObjectDefaults),
    ...structuredClone(rawRenderedDataObject),
  };
}

/**
 * Default values for {@link RawRenderedRasterDataObject}
 */
export const renderedRasterDataObjectDefaults = {} as const satisfies Partial<
  RawRenderedRasterDataObject<RawDataSource<string>>
>;

/**
 * A rendered data object whose data is a raster (e.g. images, labels)
 *
 * A raster cannot be split between layers, so it is drawn on a single layer.
 */
export interface RawRenderedRasterDataObject<
  TRawDataSource extends RawDataSource<string>,
> extends RawRenderedDataObject<TRawDataSource> {
  /** Layer ID */
  layer: string;
}

/**
 * A {@link RawRenderedRasterDataObject} with {@link renderedRasterDataObjectDefaults} applied
 */
export type RenderedRasterDataObject<TDataSource extends DataSource<string>> =
  RenderedDataObject<TDataSource> &
    Required<
      Pick<
        RawRenderedRasterDataObject<TDataSource>,
        keyof typeof renderedRasterDataObjectDefaults
      >
    > &
    Omit<
      RawRenderedRasterDataObject<TDataSource>,
      keyof typeof renderedRasterDataObjectDefaults
    >;

/**
 * Creates a {@link RenderedRasterDataObject} from a {@link RawRenderedRasterDataObject} by applying {@link renderedRasterDataObjectDefaults}
 *
 * @param rawRenderedRasterDataObject - The raw rendered raster data object
 * @returns The complete rendered raster data object with default values applied
 */
export function createRenderedRasterDataObject<
  TType extends string,
  TRawDataSource extends RawDataSource<TType>,
>(
  rawRenderedRasterDataObject: RawRenderedRasterDataObject<TRawDataSource>,
): RenderedRasterDataObject<DataSource<TType>> {
  return {
    ...createRenderedDataObject(rawRenderedRasterDataObject),
    ...structuredClone(renderedRasterDataObjectDefaults),
    ...structuredClone(rawRenderedRasterDataObject),
  };
}

/**
 * Default values for {@link RawRenderedItemsDataObject}
 */
export const renderedItemsDataObjectDefaults = {} as const satisfies Partial<
  RawRenderedItemsDataObject<RawDataSource<string>>
>;

/**
 * A rendered data object whose data consists of items (e.g. points, shapes)
 *
 * Items can be distributed across layers, so the layer may be given per item.
 * Tables consist of items as well, but are not rendered, and hence are plain
 * {@link RawDataObject}s.
 */
export interface RawRenderedItemsDataObject<
  TRawDataSource extends RawDataSource<string>,
> extends RawRenderedDataObject<TRawDataSource> {
  /**
   * Layer ID
   *
   * Can be specified as:
   * - An ID of an existing Layer
   * - A table column holding the layer ID values for each item, of the
   *   object's own table unless it names another
   */
  layer: string | TableColumnRef;
}

/**
 * A {@link RawRenderedItemsDataObject} with {@link renderedItemsDataObjectDefaults} applied
 */
export type RenderedItemsDataObject<TDataSource extends DataSource<string>> =
  RenderedDataObject<TDataSource> &
    Required<
      Pick<
        RawRenderedItemsDataObject<TDataSource>,
        keyof typeof renderedItemsDataObjectDefaults
      >
    > &
    Omit<
      RawRenderedItemsDataObject<TDataSource>,
      keyof typeof renderedItemsDataObjectDefaults
    >;

/**
 * Creates a {@link RenderedItemsDataObject} from a {@link RawRenderedItemsDataObject} by applying {@link renderedItemsDataObjectDefaults}
 *
 * @param rawRenderedItemsDataObject - The raw rendered items data object
 * @returns The complete rendered items data object with default values applied
 */
export function createRenderedItemsDataObject<
  TType extends string,
  TRawDataSource extends RawDataSource<TType>,
>(
  rawRenderedItemsDataObject: RawRenderedItemsDataObject<TRawDataSource>,
): RenderedItemsDataObject<DataSource<TType>> {
  return {
    ...createRenderedDataObject(rawRenderedItemsDataObject),
    ...structuredClone(renderedItemsDataObjectDefaults),
    ...structuredClone(rawRenderedItemsDataObject),
  };
}

/**
 * Default values for {@link RawDataSource}
 */
export const dataSourceDefaults = {} as const satisfies Partial<
  RawDataSource<string>
>;

/**
 * A data source for data objects
 */
export interface RawDataSource<TType extends string = string> extends RawModel {
  /**
   * Data source type
   */
  type: TType;

  /**
   * Where the data is loaded from: a URL, an app-relative path (`//` prefix),
   * a workspace-relative path (`/` prefix), or a project-relative path (no
   * prefix); see `SourceUtils` for the syntax and how paths are resolved
   */
  source?: string;
}

/**
 * A {@link RawDataSource} with {@link dataSourceDefaults} applied
 */
export type DataSource<TType extends string = string> = Model &
  Required<Pick<RawDataSource<TType>, keyof typeof dataSourceDefaults>> &
  Omit<RawDataSource<TType>, keyof typeof dataSourceDefaults>;

/**
 * Creates a {@link DataSource} from a {@link RawDataSource} by applying {@link dataSourceDefaults}
 *
 * @param rawDataSource - The raw data source
 * @returns The complete data source with default values applied
 */
export function createDataSource<TType extends string>(
  rawDataSource: RawDataSource<TType>,
): DataSource<TType> {
  return {
    ...createModel(rawDataSource),
    ...structuredClone(dataSourceDefaults),
    ...structuredClone(rawDataSource),
  };
}

/**
 * Default values for {@link RawAnnotatedDataSource}
 */
export const annotatedDataSourceDefaults = {} as const satisfies Partial<
  RawAnnotatedDataSource<string>
>;

/**
 * A data source for rendered data objects whose items can be annotated by a
 * referenced table (e.g. labels, points, shapes)
 */
export interface RawAnnotatedDataSource<
  TType extends string = string,
> extends RawDataSource<TType> {
  /** ID of the table holding the per-item annotations of this data source, if any */
  table?: string;
}

/**
 * A {@link RawAnnotatedDataSource} with {@link annotatedDataSourceDefaults} applied
 */
export type AnnotatedDataSource<TType extends string = string> =
  DataSource<TType> &
    Required<
      Pick<
        RawAnnotatedDataSource<TType>,
        keyof typeof annotatedDataSourceDefaults
      >
    > &
    Omit<
      RawAnnotatedDataSource<TType>,
      keyof typeof annotatedDataSourceDefaults
    >;

/**
 * Creates an {@link AnnotatedDataSource} from a {@link RawAnnotatedDataSource} by applying {@link annotatedDataSourceDefaults}
 *
 * @param rawAnnotatedDataSource - The raw annotated data source
 * @returns The complete annotated data source with default values applied
 */
export function createAnnotatedDataSource<TType extends string>(
  rawAnnotatedDataSource: RawAnnotatedDataSource<TType>,
): AnnotatedDataSource<TType> {
  return {
    ...createDataSource(rawAnnotatedDataSource),
    ...structuredClone(annotatedDataSourceDefaults),
    ...structuredClone(rawAnnotatedDataSource),
  };
}
