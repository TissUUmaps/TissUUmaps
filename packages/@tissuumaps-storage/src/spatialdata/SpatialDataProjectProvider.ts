// only this subpath: the fork ships TypeScript sources, and those of the other
// stores fail our type-checking
import FileSystemHandleStore from "@zarrita/storage/fs-handle";
import * as zarr from "zarrita";

import {
  type ProjectProvider,
  type RawImage,
  type RawLabels,
  type RawLayer,
  type RawPoints,
  type RawProject,
  type RawShapes,
  type RawTable,
  type SimilarityTransform,
  type SizeConfig,
  SourceUtils,
} from "@tissuumaps/core";

import { ZarrStore } from "../hierarchical/zarr/ZarrStore";
import {
  type ZarrTableDataSource,
  zarrTableDataSourceType,
} from "../hierarchical/zarr/ZarrTableDataSource";
import { omeZarrImageDataSourceType } from "../ome-zarr/OMEZarrImageDataSource";
import { omeZarrLabelsDataSourceType } from "../ome-zarr/OMEZarrLabelsDataSource";
import { ParquetShapesDataProvider } from "../parquet/ParquetShapesDataProvider";
import { parquetShapesDataSourceType } from "../parquet/ParquetShapesDataSource";
import { parquetTableDataSourceType } from "../parquet/ParquetTableDataSource";
import {
  type TablePointsDataSource,
  tablePointsDataSourceType,
} from "../table/TablePointsDataSource";
import {
  type CoordinateTransformation,
  SpatialDataTransformUtils,
} from "./SpatialDataTransformUtils";

/** The project type of SpatialData stores */
export const spatialDataProjectType = "spatialdata";

/**
 * Loads SpatialData stores as projects
 *
 * A SpatialData store is a Zarr store with consolidated metadata, whose root
 * group lists the elements of the dataset by kind. Its elements become data
 * objects of the existing types:
 *
 * - images and labels become OME-Zarr images and labels
 * - shapes become Parquet shapes if they are polygons, and point clouds of a
 *   Parquet table if they are circles, sized by their radius
 * - tables (AnnData) become Zarr tables, keyed by their instance key; a table
 *   annotating a single element is linked to it, unless the element is circles
 *
 * Each coordinate system becomes a layer, and an element becomes one object
 * per coordinate system it maps into, placed by its transformation into it.
 * Only one layer is visible: `global`, or else the first one holding the most
 * objects.
 *
 * @see https://spatialdata-spec.scverse.org/
 */
export class SpatialDataProjectProvider implements ProjectProvider {
  /** The default coordinate system of SpatialData */
  private static readonly _defaultCoordinateSystem = "global";

  /** Reads the GeoParquet files of shapes, to tell polygons from circles */
  private static readonly _parquetShapesDataProvider =
    new ParquetShapesDataProvider();

  /** The factor from a circle's radius to its point size, a point's diameter */
  private static readonly _circlePointSizeFactor = 2;

  /** The geometry column of shapes, whose points are the centers of circles */
  private static readonly _circleGeometryColumn = "geometry";

  /** The column holding the radius of each circle */
  private static readonly _circleRadiusColumn = "radius";

  /**
   * The attributes that mark the group of an element, which spatialdata also
   * gives SpatialData attributes: images and labels have OME-Zarr attributes,
   * shapes and points an encoding type, and tables a SpatialData encoding type
   */
  private static readonly _elementAttributes = [
    "ome",
    "multiscales",
    "encoding-type",
    "spatialdata-encoding-type",
  ];

  readonly name = "SpatialData";
  readonly urlParam = "spatialdata";

  /**
   * Returns whether a source is a SpatialData store, from the attributes of
   * its root group: SpatialData attributes, but none of those spatialdata
   * gives the groups of its elements
   *
   * @param normalizedSource - The normalized source, which may be the
   * workspace root `/`
   * @param workspace - The directory handle of the workspace, if any
   * @param options - Optional abort signal
   * @returns A promise that resolves to whether the source is a SpatialData
   * store, `false` if it is no Zarr group
   */
  async supports(
    normalizedSource: string,
    workspace: FileSystemDirectoryHandle | null,
    options?: { signal?: AbortSignal },
  ): Promise<boolean> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const store = await SpatialDataProjectProvider._openStore(
      normalizedSource,
      workspace,
      { signal },
    );
    let attributes: Record<string, unknown>;
    try {
      attributes = (await zarr.open(store, { kind: "group", signal })).attrs;
    } catch (error) {
      if (zarr.isZarritaError(error, "NotFoundError")) {
        return false;
      }
      throw error;
    }
    return (
      "spatialdata_attrs" in attributes &&
      !SpatialDataProjectProvider._elementAttributes.some(
        (name) => name in attributes,
      )
    );
  }

  async load(
    normalizedSource: string,
    options?: {
      signal?: AbortSignal;
      workspace?: FileSystemDirectoryHandle | null;
    },
  ): Promise<{ project: RawProject; source: string }> {
    const { signal, workspace = null } = options ?? {};
    signal?.throwIfAborted();
    // relative sources resolve against the store like a file in its parent
    // directory, so a trailing slash would add the store's name twice
    const source =
      normalizedSource === "/"
        ? normalizedSource
        : normalizedSource.replace(/\/+$/, "");
    const store = await ZarrStore.open(
      await SpatialDataProjectProvider._openStore(source, workspace, {
        signal,
      }),
      "",
      { signal },
    );
    const { tables, tableIDsByElement } =
      await SpatialDataProjectProvider._readTables(store, source, { signal });
    const images = await SpatialDataProjectProvider._readImages(store, source, {
      signal,
    });
    const labels = await SpatialDataProjectProvider._readLabels(
      store,
      source,
      tableIDsByElement,
      { signal },
    );
    const { shapes, circleTables, circles } =
      await SpatialDataProjectProvider._readShapes(
        store,
        source,
        workspace,
        tableIDsByElement,
        { signal },
      );
    const layers = SpatialDataProjectProvider._createLayers(
      [...images, ...labels, ...shapes, ...circles].map(
        ({ coordinateSystem }) => coordinateSystem,
      ),
    );
    const layerIDs = new Map(layers.map(({ id, name }) => [name, id]));
    const project: RawProject = {
      name:
        source === "/" ? (workspace?.name ?? "") : SourceUtils.getStem(source),
      layers,
      images: images.map(({ coordinateSystem, object }) => ({
        ...object,
        layer: layerIDs.get(coordinateSystem)!,
      })),
      labels: labels.map(({ coordinateSystem, object }) => ({
        ...object,
        layer: layerIDs.get(coordinateSystem)!,
      })),
      shapes: shapes.map(({ coordinateSystem, object }) => ({
        ...object,
        layer: layerIDs.get(coordinateSystem)!,
      })),
      points: circles.map(({ coordinateSystem, object }) => ({
        ...object,
        layer: layerIDs.get(coordinateSystem)!,
      })),
      tables: [...tables, ...circleTables],
    };
    return { project, source };
  }

  /** Opens the Zarr store at a source, which may be the workspace root */
  private static async _openStore(
    normalizedSource: string,
    workspace: FileSystemDirectoryHandle | null,
    options?: { signal?: AbortSignal },
  ): Promise<zarr.AsyncReadable> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    let directory: string | FileSystemDirectoryHandle;
    if (normalizedSource === "/") {
      if (workspace === null) {
        throw new Error("Cannot open the workspace root without workspace.");
      }
      directory = workspace;
    } else {
      directory = await SourceUtils.resolveSourceDirectory(
        normalizedSource,
        workspace,
        { signal },
      );
    }
    return typeof directory === "string"
      ? new zarr.FetchStore(directory)
      : new FileSystemHandleStore(directory);
  }

  /**
   * Reads the tables of a store, keyed by their instance key
   *
   * @param store - The store
   * @param source - The source of the store, without trailing slash
   * @param options - Optional abort signal
   * @returns The tables, and the ID of the table annotating each element that
   * a table annotates alone: instance keys repeat across the elements of a
   * table annotating several
   */
  private static async _readTables(
    store: ZarrStore,
    source: string,
    options?: { signal?: AbortSignal },
  ): Promise<{ tables: RawTable[]; tableIDsByElement: Map<string, string> }> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const groups = await SpatialDataProjectProvider._readChildGroups(
      store,
      "tables",
      { signal },
    );
    const tables: RawTable[] = [];
    const tableIDsByElement = new Map<string, string>();
    for (const { name, attrs } of groups) {
      const { region, instance_key: instanceKey } = attrs as {
        region: string | string[] | null;
        instance_key: string;
      };
      const tableSource = SpatialDataProjectProvider._getElementSource(
        source,
        `tables/${name}`,
      );
      const dataSource: ZarrTableDataSource = {
        type: zarrTableDataSourceType,
        source: tableSource,
        idColumn: `obs/${instanceKey}`,
      };
      const id = SpatialDataProjectProvider._createID(source, `tables/${name}`);
      tables.push({ id, name, dataSource });
      const regions = typeof region === "string" ? [region] : (region ?? []);
      if (regions.length === 1) {
        tableIDsByElement.set(regions[0]!, id);
      }
    }
    return { tables, tableIDsByElement };
  }

  /**
   * Reads the images of a store as OME-Zarr images, one per coordinate
   * system an image is placed in
   *
   * @param store - The store
   * @param source - The source of the store, without trailing slash
   * @param options - Optional abort signal
   * @returns The images, each with the name of its coordinate system
   */
  private static async _readImages(
    store: ZarrStore,
    source: string,
    options?: { signal?: AbortSignal },
  ): Promise<{ coordinateSystem: string; object: Omit<RawImage, "layer"> }[]> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const groups = await SpatialDataProjectProvider._readChildGroups(
      store,
      "images",
      { signal },
    );
    return groups.flatMap(({ name, attrs }) => {
      const path = `images/${name}`;
      const imageSource = SpatialDataProjectProvider._getElementSource(
        source,
        path,
      );
      return SpatialDataProjectProvider._readRasterTransforms(path, attrs).map(
        ([coordinateSystem, transform]) => ({
          coordinateSystem,
          object: {
            id: SpatialDataProjectProvider._createID(source, path),
            name,
            transform,
            dataSource: {
              type: omeZarrImageDataSourceType,
              source: imageSource,
            },
          },
        }),
      );
    });
  }

  /**
   * Reads the labels of a store as OME-Zarr labels, one per coordinate system
   * they are placed in
   *
   * @param store - The store
   * @param source - The source of the store, without trailing slash
   * @param tableIDsByElement - The ID of the table annotating an element alone
   * @param options - Optional abort signal
   * @returns The labels, each with the name of its coordinate system
   */
  private static async _readLabels(
    store: ZarrStore,
    source: string,
    tableIDsByElement: Map<string, string>,
    options?: { signal?: AbortSignal },
  ): Promise<{ coordinateSystem: string; object: Omit<RawLabels, "layer"> }[]> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const groups = await SpatialDataProjectProvider._readChildGroups(
      store,
      "labels",
      { signal },
    );
    return groups.flatMap(({ name, attrs }) => {
      const path = `labels/${name}`;
      const labelsSource = SpatialDataProjectProvider._getElementSource(
        source,
        path,
      );
      return SpatialDataProjectProvider._readRasterTransforms(path, attrs).map(
        ([coordinateSystem, transform]) => ({
          coordinateSystem,
          object: {
            id: SpatialDataProjectProvider._createID(source, path),
            name,
            transform,
            dataSource: {
              type: omeZarrLabelsDataSourceType,
              source: labelsSource,
              table: tableIDsByElement.get(name),
            },
          },
        }),
      );
    });
  }

  /**
   * Reads the shapes of a store: polygons as Parquet shapes, and circles as
   * point clouds of a Parquet table, sized by their radius
   *
   * Polygons and circles are told apart from the geometry types of the
   * GeoParquet file. A table annotating circles is not linked to them, as a
   * point cloud's table holds its coordinates.
   *
   * @param store - The store
   * @param source - The source of the store, without trailing slash
   * @param workspace - The directory handle of the workspace, if any
   * @param tableIDsByElement - The ID of the table annotating an element alone
   * @param options - Optional abort signal
   * @returns The polygon shapes and the circle point clouds, each with the
   * name of its coordinate system, and a table per circles element
   */
  private static async _readShapes(
    store: ZarrStore,
    source: string,
    workspace: FileSystemDirectoryHandle | null,
    tableIDsByElement: Map<string, string>,
    options?: { signal?: AbortSignal },
  ): Promise<{
    shapes: { coordinateSystem: string; object: Omit<RawShapes, "layer"> }[];
    circleTables: RawTable[];
    circles: { coordinateSystem: string; object: Omit<RawPoints, "layer"> }[];
  }> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const groups = (
      await SpatialDataProjectProvider._readChildGroups(store, "shapes", {
        signal,
      })
    ).map(({ name, attrs }) => {
      const path = `shapes/${name}`;
      return {
        name,
        path,
        parquetSource: `${SpatialDataProjectProvider._getElementSource(source, path)}/shapes.parquet`,
        transforms: SpatialDataProjectProvider._readVectorTransforms(
          path,
          attrs,
        ),
      };
    });
    const isPolygons = await Promise.all(
      groups.map(({ parquetSource }) =>
        SpatialDataProjectProvider._parquetShapesDataProvider.supports(
          SourceUtils.normalizeSource(parquetSource, workspace, source),
          workspace,
          { signal },
        ),
      ),
    );
    const polygonGroups = groups.filter((_, i) => isPolygons[i]);
    const circleGroups = groups.filter((_, i) => !isPolygons[i]);

    const shapes = polygonGroups.flatMap(
      ({ name, path, parquetSource, transforms }) =>
        transforms.map(([coordinateSystem, transform]) => ({
          coordinateSystem,
          object: {
            id: SpatialDataProjectProvider._createID(source, path),
            name,
            transform,
            dataSource: {
              type: parquetShapesDataSourceType,
              source: parquetSource,
              table: tableIDsByElement.get(name),
            },
          },
        })),
    );
    const circleTables: RawTable[] = circleGroups.map(
      ({ name, path, parquetSource }) => ({
        id: SpatialDataProjectProvider._createID(source, path),
        name,
        dataSource: { type: parquetTableDataSourceType, source: parquetSource },
      }),
    );
    const circles = circleGroups.flatMap(({ name, path, transforms }, i) => {
      const geometryColumn = SpatialDataProjectProvider._circleGeometryColumn;
      const dataSource: TablePointsDataSource = {
        type: tablePointsDataSourceType,
        table: circleTables[i]!.id,
        x: `${geometryColumn}[x]`,
        y: `${geometryColumn}[y]`,
      };
      const pointSize: SizeConfig = {
        from: {
          column: SpatialDataProjectProvider._circleRadiusColumn,
          unit: "data",
        },
      };
      return transforms.map(([coordinateSystem, transform]) => ({
        coordinateSystem,
        object: {
          id: SpatialDataProjectProvider._createID(source, path),
          name,
          transform,
          dataSource,
          pointSize,
          pointSizeFactor: SpatialDataProjectProvider._circlePointSizeFactor,
        },
      }));
    });
    return { shapes, circleTables, circles };
  }

  /**
   * Creates a layer per coordinate system, of which only one is visible:
   * `global`, or else the first one most objects are placed in
   *
   * @param coordinateSystems - The coordinate system of every object
   * @returns The layers, in the order their coordinate systems first appear
   */
  private static _createLayers(coordinateSystems: string[]): RawLayer[] {
    const counts = new Map<string, number>();
    for (const coordinateSystem of coordinateSystems) {
      counts.set(coordinateSystem, (counts.get(coordinateSystem) ?? 0) + 1);
    }
    const maxCount = Math.max(...counts.values());
    const visibleCoordinateSystem = counts.has(
      SpatialDataProjectProvider._defaultCoordinateSystem,
    )
      ? SpatialDataProjectProvider._defaultCoordinateSystem
      : [...counts.keys()].find(
          (coordinateSystem) => counts.get(coordinateSystem) === maxCount,
        );
    return [...counts.keys()].map((coordinateSystem) => ({
      id: SourceUtils.createID(undefined),
      name: coordinateSystem,
      visibility: coordinateSystem === visibleCoordinateSystem,
    }));
  }

  /**
   * Reads the transform by which an image or labels element is placed in each
   * coordinate system it maps into, from its multiscales
   *
   * @param path - The path of the element, for warnings
   * @param attrs - The attributes of the element's group
   * @returns The transform by coordinate system name, see
   * {@link SpatialDataProjectProvider._getTransforms}
   */
  private static _readRasterTransforms(
    path: string,
    attrs: Readonly<Record<string, unknown>>,
  ): [string, SimilarityTransform][] {
    // OME-Zarr 0.5 nests its attributes under `ome`, as spatialdata writes them
    const { multiscales } = (attrs.ome ?? attrs) as {
      multiscales: {
        axes: { name: string }[];
        coordinateTransformations: CoordinateTransformation[];
      }[];
    };
    const { axes, coordinateTransformations } = multiscales[0]!;
    return SpatialDataProjectProvider._getTransforms(
      path,
      coordinateTransformations,
      axes.map(({ name }) => name),
    );
  }

  /**
   * Reads the transform by which a shapes or points element is placed in each
   * coordinate system it maps into, from the attributes of its group
   *
   * @param path - The path of the element, for warnings
   * @param attrs - The attributes of the element's group
   * @returns The transform by coordinate system name, see
   * {@link SpatialDataProjectProvider._getTransforms}
   */
  private static _readVectorTransforms(
    path: string,
    attrs: Readonly<Record<string, unknown>>,
  ): [string, SimilarityTransform][] {
    const { axes, coordinateTransformations } = attrs as {
      axes: string[];
      coordinateTransformations: CoordinateTransformation[];
    };
    return SpatialDataProjectProvider._getTransforms(
      path,
      coordinateTransformations,
      axes,
    );
  }

  /**
   * Returns the transform by which an element is placed in each coordinate
   * system it maps into, warning about approximated ones
   *
   * @param path - The path of the element, for the warning
   * @param transformations - The coordinate transformations of the element
   * @param axes - The names of the element's axes, in order
   * @returns The transform by coordinate system name, in the order of the
   * element's transformations
   */
  private static _getTransforms(
    path: string,
    transformations: CoordinateTransformation[],
    axes: string[],
  ): [string, SimilarityTransform][] {
    const transforms = SpatialDataTransformUtils.getSimilarityTransforms(
      transformations,
      axes,
    );
    return [...transforms].map(([coordinateSystem, { transform, exact }]) => {
      if (!exact) {
        console.warn(
          `The transformation of ${path} into "${coordinateSystem}" is approximated, as it scales the axes differently or shears them.`,
        );
      }
      return [coordinateSystem, transform];
    });
  }

  /**
   * Reads the child groups of a group, e.g. the images of a store
   *
   * @param store - The store
   * @param path - The path of the group, which may be missing
   * @param options - Optional abort signal
   * @returns The name and attributes of each child group, in store order
   */
  private static async _readChildGroups(
    store: ZarrStore,
    path: string,
    options?: { signal?: AbortSignal },
  ): Promise<{ name: string; attrs: Readonly<Record<string, unknown>> }[]> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    const group = await store.get(path, { signal });
    const names = group?.kind === "group" ? group.keys : [];
    return Promise.all(
      names.map(async (name) => {
        const child = await store.get(`${path}/${name}`, { signal });
        return { name, attrs: child?.kind === "group" ? child.attrs : {} };
      }),
    );
  }

  /**
   * Returns the source of an element relative to the store's source, i.e. to
   * its parent directory, or to the workspace root if the store is the
   * workspace
   *
   * @param source - The source of the store, without trailing slash
   * @param path - The path of the element within the store
   * @returns The relative source (e.g. `data.zarr/images/a`)
   */
  private static _getElementSource(source: string, path: string): string {
    return source === "/"
      ? path
      : `${SourceUtils.getPathSegments(source).at(-1)!}/${path}`;
  }

  /**
   * Creates the ID of an object, named after its element
   *
   * An element placed in several coordinate systems becomes several objects,
   * each with an ID of its own.
   *
   * @param source - The source of the store, without trailing slash
   * @param path - The path of the element within the store
   * @returns The ID
   */
  private static _createID(source: string, path: string): string {
    return SourceUtils.createID(`${source === "/" ? "" : source}/${path}`);
  }
}
