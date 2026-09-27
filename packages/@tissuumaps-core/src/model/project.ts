import type { OpenSeadragonOptions } from "../types/openseadragon";
import type { WebGLOptions } from "../types/webgl";
import { type Model, type RawModel, createModel } from "./base";
import { type Image, type RawImage, createImage } from "./image";
import { type Labels, type RawLabels, createLabels } from "./labels";
import { type Layer, type RawLayer, createLayer } from "./layer";
import { type Points, type RawPoints, createPoints } from "./points";
import type { Color, GroupValueMap, Marker } from "./primitives";
import { type RawShapes, type Shapes, createShapes } from "./shapes";
import { type RawTable, type Table, createTable } from "./table";

/**
 * Default values for {@link RawProject}
 */
export const projectDefaults = {
  markerMaps: [],
  sizeMaps: [],
  colorMaps: [],
  visibilityMaps: [],
  opacityMaps: [],
  osOptions: {
    viewerOptions: {
      minZoomImageRatio: 0,
      maxZoomPixelRatio: Infinity,
      preserveImageSizeOnResize: true,
      visibilityRatio: 0,
      animationTime: 0,
      gestureSettingsMouse: {
        flickEnabled: false,
      },
      gestureSettingsTouch: {
        flickEnabled: false,
      },
      gestureSettingsPen: {
        flickEnabled: false,
      },
      gestureSettingsUnknown: {
        flickEnabled: false,
      },
      zoomPerClick: 1,
      showNavigator: true,
      navigatorPosition: "BOTTOM_LEFT",
      maxImageCacheCount: 2000,
      showNavigationControl: false,
      imageSmoothingEnabled: false,
    },
    viewerAnimationStartOptions: {
      immediateRender: false,
      imageLoaderLimit: 1,
    },
    viewerAnimationFinishOptions: {
      immediateRender: true, // set to true, even if initially set to false
    },
  },
  glOptions: {
    pointsRenderOptions: {
      globalPointSizeFactor: 1,
    },
    shapesRenderOptions: {
      strokeWidth: 1,
      edgesPerScanline: 8,
      binWidthFactor: 1,
      shapePadding: 0.2,
    },
  },
} as const satisfies Partial<RawProject>;

/**
 * A TissUUmaps project
 *
 * Top-level container that assembles layers, data objects (images, labels,
 * points, shapes), tables, group-to-value maps, and viewer/render options
 * into a single serializable configuration.
 */
export interface RawProject extends RawModel {
  /**
   * Project name
   */
  name: string;

  /**
   * Layers
   */
  layers?: RawLayer[];

  /**
   * Images
   */
  images?: RawImage[];

  /**
   * Labels
   */
  labels?: RawLabels[];

  /**
   * Points
   */
  points?: RawPoints[];

  /**
   * Shapes
   */
  shapes?: RawShapes[];

  /**
   * Tables
   */
  tables?: RawTable[];

  /**
   * Project-global marker maps, referenced by {@link GroupByConfig} marker configurations
   *
   * Map IDs are unique within this list; the other lists of maps may reuse them.
   *
   * @defaultValue {@link projectDefaults.markerMaps}
   */
  markerMaps?: GroupValueMap<Marker>[];

  /**
   * Project-global size maps, referenced by {@link GroupByConfig} size configurations
   *
   * Map IDs are unique within this list; the other lists of maps may reuse them.
   *
   * @defaultValue {@link projectDefaults.sizeMaps}
   */
  sizeMaps?: GroupValueMap<number>[];

  /**
   * Project-global color maps, referenced by {@link GroupByConfig} color configurations
   *
   * Map IDs are unique within this list; the other lists of maps may reuse them.
   *
   * @defaultValue {@link projectDefaults.colorMaps}
   */
  colorMaps?: GroupValueMap<Color>[];

  /**
   * Project-global visibility maps, referenced by {@link GroupByConfig} visibility configurations
   *
   * Map IDs are unique within this list; the other lists of maps may reuse them.
   *
   * @defaultValue {@link projectDefaults.visibilityMaps}
   */
  visibilityMaps?: GroupValueMap<boolean>[];

  /**
   * Project-global opacity maps, referenced by {@link GroupByConfig} opacity configurations
   *
   * Map IDs are unique within this list; the other lists of maps may reuse them.
   *
   * @defaultValue {@link projectDefaults.opacityMaps}
   */
  opacityMaps?: GroupValueMap<number>[];

  /**
   * OpenSeadragon viewer options for images/labels
   *
   * @defaultValue {@link projectDefaults.osOptions}
   * @see https://openseadragon.github.io/docs/OpenSeadragon.html#.Options
   */
  osOptions?: OpenSeadragonOptions;

  /**
   * WebGL render options for points/shapes
   *
   * @defaultValue {@link projectDefaults.glOptions}
   */
  glOptions?: WebGLOptions;
}

/**
 * A {@link RawProject} with {@link projectDefaults} applied
 */
export type Project = Model &
  Required<Pick<RawProject, keyof typeof projectDefaults>> &
  Omit<
    RawProject,
    | keyof typeof projectDefaults
    | ("layers" | "images" | "labels" | "points" | "shapes" | "tables")
  > & {
    layers: Layer[];
    images: Image[];
    labels: Labels[];
    points: Points[];
    shapes: Shapes[];
    tables: Table[];
  };

/**
 * Creates a {@link Project} from a {@link RawProject} by applying {@link projectDefaults}
 *
 * @param rawProject - The raw project
 * @returns The complete project with default values applied
 */
export function createProject(rawProject: RawProject): Project {
  return {
    ...createModel(rawProject),
    ...structuredClone(projectDefaults),
    ...structuredClone(rawProject),
    osOptions: {
      viewerOptions: {
        ...projectDefaults.osOptions.viewerOptions,
        ...rawProject.osOptions?.viewerOptions,
      },
      viewerAnimationStartOptions: {
        ...projectDefaults.osOptions.viewerAnimationStartOptions,
        ...rawProject.osOptions?.viewerAnimationStartOptions,
      },
      viewerAnimationFinishOptions: {
        ...projectDefaults.osOptions.viewerAnimationFinishOptions,
        ...rawProject.osOptions?.viewerAnimationFinishOptions,
      },
    },
    glOptions: {
      pointsRenderOptions: {
        ...projectDefaults.glOptions.pointsRenderOptions,
        ...rawProject.glOptions?.pointsRenderOptions,
      },
      shapesRenderOptions: {
        ...projectDefaults.glOptions.shapesRenderOptions,
        ...rawProject.glOptions?.shapesRenderOptions,
      },
    },
    layers: rawProject.layers?.map(createLayer) ?? [],
    images: rawProject.images?.map(createImage) ?? [],
    labels: rawProject.labels?.map(createLabels) ?? [],
    points: rawProject.points?.map(createPoints) ?? [],
    shapes: rawProject.shapes?.map(createShapes) ?? [],
    tables: rawProject.tables?.map(createTable) ?? [],
  };
}
