import type { Mutate, StoreApi } from "zustand";

import type { TableColumnRef } from "../../model/configs";
import type { ImageDataSource } from "../../model/image";
import type { LabelsDataSource } from "../../model/labels";
import type { PointsDataSource } from "../../model/points";
import type { ShapesDataSource } from "../../model/shapes";
import type { TableDataSource } from "../../model/table";
import type { ImageData, ImageDataProvider } from "../../storage/image";
import type { LabelsData, LabelsDataProvider } from "../../storage/labels";
import type { PointsData, PointsDataProvider } from "../../storage/points";
import type { ShapesData, ShapesDataProvider } from "../../storage/shapes";
import type { TableData, TableDataProvider } from "../../storage/table";
import type { InteractionMode } from "../interaction";

/** A single image channel shown on its own, without touching the project */
export type ImageChannelPreview = {
  /** The ID of the previewed image */
  imageId: string;

  /** The index of the previewed channel (0-based) */
  channelIndex: number;
};

/** The items of one group of a labels, points or shapes object, highlighted in the viewer */
export type HighlightedItemGroup = {
  /**
   * The object whose items are grouped, by the ID of its labels, points or
   * shapes; IDs are only unique within each of these lists
   */
  annotatedObject:
    { labelsId: string } | { pointsId: string } | { shapesId: string };

  /** The categorical table column */
  groupBy: TableColumnRef;

  /** The group, i.e. the cell value as a string */
  group: string;
};

/**
 * The state of the app store, holding what is not part of the project
 */
export type AppStoreState = {
  /** The directory handle of the open workspace, if any */
  workspace: FileSystemDirectoryHandle | null;

  /**
   * Whether a project is open, possibly an empty one; the Project tab shows
   * its welcome view otherwise
   */
  projectOpen: boolean;

  /** How mouse events in the viewer are currently interpreted */
  interactionMode: InteractionMode;

  /** The channel previewed on its own while it is hovered, if any */
  imageChannelPreview: ImageChannelPreview | null;

  /**
   * The group shown alone in the viewer, hiding every other item of its
   * object, or `null` for none
   */
  highlightedItemGroup: HighlightedItemGroup | null;

  /** The registered image data providers, by data source type */
  imageDataProviders: Map<
    string,
    ImageDataProvider<ImageDataSource, ImageData>
  >;

  /** The registered labels data providers, by data source type */
  labelsDataProviders: Map<
    string,
    LabelsDataProvider<LabelsDataSource, LabelsData>
  >;

  /** The registered points data providers, by data source type */
  pointsDataProviders: Map<
    string,
    PointsDataProvider<PointsDataSource, PointsData>
  >;

  /** The registered shapes data providers, by data source type */
  shapesDataProviders: Map<
    string,
    ShapesDataProvider<ShapesDataSource, ShapesData>
  >;

  /** The registered table data providers, by data source type */
  tableDataProviders: Map<
    string,
    TableDataProvider<TableDataSource, TableData>
  >;

  /**
   * The registered plugins, by plugin ID, each as its human-readable name,
   * whether it has a user interface that can be mounted, and the element its
   * user interface is mounted into while it is mounted
   *
   * Written by the plugin registry, which owns the plugin lifecycle and keeps
   * the plugin objects themselves to itself, so that nothing a plugin owns ends
   * up frozen in the store.
   */
  plugins: Map<
    string,
    { name: string; mountable: boolean; container?: HTMLElement }
  >;

  /**
   * The latest request to show the settings of an image, or `null` if there has
   * been none
   *
   * Set by {@link AppStoreActions.showImageSettings} to a new object on every call, so
   * that a repeated request for the same image is a change too.
   */
  imageSettingsRequest: { imageId: string } | null;

  /**
   * The latest request to show the settings of labels, or `null` if there has
   * been none
   *
   * Set by {@link AppStoreActions.showLabelsSettings} to a new object on every call, so
   * that a repeated request for the same labels is a change too.
   */
  labelsSettingsRequest: { labelsId: string } | null;

  /**
   * The latest request to show the settings of points, or `null` if there has
   * been none
   *
   * Set by {@link AppStoreActions.showPointsSettings} to a new object on every call, so
   * that a repeated request for the same points is a change too.
   */
  pointsSettingsRequest: { pointsId: string } | null;

  /**
   * The latest request to show the settings of shapes, or `null` if there has
   * been none
   *
   * Set by {@link AppStoreActions.showShapesSettings} to a new object on every call, so
   * that a repeated request for the same shapes is a change too.
   */
  shapesSettingsRequest: { shapesId: string } | null;
};

/**
 * The actions of the app store
 *
 * Registering a data provider for a data source type that is already taken
 * replaces the previously registered provider, which invalidates the data
 * loaded through it.
 */
export type AppStoreActions = {
  /**
   * Opens a workspace, against which local data source paths are resolved
   *
   * @param workspace - The directory handle of the workspace, or `null` to
   * close the open workspace
   */
  setWorkspace: (workspace: FileSystemDirectoryHandle | null) => void;

  /**
   * Sets how mouse events in the viewer are interpreted
   *
   * @param interactionMode - The interaction mode to switch to
   */
  setInteractionMode: (interactionMode: InteractionMode) => void;

  /**
   * Previews a single image channel on its own, or ends the preview
   *
   * @param imageChannelPreview - The image and channel to preview, or `null` to
   * end the preview
   */
  setImageChannelPreview: (
    imageChannelPreview: ImageChannelPreview | null,
  ) => void;

  /**
   * Marks a project as open, or as closed
   *
   * @param projectOpen - Whether a project is open
   */
  setProjectOpen: (projectOpen: boolean) => void;

  /**
   * Highlights a group of an object in the viewer
   *
   * @param highlightedItemGroup - The group to highlight, or `null` to highlight
   * none
   */
  setHighlightedItemGroup: (
    highlightedItemGroup: HighlightedItemGroup | null,
  ) => void;

  /**
   * Shows the settings of an image: the Images panel is brought to the front
   * and the settings are expanded
   *
   * @param imageId - ID of the image whose settings to show
   */
  showImageSettings: (imageId: string) => void;

  /**
   * Shows the settings of labels: the Labels panel is brought to the front
   * and the settings are expanded
   *
   * @param labelsId - ID of the labels whose settings to show
   */
  showLabelsSettings: (labelsId: string) => void;

  /**
   * Shows the settings of points: the Points panel is brought to the front
   * and the settings are expanded
   *
   * @param pointsId - ID of the points whose settings to show
   */
  showPointsSettings: (pointsId: string) => void;

  /**
   * Shows the settings of shapes: the Shapes panel is brought to the front
   * and the settings are expanded
   *
   * @param shapesId - ID of the shapes whose settings to show
   */
  showShapesSettings: (shapesId: string) => void;

  /**
   * Registers an image data provider
   *
   * @param type - The data source type the provider handles
   * @param dataProvider - The data provider to register
   */
  registerImageDataProvider: (
    type: string,
    dataProvider: ImageDataProvider<ImageDataSource, ImageData>,
  ) => void;

  /**
   * Registers a labels data provider
   *
   * @param type - The data source type the provider handles
   * @param dataProvider - The data provider to register
   */
  registerLabelsDataProvider: (
    type: string,
    dataProvider: LabelsDataProvider<LabelsDataSource, LabelsData>,
  ) => void;

  /**
   * Registers a points data provider
   *
   * @param type - The data source type the provider handles
   * @param dataProvider - The data provider to register
   */
  registerPointsDataProvider: (
    type: string,
    dataProvider: PointsDataProvider<PointsDataSource, PointsData>,
  ) => void;

  /**
   * Registers a shapes data provider
   *
   * @param type - The data source type the provider handles
   * @param dataProvider - The data provider to register
   */
  registerShapesDataProvider: (
    type: string,
    dataProvider: ShapesDataProvider<ShapesDataSource, ShapesData>,
  ) => void;

  /**
   * Registers a table data provider
   *
   * @param type - The data source type the provider handles
   * @param dataProvider - The data provider to register
   */
  registerTableDataProvider: (
    type: string,
    dataProvider: TableDataProvider<TableDataSource, TableData>,
  ) => void;
};

/**
 * The app store, i.e. its state and actions
 */
export type AppStore = AppStoreState & AppStoreActions;

/**
 * The API through which the app store is read, written and subscribed to
 */
export type AppStoreApi = Mutate<
  StoreApi<AppStore>,
  [["zustand/immer", never]]
>;
