import type { Mutate, StoreApi } from "zustand";

import type { Image } from "../../model/image";
import type { Labels } from "../../model/labels";
import type { Layer } from "../../model/layer";
import type { Points } from "../../model/points";
import type { Color, GroupValueMap, Marker } from "../../model/primitives";
import type { Project } from "../../model/project";
import type { Shapes } from "../../model/shapes";
import type { Table } from "../../model/table";

/**
 * The state of the project store, i.e. the currently open {@link Project}
 *
 * In addition to the project itself, the store keeps track of where the
 * project was loaded from, which is what project-relative paths in its data
 * sources are resolved against, and of an ID identifying the open project as
 * such. Both
 * describe how the project was opened rather than what it contains, so they are
 * not part of {@link Project} and are not to be saved, serialized or exported
 * with it.
 */
export type ProjectStoreState = Project & {
  /**
   * Where the open project was loaded from: its absolute URL, the
   * workspace-relative path of the project file, or `null` if it was loaded
   * from neither
   */
  source: string | null;
  /**
   * Identifies the open project regardless of its content: a fresh ID is
   * generated whenever a project is loaded or the store is cleared, so that a
   * new project can be told apart from edits to the open one
   */
  instanceId: string;
};

/**
 * The actions of the project store
 *
 * Every collection of the project - its layers, images, labels, points, shapes
 * and tables - is manipulated through the same five families of actions:
 * `add*` inserts an object, `update*` overwrites some of its properties,
 * `move*` reorders it, `delete*` removes it, and `clear*` empties the whole
 * collection. Objects are addressed by their ID, and their order within a
 * collection is meaningful: it is the order in which they are listed and
 * rendered.
 *
 * Adding an object whose ID is already taken, referring to one that is not
 * part of the project, or passing an index outside of a collection's bounds is
 * an error.
 *
 * The lists of maps are not reordered, so they have no `move*` actions.
 */
export type ProjectStoreActions = {
  /**
   * Renames the project
   *
   * @param name - The new project name
   */
  setName: (name: string) => void;

  /**
   * Adds a layer to the project
   *
   * @param layer - The layer to add
   * @param index - The index to insert at, defaulting to the end of the
   * collection
   */
  addLayer: (layer: Layer, index?: number) => void;

  /**
   * Adds an image to the project
   *
   * @param image - The image to add
   * @param index - The index to insert at, defaulting to the end of the
   * collection
   */
  addImage: (image: Image, index?: number) => void;

  /**
   * Adds labels to the project
   *
   * @param labels - The labels to add
   * @param index - The index to insert at, defaulting to the end of the
   * collection
   */
  addLabels: (labels: Labels, index?: number) => void;

  /**
   * Adds points to the project
   *
   * @param points - The points to add
   * @param index - The index to insert at, defaulting to the end of the
   * collection
   */
  addPoints: (points: Points, index?: number) => void;

  /**
   * Adds shapes to the project
   *
   * @param shapes - The shapes to add
   * @param index - The index to insert at, defaulting to the end of the
   * collection
   */
  addShapes: (shapes: Shapes, index?: number) => void;

  /**
   * Adds a table to the project
   *
   * @param table - The table to add
   * @param index - The index to insert at, defaulting to the end of the
   * collection
   */
  addTable: (table: Table, index?: number) => void;

  /**
   * Adds a marker map to the project
   *
   * @param map - The map to add
   */
  addMarkerMap: (map: GroupValueMap<Marker>) => void;

  /**
   * Adds a size map to the project
   *
   * @param map - The map to add
   */
  addSizeMap: (map: GroupValueMap<number>) => void;

  /**
   * Adds a color map to the project
   *
   * @param map - The map to add
   */
  addColorMap: (map: GroupValueMap<Color>) => void;

  /**
   * Adds a visibility map to the project
   *
   * @param map - The map to add
   */
  addVisibilityMap: (map: GroupValueMap<boolean>) => void;

  /**
   * Adds an opacity map to the project
   *
   * @param map - The map to add
   */
  addOpacityMap: (map: GroupValueMap<number>) => void;

  /**
   * Applies updates to a layer of the project
   *
   * @param layerId - The ID of the layer to update
   * @param updates - The properties to overwrite on the layer
   */
  updateLayer: (layerId: string, updates: Partial<Omit<Layer, "id">>) => void;

  /**
   * Applies updates to an image of the project
   *
   * @param imageId - The ID of the image to update
   * @param updates - The properties to overwrite on the image
   */
  updateImage: (imageId: string, updates: Partial<Omit<Image, "id">>) => void;

  /**
   * Applies updates to labels of the project
   *
   * @param labelsId - The ID of the labels to update
   * @param updates - The properties to overwrite on the labels
   */
  updateLabels: (
    labelsId: string,
    updates: Partial<Omit<Labels, "id">>,
  ) => void;

  /**
   * Applies updates to points of the project
   *
   * @param pointsId - The ID of the points to update
   * @param updates - The properties to overwrite on the points
   */
  updatePoints: (
    pointsId: string,
    updates: Partial<Omit<Points, "id">>,
  ) => void;

  /**
   * Applies updates to shapes of the project
   *
   * @param shapesId - The ID of the shapes to update
   * @param updates - The properties to overwrite on the shapes
   */
  updateShapes: (
    shapesId: string,
    updates: Partial<Omit<Shapes, "id">>,
  ) => void;

  /**
   * Applies updates to a table of the project
   *
   * @param tableId - The ID of the table to update
   * @param updates - The properties to overwrite on the table
   */
  updateTable: (tableId: string, updates: Partial<Omit<Table, "id">>) => void;

  /**
   * Applies updates to a marker map of the project
   *
   * @param mapId - The ID of the map to update
   * @param updates - The properties to overwrite on the map
   */
  updateMarkerMap: (
    mapId: string,
    updates: Partial<Omit<GroupValueMap<Marker>, "id">>,
  ) => void;

  /**
   * Applies updates to a size map of the project
   *
   * @param mapId - The ID of the map to update
   * @param updates - The properties to overwrite on the map
   */
  updateSizeMap: (
    mapId: string,
    updates: Partial<Omit<GroupValueMap<number>, "id">>,
  ) => void;

  /**
   * Applies updates to a color map of the project
   *
   * @param mapId - The ID of the map to update
   * @param updates - The properties to overwrite on the map
   */
  updateColorMap: (
    mapId: string,
    updates: Partial<Omit<GroupValueMap<Color>, "id">>,
  ) => void;

  /**
   * Applies updates to a visibility map of the project
   *
   * @param mapId - The ID of the map to update
   * @param updates - The properties to overwrite on the map
   */
  updateVisibilityMap: (
    mapId: string,
    updates: Partial<Omit<GroupValueMap<boolean>, "id">>,
  ) => void;

  /**
   * Applies updates to an opacity map of the project
   *
   * @param mapId - The ID of the map to update
   * @param updates - The properties to overwrite on the map
   */
  updateOpacityMap: (
    mapId: string,
    updates: Partial<Omit<GroupValueMap<number>, "id">>,
  ) => void;

  /**
   * Moves a layer of the project to another index
   *
   * @param layerId - The ID of the layer to move
   * @param newIndex - The index to move the layer to
   */
  moveLayer: (layerId: string, newIndex: number) => void;

  /**
   * Moves an image of the project to another index
   *
   * @param imageId - The ID of the image to move
   * @param newIndex - The index to move the image to
   */
  moveImage: (imageId: string, newIndex: number) => void;

  /**
   * Moves labels of the project to another index
   *
   * @param labelsId - The ID of the labels to move
   * @param newIndex - The index to move the labels to
   */
  moveLabels: (labelsId: string, newIndex: number) => void;

  /**
   * Moves points of the project to another index
   *
   * @param pointsId - The ID of the points to move
   * @param newIndex - The index to move the points to
   */
  movePoints: (pointsId: string, newIndex: number) => void;

  /**
   * Moves shapes of the project to another index
   *
   * @param shapesId - The ID of the shapes to move
   * @param newIndex - The index to move the shapes to
   */
  moveShapes: (shapesId: string, newIndex: number) => void;

  /**
   * Moves a table of the project to another index
   *
   * @param tableId - The ID of the table to move
   * @param newIndex - The index to move the table to
   */
  moveTable: (tableId: string, newIndex: number) => void;

  /**
   * Removes a layer from the project
   *
   * @param layerId - The ID of the layer to remove
   */
  deleteLayer: (layerId: string) => void;

  /**
   * Removes an image from the project
   *
   * @param imageId - The ID of the image to remove
   */
  deleteImage: (imageId: string) => void;

  /**
   * Removes labels from the project
   *
   * @param labelsId - The ID of the labels to remove
   */
  deleteLabels: (labelsId: string) => void;

  /**
   * Removes points from the project
   *
   * @param pointsId - The ID of the points to remove
   */
  deletePoints: (pointsId: string) => void;

  /**
   * Removes shapes from the project
   *
   * @param shapesId - The ID of the shapes to remove
   */
  deleteShapes: (shapesId: string) => void;

  /**
   * Removes a table from the project
   *
   * @param tableId - The ID of the table to remove
   */
  deleteTable: (tableId: string) => void;

  /**
   * Removes a marker map from the project
   *
   * @param mapId - The ID of the map to remove
   */
  deleteMarkerMap: (mapId: string) => void;

  /**
   * Removes a size map from the project
   *
   * @param mapId - The ID of the map to remove
   */
  deleteSizeMap: (mapId: string) => void;

  /**
   * Removes a color map from the project
   *
   * @param mapId - The ID of the map to remove
   */
  deleteColorMap: (mapId: string) => void;

  /**
   * Removes a visibility map from the project
   *
   * @param mapId - The ID of the map to remove
   */
  deleteVisibilityMap: (mapId: string) => void;

  /**
   * Removes an opacity map from the project
   *
   * @param mapId - The ID of the map to remove
   */
  deleteOpacityMap: (mapId: string) => void;

  /**
   * Removes all layers from the project
   */
  clearLayers: () => void;

  /**
   * Removes all images from the project
   */
  clearImages: () => void;

  /**
   * Removes all labels from the project
   */
  clearLabels: () => void;

  /**
   * Removes all points from the project
   */
  clearPoints: () => void;

  /**
   * Removes all shapes from the project
   */
  clearShapes: () => void;

  /**
   * Removes all tables from the project
   */
  clearTables: () => void;

  /**
   * Removes all marker maps from the project
   */
  clearMarkerMaps: () => void;

  /**
   * Removes all size maps from the project
   */
  clearSizeMaps: () => void;

  /**
   * Removes all color maps from the project
   */
  clearColorMaps: () => void;

  /**
   * Removes all visibility maps from the project
   */
  clearVisibilityMaps: () => void;

  /**
   * Removes all opacity maps from the project
   */
  clearOpacityMaps: () => void;

  /**
   * Sets the project's OpenSeadragon viewer options
   *
   * @param osOptions - The OpenSeadragon options to apply
   */
  setOSOptions: (osOptions: Project["osOptions"]) => void;

  /**
   * Sets the project's WebGL render options
   *
   * @param glOptions - The WebGL options to apply
   */
  setGLOptions: (glOptions: Project["glOptions"]) => void;

  /**
   * Replaces the project with a new, empty one
   */
  clear: () => void;
};

/**
 * The project store, i.e. its state and actions
 */
export type ProjectStore = ProjectStoreState & ProjectStoreActions;

/**
 * The API through which the project store is read, written and subscribed to
 */
export type ProjectStoreApi = Mutate<
  StoreApi<ProjectStore>,
  [["zustand/immer", never]]
>;
