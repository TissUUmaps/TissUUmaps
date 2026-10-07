import {
  type ImageDataSource,
  type LabelsDataSource,
  type PointsDataSource,
  type ShapesDataSource,
  SourceUtils,
  type TableDataSource,
  createImage,
  createLabels,
  createPoints,
  createShapes,
  createTable,
} from "@tissuumaps/core";

import { appStore } from "@/stores/app";
import { projectStore } from "@/stores/project";

/**
 * Creates an ID for a data object that is unique among the IDs of its kind
 *
 * The ID is the file name of the source with every `.` replaced by `-`, so
 * `cells.ome.zarr` gives `cells-ome-zarr`; if that is taken, it gets the first
 * free numeric suffix, starting at `-2`. Without a source, or for URLs without
 * a path, it is a random UUID, which is assumed to be unique.
 *
 * @param normalizedSource - The normalized source of the data object, if any
 * @param existingIds - The IDs of the existing data objects of the same kind
 * @returns The ID
 * @throws See `SourceUtils.getPathSegments`
 */
export function createDataObjectID(
  normalizedSource: string | undefined,
  existingIds: string[],
): string {
  const fileName =
    normalizedSource !== undefined
      ? SourceUtils.getPathSegments(normalizedSource).at(-1)
      : undefined;
  if (!fileName) {
    return crypto.randomUUID();
  }
  const id = fileName.replaceAll(".", "-");
  const ids = new Set(existingIds);
  if (!ids.has(id)) {
    return id;
  }
  let n = 2;
  while (ids.has(`${id}-${n}`)) {
    n++;
  }
  return `${id}-${n}`;
}

/**
 * Normalizes the source of a data object to add, against the open workspace
 * and the project source
 *
 * @param source - The source as entered, if any
 * @returns The normalized source, if any
 * @throws See `SourceUtils.normalizeSource`
 */
function normalizeSource(source: string | undefined): string | undefined {
  if (source === undefined) {
    return undefined;
  }
  return SourceUtils.normalizeSource(
    source,
    appStore.getState().workspace,
    projectStore.getState().source,
  );
}

/**
 * Adds an image to the project and expands it in the images panel
 *
 * @param name - The name of the image
 * @param layerId - The ID of the layer to add the image to
 * @param origSource - The source the data source had before it was prepared,
 * which the ID is created from (see {@link createDataObjectID})
 * @param preparedDataSource - The data source of the image, prepared (see
 * `DataProvider.prepareDataSource`)
 * @returns The ID of the added image
 * @throws Error if no layer is given, or if the source cannot be normalized
 */
export function addImageDataObject(
  name: string,
  layerId: string | undefined,
  origSource: string | undefined,
  preparedDataSource: ImageDataSource,
): string {
  if (layerId === undefined) {
    throw new Error("Images have to be added to a layer.");
  }
  const id = createDataObjectID(
    normalizeSource(origSource),
    projectStore.getState().images.map((image) => image.id),
  );
  projectStore
    .getState()
    .addImage(
      createImage({ id, name, dataSource: preparedDataSource, layer: layerId }),
    );
  const { expandedImageIds, setExpandedImageIds } = appStore.getState();
  if (!expandedImageIds.includes(id)) {
    setExpandedImageIds([...expandedImageIds, id]);
  }
  return id;
}

/**
 * Adds labels to the project and expands them in the labels panel
 *
 * @param name - The name of the labels
 * @param layerId - The ID of the layer to add the labels to
 * @param origSource - The source the data source had before it was prepared,
 * which the ID is created from (see {@link createDataObjectID})
 * @param preparedDataSource - The data source of the labels, prepared (see
 * `DataProvider.prepareDataSource`)
 * @returns The ID of the added labels
 * @throws Error if no layer is given, or if the source cannot be normalized
 */
export function addLabelsDataObject(
  name: string,
  layerId: string | undefined,
  origSource: string | undefined,
  preparedDataSource: LabelsDataSource,
): string {
  if (layerId === undefined) {
    throw new Error("Labels have to be added to a layer.");
  }
  const id = createDataObjectID(
    normalizeSource(origSource),
    projectStore.getState().labels.map((labels) => labels.id),
  );
  projectStore.getState().addLabels(
    createLabels({
      id,
      name,
      dataSource: preparedDataSource,
      layer: layerId,
    }),
  );
  const { expandedLabelsIds, setExpandedLabelsIds } = appStore.getState();
  if (!expandedLabelsIds.includes(id)) {
    setExpandedLabelsIds([...expandedLabelsIds, id]);
  }
  return id;
}

/**
 * Adds points to the project and expands them in the points panel
 *
 * @param name - The name of the points
 * @param layerId - The ID of the layer to add the points to
 * @param origSource - The source the data source had before it was prepared,
 * which the ID is created from (see {@link createDataObjectID})
 * @param preparedDataSource - The data source of the points, prepared (see
 * `DataProvider.prepareDataSource`)
 * @returns The ID of the added points
 * @throws Error if no layer is given, or if the source cannot be normalized
 */
export function addPointsDataObject(
  name: string,
  layerId: string | undefined,
  origSource: string | undefined,
  preparedDataSource: PointsDataSource,
): string {
  if (layerId === undefined) {
    throw new Error("Points have to be added to a layer.");
  }
  const id = createDataObjectID(
    normalizeSource(origSource),
    projectStore.getState().points.map((points) => points.id),
  );
  projectStore.getState().addPoints(
    createPoints({
      id,
      name,
      dataSource: preparedDataSource,
      layer: layerId,
    }),
  );
  const { expandedPointsIds, setExpandedPointsIds } = appStore.getState();
  if (!expandedPointsIds.includes(id)) {
    setExpandedPointsIds([...expandedPointsIds, id]);
  }
  return id;
}

/**
 * Adds shapes to the project and expands them in the shapes panel
 *
 * @param name - The name of the shapes
 * @param layerId - The ID of the layer to add the shapes to
 * @param origSource - The source the data source had before it was prepared,
 * which the ID is created from (see {@link createDataObjectID})
 * @param preparedDataSource - The data source of the shapes, prepared (see
 * `DataProvider.prepareDataSource`)
 * @returns The ID of the added shapes
 * @throws Error if no layer is given, or if the source cannot be normalized
 */
export function addShapesDataObject(
  name: string,
  layerId: string | undefined,
  origSource: string | undefined,
  preparedDataSource: ShapesDataSource,
): string {
  if (layerId === undefined) {
    throw new Error("Shapes have to be added to a layer.");
  }
  const id = createDataObjectID(
    normalizeSource(origSource),
    projectStore.getState().shapes.map((shapes) => shapes.id),
  );
  projectStore.getState().addShapes(
    createShapes({
      id,
      name,
      dataSource: preparedDataSource,
      layer: layerId,
    }),
  );
  const { expandedShapesIds, setExpandedShapesIds } = appStore.getState();
  if (!expandedShapesIds.includes(id)) {
    setExpandedShapesIds([...expandedShapesIds, id]);
  }
  return id;
}

/**
 * Adds a table to the project and expands it in the tables panel
 *
 * @param name - The name of the table
 * @param origSource - The source the data source had before it was prepared,
 * which the ID is created from (see {@link createDataObjectID})
 * @param preparedDataSource - The data source of the table, prepared (see
 * `DataProvider.prepareDataSource`)
 * @returns The ID of the added table
 * @throws Error if the source cannot be normalized
 */
export function addTableDataObject(
  name: string,
  origSource: string | undefined,
  preparedDataSource: TableDataSource,
): string {
  const id = createDataObjectID(
    normalizeSource(origSource),
    projectStore.getState().tables.map((table) => table.id),
  );
  projectStore
    .getState()
    .addTable(createTable({ id, name, dataSource: preparedDataSource }));
  const { expandedTableIds, setExpandedTableIds } = appStore.getState();
  if (!expandedTableIds.includes(id)) {
    setExpandedTableIds([...expandedTableIds, id]);
  }
  return id;
}
