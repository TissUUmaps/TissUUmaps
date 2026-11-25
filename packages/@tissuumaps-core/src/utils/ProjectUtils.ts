import type {
  ColorConfig,
  MarkerConfig,
  OpacityConfig,
  SizeConfig,
  TableColumnRef,
  VisibilityConfig,
} from "../model/configs";
import type { Layer } from "../model/layer";
import type { Project } from "../model/project";

/** Utility methods for projects */
export class ProjectUtils {
  /**
   * Returns whether a project has any data objects
   *
   * @param project - The project, or only its data objects
   * @returns `true` if the project has images, labels, points, shapes or tables
   */
  static hasData(
    project: Pick<
      Project,
      "images" | "labels" | "points" | "shapes" | "tables"
    >,
  ): boolean {
    return (
      project.images.length > 0 ||
      project.labels.length > 0 ||
      project.points.length > 0 ||
      project.shapes.length > 0 ||
      project.tables.length > 0
    );
  }

  /**
   * Returns whether an object is visible
   *
   * An object is visible if it is visible itself and its layer is visible. An
   * object whose items are assigned to layers by a table column is visible if
   * it is visible itself.
   *
   * @param object - The object, or only its visibility and layer
   * @param layers - The layers of the project
   * @returns `true` if the object is visible
   */
  static isObjectVisible(
    object: { visibility: boolean; layer: string | TableColumnRef },
    layers: Layer[],
  ): boolean {
    return (
      object.visibility &&
      (typeof object.layer !== "string" ||
        layers.find(({ id }) => id === object.layer)?.visibility !== false)
    );
  }

  /**
   * Returns the IDs of the loaded objects of one kind after a project change
   *
   * The loaded objects are the previously loaded objects that are still in the
   * project, and every visible object (see {@link ProjectUtils.isObjectVisible}).
   *
   * @param objects - The objects of one kind, e.g. the project's images
   * @param layers - The layers of the project
   * @param loadedObjectIds - The IDs of the previously loaded objects
   * @returns The IDs of the loaded objects, in project order, or
   * `loadedObjectIds` itself if they are the same objects
   */
  static getLoadedObjectIds(
    objects: {
      id: string;
      visibility: boolean;
      layer: string | TableColumnRef;
    }[],
    layers: Layer[],
    loadedObjectIds: string[],
  ): string[] {
    const prevLoadedObjectIds = new Set(loadedObjectIds);
    const newLoadedObjectIds = objects
      .filter(
        (object) =>
          prevLoadedObjectIds.has(object.id) ||
          ProjectUtils.isObjectVisible(object, layers),
      )
      .map((object) => object.id);
    return newLoadedObjectIds.length === prevLoadedObjectIds.size &&
      newLoadedObjectIds.every((id) => prevLoadedObjectIds.has(id))
      ? loadedObjectIds
      : newLoadedObjectIds;
  }

  /**
   * Returns the marker configurations of a project's points
   *
   * @param project - The project, or only its labels, points and shapes
   * @returns The marker configurations
   */
  static getMarkerConfigs(
    project: Pick<Project, "labels" | "points" | "shapes">,
  ): MarkerConfig[] {
    return project.points.map((points) => points.pointMarker);
  }

  /**
   * Returns the size configurations of a project's points
   *
   * @param project - The project, or only its labels, points and shapes
   * @returns The size configurations
   */
  static getSizeConfigs(
    project: Pick<Project, "labels" | "points" | "shapes">,
  ): SizeConfig[] {
    return project.points.map((points) => points.pointSize);
  }

  /**
   * Returns the color configurations of a project's labels, points and shapes
   *
   * @param project - The project, or only its labels, points and shapes
   * @returns The color configurations, including the fill and stroke colors of
   * shapes
   */
  static getColorConfigs(
    project: Pick<Project, "labels" | "points" | "shapes">,
  ): ColorConfig[] {
    return [
      ...project.labels.map((labels) => labels.labelColor),
      ...project.points.map((points) => points.pointColor),
      ...project.shapes.flatMap((shapes) => [
        shapes.shapeFillColor,
        shapes.shapeStrokeColor,
      ]),
    ];
  }

  /**
   * Returns the visibility configurations of a project's labels, points and
   * shapes
   *
   * @param project - The project, or only its labels, points and shapes
   * @returns The visibility configurations, including the shape, fill and
   * stroke visibilities of shapes
   */
  static getVisibilityConfigs(
    project: Pick<Project, "labels" | "points" | "shapes">,
  ): VisibilityConfig[] {
    return [
      ...project.labels.map((labels) => labels.labelVisibility),
      ...project.points.map((points) => points.pointVisibility),
      ...project.shapes.flatMap((shapes) => [
        shapes.shapeVisibility,
        shapes.shapeFillVisibility,
        shapes.shapeStrokeVisibility,
      ]),
    ];
  }

  /**
   * Returns the opacity configurations of a project's labels, points and shapes
   *
   * @param project - The project, or only its labels, points and shapes
   * @returns The opacity configurations, including the shape, fill and stroke
   * opacities of shapes
   */
  static getOpacityConfigs(
    project: Pick<Project, "labels" | "points" | "shapes">,
  ): OpacityConfig[] {
    return [
      ...project.labels.map((labels) => labels.labelOpacity),
      ...project.points.map((points) => points.pointOpacity),
      ...project.shapes.flatMap((shapes) => [
        shapes.shapeOpacity,
        shapes.shapeFillOpacity,
        shapes.shapeStrokeOpacity,
      ]),
    ];
  }
}
