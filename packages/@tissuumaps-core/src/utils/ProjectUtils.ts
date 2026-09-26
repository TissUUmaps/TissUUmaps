import type {
  ColorConfig,
  MarkerConfig,
  OpacityConfig,
  SizeConfig,
  VisibilityConfig,
} from "../model/configs";
import type { Project } from "../model/project";

/** Utility methods for projects */
export class ProjectUtils {
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
