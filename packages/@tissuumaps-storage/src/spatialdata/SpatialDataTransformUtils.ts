import { type SimilarityTransform, TransformUtils } from "@tissuumaps/core";

/**
 * A coordinate transformation of a SpatialData element, as written in its
 * attributes (following the NGFF coordinate transformations)
 *
 * `input` and `output` name the coordinate systems and their axes; the other
 * fields hold the parameters of the respective `type`.
 */
export type CoordinateTransformation = {
  input?: { name: string; axes: { name: string }[] };
  output?: { name: string; axes: { name: string }[] };
} & (
  | { type: "identity" }
  | { type: "scale"; scale: number[] }
  | { type: "translation"; translation: number[] }
  | {
      type: "affine";
      /** Rows by output axis, columns by input axis and then the translation */
      affine: number[][];
    }
  | {
      type: "rotation";
      /** Rows by output axis, columns by input axis */
      rotation: number[][];
    }
  | {
      type: "mapAxis";
      /** The input axis by output axis, for the output axes that change */
      mapAxis: { [outputAxis: string]: string };
    }
  | { type: "sequence"; transformations: CoordinateTransformation[] }
);

/**
 * The value of an axis as a linear form `[x, y, 1]` of the element's x and y
 * coordinates, i.e. their coefficients and a constant
 */
type LinearForm = [number, number, number];

/**
 * Turns the coordinate transformations of SpatialData elements into the
 * similarity transforms of TissUUmaps objects
 *
 * Only the x and y axes are kept, so a transformation is reduced to the affine
 * map of the plane it applies to them. TissUUmaps places objects with
 * similarity transforms only, so a map that also scales the axes differently
 * or shears them is approximated.
 */
export class SpatialDataTransformUtils {
  /**
   * The largest deviation from a similarity transform, relative to its scale,
   * that is attributed to rounding rather than approximated
   */
  private static readonly _tolerance = 1e-9;

  /**
   * Returns the similarity transform by which an element maps into each of
   * its coordinate systems
   *
   * The uniform scale of an approximated transform is the geometric mean of
   * the scales of its axes (i.e. it keeps areas), and its rotation and flip are
   * those of the x axis.
   *
   * @param transformations - The coordinate transformations of the element,
   * one per coordinate system it maps into
   * @param axes - The names of the element's axes, in order, for
   * transformations that do not name their input axes
   * @returns The transform by coordinate system name, in the order of the
   * transformations, and whether it is exact rather than approximated
   * @throws Error if a transformation has an unknown type
   */
  static getSimilarityTransforms(
    transformations: CoordinateTransformation[],
    axes: string[],
  ): Map<string, { transform: SimilarityTransform; exact: boolean }> {
    const transforms = new Map<
      string,
      { transform: SimilarityTransform; exact: boolean }
    >();
    for (const transformation of transformations) {
      const coordinateSystem = transformation.output!.name;
      if (!transforms.has(coordinateSystem)) {
        transforms.set(
          coordinateSystem,
          SpatialDataTransformUtils._toSimilarityTransform(
            transformation,
            axes,
          ),
        );
      }
    }
    return transforms;
  }

  /**
   * Reduces a transformation to the affine map of the plane it applies to the
   * x and y axes, and decomposes that into a similarity transform
   *
   * @param transformation - The coordinate transformation
   * @param axes - The names of the element's axes, in order
   * @returns The similarity transform, and whether it is exact rather than
   * approximated
   */
  private static _toSimilarityTransform(
    transformation: CoordinateTransformation,
    axes: string[],
  ): { transform: SimilarityTransform; exact: boolean } {
    const forms = new Map<string, LinearForm>(
      axes.map((axis) => [
        axis,
        axis === "x" ? [1, 0, 0] : axis === "y" ? [0, 1, 0] : [0, 0, 0],
      ]),
    );
    const result = SpatialDataTransformUtils._apply(transformation, {
      axes,
      forms,
    });
    const [a, b, tx] = result.forms.get("x")!;
    const [c, d, ty] = result.forms.get("y")!;
    // gl-matrix matrices are column-major
    const decomposed = TransformUtils.decompose([a, c, 0, b, d, 0, tx, ty, 1]);
    const scale = Math.sqrt(Math.abs(a * d - b * c));
    const tolerance = SpatialDataTransformUtils._tolerance * scale;
    const exact = decomposed.flip
      ? Math.abs(a + d) <= tolerance && Math.abs(b - c) <= tolerance
      : Math.abs(a - d) <= tolerance && Math.abs(b + c) <= tolerance;
    return { transform: { ...decomposed, scale }, exact };
  }

  /** Applies a transformation to the linear forms of the axes */
  private static _apply(
    transformation: CoordinateTransformation,
    point: { axes: string[]; forms: Map<string, LinearForm> },
  ): { axes: string[]; forms: Map<string, LinearForm> } {
    if (transformation.type === "sequence") {
      return transformation.transformations.reduce(
        (current, inner) => SpatialDataTransformUtils._apply(inner, current),
        point,
      );
    }
    const inputAxes =
      transformation.input?.axes.map(({ name }) => name) ?? point.axes;
    const outputAxes =
      transformation.output?.axes.map(({ name }) => name) ?? inputAxes;
    const input = inputAxes.map(
      (axis): LinearForm => point.forms.get(axis) ?? [0, 0, 0],
    );
    let output: LinearForm[];
    switch (transformation.type) {
      case "identity":
        output = input;
        break;
      case "scale":
        output = input.map(([x, y, t], i) => {
          const s = transformation.scale[i]!;
          return [x * s, y * s, t * s];
        });
        break;
      case "translation":
        output = input.map(([x, y, t], i) => [
          x,
          y,
          t + transformation.translation[i]!,
        ]);
        break;
      case "affine":
      case "rotation": {
        const matrix =
          transformation.type === "affine"
            ? transformation.affine
            : transformation.rotation;
        output = outputAxes.map((_, i) => {
          const row = matrix[i]!;
          const form: LinearForm = [0, 0, row[input.length] ?? 0];
          input.forEach(([x, y, t], j) => {
            const coefficient = row[j]!;
            form[0] += coefficient * x;
            form[1] += coefficient * y;
            form[2] += coefficient * t;
          });
          return form;
        });
        break;
      }
      case "mapAxis":
        // an output axis that is not mapped takes the input axis of its name
        output = outputAxes.map(
          (axis) =>
            point.forms.get(transformation.mapAxis[axis] ?? axis) ?? [0, 0, 0],
        );
        break;
      default:
        throw new Error(
          `Unknown coordinate transformation type '${(transformation as { type: string }).type}'`,
        );
    }
    return {
      axes: outputAxes,
      forms: new Map(outputAxes.map((axis, i) => [axis, output[i]!])),
    };
  }
}
