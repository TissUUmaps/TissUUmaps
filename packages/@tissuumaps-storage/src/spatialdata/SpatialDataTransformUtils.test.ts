import { describe, expect, it } from "vitest";

import { TransformUtils } from "@tissuumaps/core";

import {
  type CoordinateTransformation,
  SpatialDataTransformUtils,
} from "./SpatialDataTransformUtils";

function system(name: string, axes: string[]) {
  return { name, axes: axes.map((axis) => ({ name: axis })) };
}

/** Rounds the numbers of a transform, for comparisons */
function rounded(transform: {
  flip: boolean;
  scale: number;
  rotation: number;
  translation: { x: number; y: number };
}) {
  const round = (value: number) => Math.round(value * 1e9) / 1e9 + 0;
  return {
    flip: transform.flip,
    scale: round(transform.scale),
    rotation: round(transform.rotation),
    translation: {
      x: round(transform.translation.x),
      y: round(transform.translation.y),
    },
  };
}

function getTransform(
  transformation: CoordinateTransformation,
  axes = ["x", "y"],
) {
  const transforms = SpatialDataTransformUtils.getSimilarityTransforms(
    [transformation],
    axes,
  );
  const { transform, exact } = [...transforms.values()][0]!;
  return { transform: rounded(transform), exact };
}

describe("SpatialDataTransformUtils", () => {
  describe("getSimilarityTransforms", () => {
    const xy = system("xy", ["x", "y"]);
    const global = system("global", ["x", "y"]);

    it("maps the identity to the identity", () => {
      expect(
        getTransform({ type: "identity", input: xy, output: global }),
      ).toEqual({
        transform: {
          flip: false,
          scale: 1,
          rotation: 0,
          translation: { x: 0, y: 0 },
        },
        exact: true,
      });
    });

    it("applies a sequence in order, by axis name, ignoring other axes", () => {
      // as written by spatialdata for a cyx image
      const cyx = system("cyx", ["c", "y", "x"]);
      const output = system("global", ["c", "y", "x"]);
      expect(
        getTransform(
          {
            type: "sequence",
            input: cyx,
            output,
            transformations: [
              { type: "scale", scale: [1, 2, 2], input: cyx, output },
              {
                type: "translation",
                translation: [0, 30, 10],
                input: cyx,
                output,
              },
            ],
          },
          ["c", "y", "x"],
        ),
      ).toEqual({
        transform: {
          flip: false,
          scale: 2,
          rotation: 0,
          translation: { x: 10, y: 30 },
        },
        exact: true,
      });
    });

    it("approximates different scales of the axes by their geometric mean", () => {
      expect(
        getTransform({
          type: "scale",
          scale: [4, 9],
          input: xy,
          output: global,
        }),
      ).toEqual({
        transform: {
          flip: false,
          scale: 6,
          rotation: 0,
          translation: { x: 0, y: 0 },
        },
        exact: false,
      });
    });

    it("reads an affine matrix by axis, with its translation last", () => {
      // a quarter turn, scaled by 2, then translated; the last row is unused
      expect(
        getTransform({
          type: "affine",
          affine: [
            [0, -2, 5],
            [2, 0, 7],
            [0, 0, 1],
          ],
          input: xy,
          output: global,
        }),
      ).toEqual({
        transform: {
          flip: false,
          scale: 2,
          rotation: 90,
          translation: { x: 5, y: 7 },
        },
        exact: true,
      });
    });

    it("reads a rotation matrix", () => {
      const c = Math.cos(Math.PI / 6);
      const s = Math.sin(Math.PI / 6);
      expect(
        getTransform({
          type: "rotation",
          rotation: [
            [c, -s],
            [s, c],
          ],
          input: xy,
          output: global,
        }),
      ).toEqual({
        transform: {
          flip: false,
          scale: 1,
          rotation: 30,
          translation: { x: 0, y: 0 },
        },
        exact: true,
      });
    });

    it("maps output axes to input axes, which may flip the plane", () => {
      expect(
        getTransform({
          type: "mapAxis",
          mapAxis: { x: "y", y: "x" },
          input: xy,
          output: global,
        }),
      ).toEqual({
        transform: {
          flip: true,
          scale: 1,
          rotation: -90,
          translation: { x: 0, y: 0 },
        },
        exact: true,
      });
    });

    it("keeps an output axis that is not mapped as the input axis of its name", () => {
      expect(
        getTransform({
          type: "sequence",
          input: xy,
          output: global,
          transformations: [
            { type: "translation", translation: [5, 7], input: xy, output: xy },
            { type: "mapAxis", mapAxis: {}, input: xy, output: global },
          ],
        }),
      ).toEqual({
        transform: {
          flip: false,
          scale: 1,
          rotation: 0,
          translation: { x: 5, y: 7 },
        },
        exact: true,
      });
    });

    it("returns one transform per coordinate system, the first one given", () => {
      const transforms = SpatialDataTransformUtils.getSimilarityTransforms(
        [
          { type: "identity", input: xy, output: global },
          {
            type: "scale",
            scale: [2, 2],
            input: xy,
            output: system("aligned", ["x", "y"]),
          },
          {
            type: "scale",
            scale: [3, 3],
            input: xy,
            output: global,
          },
        ],
        ["x", "y"],
      );
      expect([...transforms.keys()]).toEqual(["global", "aligned"]);
      expect(transforms.get("global")!.transform.scale).toBe(1);
      expect(transforms.get("aligned")!.transform.scale).toBe(2);
    });

    it("uses the element's axes for a transformation without input axes", () => {
      expect(
        getTransform(
          {
            type: "translation",
            translation: [1, 2],
            output: system("global", ["y", "x"]),
          },
          ["y", "x"],
        ).transform.translation,
      ).toEqual({ x: 2, y: 1 });
    });

    const cos = Math.cos(0.7);
    const sin = Math.sin(0.7);
    it.each([
      [
        "a rotation, scale and translation",
        [
          [2 * cos, -2 * sin, 5],
          [2 * sin, 2 * cos, -3],
        ],
      ],
      [
        "a swap of the axes",
        [
          [0, 1, 0],
          [1, 0, 0],
        ],
      ],
      [
        "a reflection of x",
        [
          [-1, 0, 10],
          [0, 1, 0],
        ],
      ],
      [
        "a reflection of y",
        [
          [1, 0, 0],
          [0, -1, 7],
        ],
      ],
      [
        "a reflection, rotation and scale",
        [
          [-3 * cos, -3 * sin, 1],
          [-3 * sin, 3 * cos, 2],
        ],
      ],
    ])("returns a transform that composes back to %s", (_, [xRow, yRow]) => {
      const { transform, exact } = [
        ...SpatialDataTransformUtils.getSimilarityTransforms(
          [
            {
              type: "affine",
              affine: [xRow!, yRow!, [0, 0, 1]],
              input: xy,
              output: global,
            },
          ],
          ["x", "y"],
        ).values(),
      ][0]!;
      expect(exact).toBe(true);
      // gl-matrix matrices are column-major
      const m = TransformUtils.compose(transform);
      const composed = [m[0], m[3], m[6], m[1], m[4], m[7]];
      [...xRow!, ...yRow!].forEach((value, i) => {
        expect(composed[i]).toBeCloseTo(value, 4);
      });
    });

    it("rejects an unknown transformation type", () => {
      expect(() =>
        getTransform({
          type: "bspline",
          input: xy,
          output: global,
        } as unknown as CoordinateTransformation),
      ).toThrow('Unknown coordinate transformation type "bspline"');
    });
  });
});
