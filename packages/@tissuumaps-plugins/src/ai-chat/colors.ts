import {
  type Color,
  categoricalColorPalettes,
  continuousColorPalettes,
} from "@tissuumaps/core";

import { isRecord, same } from "./objects";

/** The colors the model may name */
export const colorNames: Record<string, Color> = {
  red: { r: 255, g: 0, b: 0 },
  green: { r: 0, g: 200, b: 0 },
  blue: { r: 0, g: 90, b: 255 },
  yellow: { r: 255, g: 220, b: 0 },
  orange: { r: 255, g: 140, b: 0 },
  purple: { r: 140, g: 60, b: 200 },
  pink: { r: 255, g: 105, b: 180 },
  brown: { r: 140, g: 80, b: 30 },
  cyan: { r: 0, g: 220, b: 220 },
  magenta: { r: 255, g: 0, b: 255 },
  white: { r: 255, g: 255, b: 255 },
  black: { r: 0, g: 0, b: 0 },
  grey: { r: 128, g: 128, b: 128 },
  "light grey": { r: 211, g: 211, b: 211 },
  "dark grey": { r: 80, g: 80, b: 80 },
};

/** The color of the other groups when the user does not name one */
export const othersColor = "light grey";

/** Words a small model may put into the colors for "all the other groups" */
export const othersWords = [
  "others",
  "other",
  "the others",
  "rest",
  "all others",
];

/** The IDs of the palettes for groups, from TissUUmaps */
export const groupPalettes = categoricalColorPalettes.map(
  (palette) => palette.id,
);

/** The IDs of the palettes for numbers, from TissUUmaps */
export const numberPalettes = continuousColorPalettes.map(
  (palette) => palette.id,
);

/**
 * Turns "Red", "light gray" or "#ff0000" into a color
 *
 * @param text - The color as the model wrote it
 * @returns The color, or `undefined` if it is not known
 */
export function toColor(text: unknown): Color | undefined {
  const name = String(text)
    .trim()
    .toLowerCase()
    .replaceAll("gray", "grey")
    .replace(/^(light|dark)-?grey$/, "$1 grey");
  const named = colorNames[name];
  if (named !== undefined) {
    return named;
  }
  const hex = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/.exec(name);
  if (hex?.[1] === undefined || hex[2] === undefined || hex[3] === undefined) {
    return undefined;
  }
  return {
    r: parseInt(hex[1], 16),
    g: parseInt(hex[2], 16),
    b: parseInt(hex[3], 16),
  };
}

/**
 * Names a color for the summary, e.g. "red" or "rgb(10, 20, 30)"
 *
 * @param color - The color
 * @returns Its name
 */
export function colorName(color: Color): string {
  return (
    Object.keys(colorNames).find((name) => same(colorNames[name], color)) ??
    `rgb(${color.r}, ${color.g}, ${color.b})`
  );
}

/**
 * Finds a palette by name, among those for groups or those for numbers
 *
 * @param name - The palette name the model gave
 * @param forGroups - Whether a palette for groups is wanted
 * @returns The palette ID, or `undefined` if there is none
 */
export function findPalette(
  name: unknown,
  forGroups: boolean,
): string | undefined {
  const lower = String(name).toLowerCase();
  // the batlow made for groups
  if (forGroups && lower === "batlow") {
    return "batlowS";
  }
  return (forGroups ? groupPalettes : numberPalettes).find(
    (palette) => palette.toLowerCase() === lower,
  );
}

/** Whether a value is a color, e.g. {"r": 255, "g": 0, "b": 0} */
export function isRgb(value: unknown): value is Color {
  return (
    isRecord(value) &&
    Object.keys(value).length === 3 &&
    ["r", "g", "b"].every((key) => {
      const component = value[key];
      return (
        typeof component === "number" &&
        Number.isInteger(component) &&
        component >= 0 &&
        component <= 255
      );
    })
  );
}
