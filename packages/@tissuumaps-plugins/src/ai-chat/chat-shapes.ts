import {
  type AppStoreApi,
  type Color,
  type ProjectStoreApi,
  type ShapesData,
  type ShapesDataProvider,
  type ShapesDataSource,
  type ShapesGeometry,
  createShapes,
} from "@tissuumaps/core";

import { toColor } from "./colors";

// Plugins made by the chat can show temporary shapes on the image, drawn by
// TissUUmaps itself, in the coordinates of the data: e.g. a square of the
// size of a pixel of a method and a circle of its smoothing radius, where
// the cells are (chat.showShapes, chat.hideShapes). The shapes are a shapes
// object of the project, with the layer and the transform of the points they
// go with, so that they use the coordinates of those points. Their geometry
// lives in memory: they have the data source type below, which this file
// loads from memory, as computed-tables.ts does for the computed tables.
// They are removed with chat.hideShapes, or when the plugin is removed.

/** The data source type of the shapes shown by plugins */
export const chatShapesType = "ai-chat-shapes";

/** The most shapes that one call shows */
export const maxChatShapes = 1000;

/** The geometries of the shapes shown, by data source */
const geometries = new Map<string, ShapesGeometry>();

/** A shape as a plugin gives it, in the coordinates of the data */
type ShapeInput =
  | { rect: [number, number, number, number] }
  | { circle: [number, number, number] }
  | { polygon: [number, number][] };

/**
 * Turns the shapes of a plugin into rings of vertices, or says what is wrong
 *
 * @param shapes - The shapes, as the plugin gave them
 * @returns The ring of each shape, as x, y, x, y, …
 */
export function shapeRings(shapes: unknown): number[][] | string {
  if (!Array.isArray(shapes) || shapes.length === 0) {
    return "give shapes, a list such as [{ rect: [x, y, width, height] }, { circle: [x, y, radius] }, { polygon: [[x, y], [x, y], [x, y]] }]";
  }
  if (shapes.length > maxChatShapes) {
    return `at most ${maxChatShapes} shapes at a time`;
  }
  const numbers = (value: unknown, count?: number): value is number[] =>
    Array.isArray(value) &&
    (count === undefined || value.length === count) &&
    value.every((item) => typeof item === "number" && Number.isFinite(item));
  const rings: number[][] = [];
  for (const [index, shape] of (shapes as unknown[]).entries()) {
    const given = (
      typeof shape === "object" && shape !== null ? shape : {}
    ) as Record<string, unknown>;
    const { rect, circle, polygon } = given as Partial<ShapeInput> &
      Record<string, unknown>;
    if (numbers(rect, 4) && rect[2]! > 0 && rect[3]! > 0) {
      const [x, y, width, height] = rect as [number, number, number, number];
      rings.push([x, y, x + width, y, x + width, y + height, x, y + height]);
    } else if (numbers(circle, 3) && circle[2]! > 0) {
      const [x, y, radius] = circle as [number, number, number];
      const ring: number[] = [];
      for (let step = 0; step < 64; step++) {
        const angle = (2 * Math.PI * step) / 64;
        ring.push(x + radius * Math.cos(angle), y + radius * Math.sin(angle));
      }
      rings.push(ring);
    } else if (
      Array.isArray(polygon) &&
      polygon.length >= 3 &&
      polygon.every((point) => numbers(point, 2))
    ) {
      rings.push((polygon as [number, number][]).flat());
    } else {
      return `shapes[${index}] must be { rect: [x, y, width, height] } (width and height above 0), { circle: [x, y, radius] } (radius above 0) or { polygon: [[x, y], …] } (at least 3 points), with numbers`;
    }
  }
  return rings;
}

/**
 * Makes the geometry of TissUUmaps for shapes of one ring each
 *
 * @param rings - The rings, as x, y, x, y, …
 * @returns The geometry
 */
export function ringsGeometry(rings: number[][]): ShapesGeometry {
  const count = rings.length;
  const offsets = Uint32Array.from({ length: count + 1 }, (_, i) => i);
  const vertexOffsets = new Uint32Array(count + 1);
  rings.forEach((ring, index) => {
    vertexOffsets[index + 1] = vertexOffsets[index]! + ring.length / 2;
  });
  return {
    shapePolygonOffsets: offsets,
    polygonRingOffsets: offsets.slice(),
    ringVertexOffsets: vertexOffsets,
    coords: Float32Array.from(rings.flat()),
  };
}

/**
 * The data of shapes kept in memory
 *
 * @param geometry - Their geometry
 * @returns The data, as TissUUmaps reads it
 */
function shapesData(geometry: ShapesGeometry): ShapesData {
  const count = geometry.shapePolygonOffsets.length - 1;
  const ids = Uint32Array.from({ length: count }, (_, i) => i);
  return {
    getIds: () => ids,
    getSize: () => count,
    loadGeometry: () => Promise.resolve(geometry),
    close: () => undefined,
  };
}

/** The data provider of the shapes shown by plugins, from memory */
export const chatShapesProvider: ShapesDataProvider<
  ShapesDataSource,
  ShapesData
> = {
  name: "AI chat (shapes of a plugin)",
  schema: {
    type: "object",
    properties: { source: { type: "string", readOnly: true } },
  },
  uischema: { type: "VerticalLayout", elements: [] },
  normalize: (dataSource) => dataSource,
  load: (dataSource) => {
    const geometry =
      dataSource.source === undefined
        ? undefined
        : geometries.get(dataSource.source);
    return geometry === undefined
      ? Promise.reject(
          new Error(
            "these shapes were shown by a plugin of the AI chat in an earlier session; they are gone",
          ),
        )
      : Promise.resolve(shapesData(geometry));
  },
};

/**
 * Registers the data provider of the shapes of plugins, once
 *
 * @param appStore - The app store
 */
export function registerChatShapes(appStore: AppStoreApi): void {
  const state = appStore.getState();
  if (
    typeof state.registerShapesDataProvider === "function" &&
    state.shapesDataProviders?.get(chatShapesType) === undefined
  ) {
    state.registerShapesDataProvider(chatShapesType, chatShapesProvider);
  }
}

/**
 * Reads a color given by a plugin: a name, "#rrggbb", { r, g, b } or [r, g, b]
 *
 * @param value - The color
 * @param fallback - The color when none is given
 * @returns The color, or undefined if it is not one
 */
function colorOf(value: unknown, fallback: Color): Color | undefined {
  if (value === undefined || value === null) {
    return fallback;
  }
  if (Array.isArray(value)) {
    const [r, g, b] = value as unknown[];
    value = { r, g, b };
  }
  if (typeof value === "object") {
    const { r, g, b } = value as Record<string, unknown>;
    return [r, g, b].every(
      (part) =>
        typeof part === "number" &&
        Number.isInteger(part) &&
        part >= 0 &&
        part <= 255,
    )
      ? { r: r as number, g: g as number, b: b as number }
      : undefined;
  }
  return toColor(value);
}

/** The shapes of a plugin on the image */
export type ChatShapes = {
  /** Shows shapes; returns the id of their shapes object */
  show: (options: unknown) => string;
  /** Removes shapes shown before, or all of them */
  hide: (id?: unknown) => void;
  /** Removes all of them, when the plugin is removed */
  dispose: () => void;
};

/**
 * Makes what shows and hides the shapes of one plugin
 *
 * @param projectStore - The project store
 * @param owner - The id of the plugin, e.g. "chat-points2regions"
 * @returns showShapes and hideShapes, for chat
 */
export function makeChatShapes(
  projectStore: ProjectStoreApi,
  owner: string,
): ChatShapes {
  const shown = new Set<string>();
  let next = 0;

  const hide = (id?: unknown) => {
    const project = projectStore.getState();
    const which =
      id === undefined || id === null
        ? [...shown]
        : typeof id === "string" && shown.has(id)
          ? [id]
          : [];
    for (const shapesId of which) {
      const object = project.shapes.find((item) => item.id === shapesId);
      if (object !== undefined) {
        project.deleteShapes(shapesId);
        if (object.dataSource.source !== undefined) {
          geometries.delete(object.dataSource.source);
        }
      }
      shown.delete(shapesId);
    }
  };

  const show = (given: unknown): string => {
    const options = (
      typeof given === "object" && given !== null && !Array.isArray(given)
        ? given
        : {}
    ) as Record<string, unknown>;
    const rings = shapeRings(options["shapes"]);
    if (typeof rings === "string") {
      throw new Error(`chat.showShapes: ${rings}`);
    }
    const project = projectStore.getState();
    // the points (or labels) whose coordinates the shapes use
    const like = options["like"];
    const all = [...project.points, ...project.labels];
    const found =
      like === undefined
        ? project.points[0]
        : all.find(({ id, name }) => id === like || name === like);
    if (found === undefined) {
      throw new Error(
        like === undefined
          ? "chat.showShapes: the project has no points, whose coordinates the shapes would use; give like with the id or name of labels"
          : `chat.showShapes: no points or labels ${JSON.stringify(like)}; the project has ${all.map(({ name }) => `"${name}"`).join(", ") || "none"}`,
      );
    }
    const layer =
      typeof found.layer === "string" ? found.layer : project.layers[0]?.id;
    if (layer === undefined) {
      throw new Error("chat.showShapes: the project has no layer");
    }
    const stroke = colorOf(options["color"], { r: 255, g: 0, b: 255 });
    if (stroke === undefined) {
      throw new Error(
        'chat.showShapes: color must be a name, "#rrggbb", { r, g, b } or [r, g, b]',
      );
    }
    const fill = options["fill"] === true;
    const source = `${chatShapesType}:${crypto.randomUUID()}`;
    geometries.set(source, ringsGeometry(rings));
    const id = `${owner}-shapes-${++next}`;
    const name =
      typeof options["name"] === "string" && options["name"].trim() !== ""
        ? options["name"].trim()
        : "Shapes";
    project.addShapes(
      createShapes({
        id,
        name: `${name} (${owner}, temporary)`,
        dataSource: { type: chatShapesType, source },
        layer,
        transform: structuredClone(found.transform),
        shapeStrokeColor: { constant: { value: stroke } },
        shapeFillColor: { constant: { value: stroke } },
        shapeFillVisibility: { constant: { value: fill } },
        shapeFillOpacity: { constant: { value: 0.25 } },
      }),
    );
    shown.add(id);
    return id;
  };

  return { show, hide, dispose: () => hide() };
}
