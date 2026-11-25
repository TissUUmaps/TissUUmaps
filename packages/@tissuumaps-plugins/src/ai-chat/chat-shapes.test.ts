import { describe, expect, it, vi } from "vitest";

import type { ProjectStoreApi, Shapes } from "@tissuumaps/core";

import {
  chatShapesProvider,
  chatShapesType,
  makeChatShapes,
  registerChatShapes,
  ringsGeometry,
  shapeRings,
} from "./chat-shapes";

/** A project with points in a layer, with a transform */
function makeProject() {
  const shapes: Shapes[] = [];
  const transform = {
    flip: false,
    scale: 2,
    rotation: 0,
    translation: { x: 5, y: 7 },
  };
  const state = {
    layers: [{ id: "global" }],
    points: [{ id: "pts", name: "cells", layer: "global", transform }],
    labels: [{ id: "lab", name: "nuclei", layer: "other", transform }],
    shapes,
    addShapes: vi.fn((object: Shapes) => shapes.push(object)),
    deleteShapes: vi.fn((id: string) =>
      shapes.splice(
        shapes.findIndex((object) => object.id === id),
        1,
      ),
    ),
  };
  return {
    state,
    projectStore: { getState: () => state } as unknown as ProjectStoreApi,
  };
}

describe("the shapes of plugins", () => {
  it("turns rectangles, circles and polygons into rings, or says what is wrong", () => {
    const rings = shapeRings([
      { rect: [10, 20, 4, 2] },
      { circle: [0, 0, 1] },
      {
        polygon: [
          [0, 0],
          [1, 0],
          [0, 1],
        ],
      },
    ]);
    expect(Array.isArray(rings)).toBe(true);
    const [rect, circle, polygon] = rings as number[][];
    expect(rect).toEqual([10, 20, 14, 20, 14, 22, 10, 22]);
    expect(circle).toHaveLength(128);
    expect(polygon).toEqual([0, 0, 1, 0, 0, 1]);
    expect(shapeRings([])).toMatch(/^give shapes/);
    expect(shapeRings([{ rect: [0, 0, 0, 1] }])).toMatch(
      /^shapes\[0\] must be \{ rect/,
    );
    expect(
      shapeRings([
        {
          polygon: [
            [0, 0],
            [1, 1],
          ],
        },
      ]),
    ).toMatch(/^shapes\[0\] must be/);
    const geometry = ringsGeometry([rect!, polygon!]);
    expect([...geometry.shapePolygonOffsets]).toEqual([0, 1, 2]);
    expect([...geometry.polygonRingOffsets]).toEqual([0, 1, 2]);
    expect([...geometry.ringVertexOffsets]).toEqual([0, 4, 7]);
    expect(geometry.coords).toHaveLength(14);
  });

  it("shows shapes in the layer and coordinates of the points, from memory, and hides them", async () => {
    const { state, projectStore } = makeProject();
    const shapes = makeChatShapes(projectStore, "chat-p2r");
    const id = shapes.show({
      name: "Pixel",
      shapes: [{ rect: [0, 0, 10, 10] }],
      color: "#ff0000",
    });
    expect(id).toBe("chat-p2r-shapes-1");
    const object = state.shapes[0]!;
    expect(object.name).toBe("Pixel (chat-p2r, temporary)");
    expect(object.layer).toBe("global");
    expect(object.transform).toEqual(state.points[0]!.transform);
    expect(object.dataSource.type).toBe(chatShapesType);
    expect(object.shapeStrokeColor).toEqual({
      constant: { value: { r: 255, g: 0, b: 0 } },
    });
    expect(object.shapeFillVisibility).toEqual({ constant: { value: false } });
    // TissUUmaps loads them from memory
    const data = await chatShapesProvider.load(object.dataSource);
    expect(data.getSize()).toBe(1);
    expect([...(await data.loadGeometry()).coords]).toEqual([
      0, 0, 10, 0, 10, 10, 0, 10,
    ]);
    // like labels, filled
    const second = shapes.show({
      shapes: [{ circle: [0, 0, 5] }],
      like: "nuclei",
      fill: true,
    });
    expect(state.shapes[1]!.layer).toBe("other");
    expect(state.shapes[1]!.shapeFillVisibility).toEqual({
      constant: { value: true },
    });
    shapes.hide(id);
    expect(state.shapes.map((item) => item.id)).toEqual([second]);
    await expect(chatShapesProvider.load(object.dataSource)).rejects.toThrow(
      "they are gone",
    );
    // a shapes object of someone else is not removed
    shapes.hide("other-shapes");
    expect(state.shapes).toHaveLength(1);
    shapes.dispose();
    expect(state.shapes).toHaveLength(0);
  });

  it("says clearly what is wrong", () => {
    const { projectStore } = makeProject();
    const shapes = makeChatShapes(projectStore, "chat-p2r");
    expect(() =>
      shapes.show({ shapes: [{ rect: [0, 0, 1, 1] }], like: "x" }),
    ).toThrow(
      'chat.showShapes: no points or labels "x"; the project has "cells", "nuclei"',
    );
    expect(() =>
      shapes.show({ shapes: [{ rect: [0, 0, 1, 1] }], color: "not a color" }),
    ).toThrow("chat.showShapes: color must be");
    expect(() => shapes.show({})).toThrow("chat.showShapes: give shapes");
  });

  it("registers its data provider once", () => {
    const providers = new Map<string, unknown>();
    const registerShapesDataProvider = vi.fn(
      (type: string, provider: unknown) => providers.set(type, provider),
    );
    const appStore = {
      getState: () => ({
        shapesDataProviders: providers,
        registerShapesDataProvider,
      }),
    } as unknown as Parameters<typeof registerChatShapes>[0];
    registerChatShapes(appStore);
    registerChatShapes(appStore);
    expect(registerShapesDataProvider).toHaveBeenCalledTimes(1);
    expect(providers.get(chatShapesType)).toBe(chatShapesProvider);
  });
});
