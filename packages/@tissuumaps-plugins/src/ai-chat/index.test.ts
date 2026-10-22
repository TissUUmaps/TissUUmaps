// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createStore } from "zustand";
import { immer } from "zustand/middleware/immer";

import {
  type Color,
  type DataStoreApi,
  type GroupValueMap,
  type Labels,
  type Points,
  type ProjectStore,
  type ProjectStoreApi,
  type TableData,
  createImage,
  createLabels,
  createLayer,
  createPoints,
  createProject,
  createShapes,
} from "@tissuumaps/core";

import { aiChatPlugin } from ".";

/** The ids of the objects of the fake project */
const cellLabels = "labels/cell_labels@global";
const nucleusLabels = "labels/nucleus_labels@global";
const cellCircles = "shapes/cell_circles@global";
const transcripts = "points/transcripts@global";
const heImage = "images/he_image@global";

/** Makes `count` copies of `values` after one another */
function repeat<TValue>(values: TValue[], count: number): TValue[] {
  return Array.from(
    { length: count },
    (_, i) => values[i % values.length] as TValue,
  );
}

/** Gene names: five known ones, then GENE5, GENE6, ... */
function genes(count: number): string[] {
  return Array.from(
    { length: count },
    (_, i) => ["EPCAM", "KRT7", "CD3E", "MS4A1", "PTPRC"][i] ?? `GENE${i}`,
  );
}

/**
 * A fake table: `columns` by name, AnnData-like `groups` ("obs/") and
 * `matrices` (e.g. X, with its column names), and columns that fail to load
 */
function makeTable({
  rows,
  columns = {},
  groups = [],
  matrices = {},
  failing = {},
}: {
  rows: number;
  columns?: Record<string, unknown[] | ArrayBufferView>;
  groups?: string[];
  matrices?: Record<string, string[]>;
  failing?: Record<string, Error>;
}): TableData & { loaded: string[] } {
  const loaded: string[] = [];
  const isChild = (query: string, path: string) =>
    path.startsWith(query) && !path.slice(query.length).includes("/");
  const table = {
    loaded,
    getSize: () => rows,
    suggestColumnQueries: (query: string) => {
      const matrix = matrices[query];
      if (matrix !== undefined) {
        return Promise.resolve(
          matrix.map((name) => ({ query: `${query}[${name}]` })),
        );
      }
      return Promise.resolve([
        ...(query === ""
          ? groups.map((group) => ({ query: group, group: true }))
          : []),
        ...Object.keys(matrices)
          .filter((path) => isChild(query, path))
          .map((path) => ({ query: path, group: true })),
        ...[...Object.keys(columns), ...Object.keys(failing)]
          .filter((path) => isChild(query, path))
          .map((path) => ({ query: path })),
      ]);
    },
    loadValues: (column: string) => {
      loaded.push(column);
      const error = failing[column];
      if (error !== undefined) {
        return Promise.reject(error);
      }
      const [, matrix, gene] = /^(.*)\[(.*)\]$/.exec(column) ?? [];
      if (
        matrix !== undefined &&
        gene !== undefined &&
        matrices[matrix]?.includes(gene)
      ) {
        return Promise.resolve(
          Float32Array.from({ length: rows }, (_, i) => (i * 7) % 50),
        );
      }
      return Promise.resolve(columns[column]);
    },
  };
  return table as unknown as TableData & { loaded: string[] };
}

/** The tables of the fake project, like those of a Xenium dataset */
function makeTables(
  options: { geneNames?: boolean } = {},
): Record<string, TableData & { loaded: string[] }> {
  const { geneNames = true } = options;
  const geneList = geneNames
    ? genes(377)
    : Array.from({ length: 377 }, (_, i) => String(i));
  return {
    // an AnnData table with one row per cell
    "tables/table": makeTable({
      rows: 2000,
      groups: ["obs/", "var/", "uns/", "obsm/"],
      matrices: { X: geneList, "obsm/spatial": ["0", "1"] },
      failing: {
        "var/_index": new Error(
          geneNames
            ? 'Column "var/_index" has 377 rows, but the table has 2000'
            : "Failed to fetch",
        ),
        "obs/broken": new Error("Unsupported data type"),
      },
      columns: {
        "obs/leiden": repeat(
          Array.from({ length: 16 }, (_, i) => String(i)),
          2000,
        ),
        "obs/kmeans": Int32Array.from({ length: 2000 }, (_, i) => i % 12),
        "obs/cell_type": repeat(
          ["T cell", "B cell", "Macrophage", "Fibroblast", "Epithelial"],
          2000,
        ),
        "obs/total_counts": Float32Array.from(
          { length: 2000 },
          (_, i) => 100 + i * 1.5,
        ),
        "obs/_index": Array.from({ length: 2000 }, (_, i) => String(i)),
        "obs/cell_id": Array.from({ length: 2000 }, (_, i) => `cell_${i}`),
        "obs/region": repeat(["cell_circles"], 2000),
        "obs/control_probe_counts": Int32Array.from(
          { length: 2000 },
          (_, i) => i % 3,
        ),
      },
    }),
    "shapes/cell_circles@global": makeTable({
      rows: 1000,
      columns: {
        radius: Float32Array.from(
          { length: 1000 },
          (_, i) => 3 + (i % 50) / 10,
        ),
      },
    }),
    // too many rows: only the columns used now are read
    "points/transcripts@global": makeTable({
      rows: 6_000_000,
      columns: {
        feature_name: repeat(genes(300), 3000),
        x_location: new Float32Array(0),
        y_location: new Float32Array(0),
        qv: Float32Array.from({ length: 3000 }, (_, i) => (i % 400) / 10),
      },
    }),
  };
}

/** A project store like the application's, with the actions the plugin uses */
function makeProjectStore(): ProjectStoreApi & {
  openOtherProject: (changes?: Partial<ProjectStore>) => void;
} {
  const labels = (id: string, visibility: boolean, table: string) =>
    createLabels({
      id,
      name: id.replace(/^.*\/|@.*$/g, ""),
      visibility,
      layer: "global",
      dataSource: { type: "ome-zarr", source: `${id}.zarr`, table },
    });
  const store = createStore<ProjectStore>()(
    immer((set) => {
      const update =
        (
          kind:
            | "images"
            | "labels"
            | "points"
            | "shapes"
            | "colorMaps"
            | "visibilityMaps",
        ) =>
        (id: string, updates: object) =>
          set((draft) => {
            const object = (draft[kind] as { id: string }[]).find(
              (o) => o.id === id,
            );
            if (object === undefined) {
              throw new Error(`${kind} "${id}" does not exist`);
            }
            Object.assign(object, updates);
          });
      const state = {
        ...createProject({ name: "xenium" }),
        source: null,
        instanceId: "project-1",
        layers: [createLayer({ id: "global", name: "global" })],
        images: [
          createImage({
            id: heImage,
            name: "he_image",
            layer: "global",
            dataSource: { type: "ome-zarr", source: "he.zarr" },
          }),
          createImage({
            id: "images/morphology_focus@global",
            name: "morphology_focus",
            visibility: false,
            layer: "global",
            dataSource: { type: "ome-zarr", source: "focus.zarr" },
          }),
        ],
        labels: [
          labels(nucleusLabels, false, "tables/nucleus_labels_table"),
          labels(cellLabels, false, "tables/table"),
        ],
        points: [
          // colored by a column of another table, like in the real Xenium project
          createPoints({
            id: cellCircles,
            name: "cell_circles",
            layer: "global",
            dataSource: {
              type: "spatialdata",
              table: "shapes/cell_circles@global",
            },
            pointSize: { from: { column: "radius" } },
            pointColor: {
              groupBy: {
                table: "tables/table",
                column: "obs/cell_type",
                map: "map-1",
              },
            },
          }),
          createPoints({
            id: transcripts,
            name: "transcripts",
            visibility: false,
            layer: "global",
            dataSource: {
              type: "table",
              table: "points/transcripts@global",
              x: "x_location",
              y: "y_location",
            } as Points["dataSource"],
            pointColor: {
              groupBy: {
                column: "feature_name",
                map: undefined,
                palette: "distinct",
              },
            },
          }),
        ],
        shapes: [
          createShapes({
            id: "shapes/cell_boundaries@global",
            name: "cell_boundaries",
            visibility: false,
            layer: "global",
            dataSource: { type: "geoparquet", source: "cells.parquet" },
          }),
        ],
        tables: [{ id: "tables/table", name: "table" }],
        colorMaps: [
          {
            id: "map-1",
            name: "Cell types",
            values: { "T cell": { r: 255, g: 0, b: 0 } },
          },
        ],
        visibilityMaps: [],
        updateImage: update("images"),
        updateLabels: update("labels"),
        updatePoints: update("points"),
        updateShapes: update("shapes"),
        updateColorMap: update("colorMaps"),
        updateVisibilityMap: update("visibilityMaps"),
        addColorMap: (map: GroupValueMap<Color>) =>
          set((draft) => void draft.colorMaps.push(map)),
        addVisibilityMap: (map: GroupValueMap<boolean>) =>
          set((draft) => void draft.visibilityMaps.push(map)),
      };
      return state as unknown as ProjectStore;
    }),
  );
  return Object.assign(store, {
    openOtherProject: (changes: Partial<ProjectStore> = {}) =>
      store.setState(
        (draft) =>
          void Object.assign(draft, changes, { instanceId: "project-2" }),
      ),
  });
}

/** A data store with the given loaded tables */
function makeDataStore(tables: Record<string, TableData>): DataStoreApi {
  const tableDataRefs = new Map(
    Object.entries(tables).map(([id, data]) => [
      id,
      { status: "loaded", data, promise: Promise.resolve(data) },
    ]),
  );
  return createStore(() => ({ tableDataRefs })) as unknown as DataStoreApi;
}

/** What the fake Ollama does, and what it was asked */
const ollama = {
  running: true,
  blocked: false,
  installed: ["gemma4:e2b"],
  /** The next answers of the model, in order */
  answers: [] as (
    | { content?: string; toolCalls?: [string, Record<string, unknown>][] }
    | { error: string }
  )[],
  /** The bodies of the chat requests */
  requests: [] as {
    messages: { role: string; content: string }[];
    options: { num_ctx: number };
  }[],
};

/** An answer that arrives in small pieces, cutting lines in half like a network can */
function streamOf(text: string): Response {
  const pieces = text.match(/.{1,25}/gs) ?? [];
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const piece of pieces) {
          controller.enqueue(new TextEncoder().encode(piece));
        }
        controller.close();
      },
    }),
  );
}

/** The fake Ollama server */
function fakeFetch(
  url: string | URL,
  options?: RequestInit,
): Promise<Response> {
  if (!ollama.running) {
    return Promise.reject(new TypeError("Failed to fetch"));
  }
  if (options?.mode === "no-cors") {
    return Promise.resolve(new Response("Ollama is running"));
  }
  if (ollama.blocked) {
    return Promise.reject(new TypeError("Failed to fetch"));
  }
  const path = new URL(url).pathname;
  if (path === "/api/tags") {
    return Promise.resolve(
      Response.json({ models: ollama.installed.map((name) => ({ name })) }),
    );
  }
  if (path === "/api/chat") {
    ollama.requests.push(
      JSON.parse(options?.body as string) as (typeof ollama.requests)[number],
    );
    const answer = ollama.answers.shift() ?? { content: "ok" };
    if ("error" in answer) {
      return Promise.resolve(streamOf(JSON.stringify(answer)));
    }
    const lines = (answer.content?.match(/.{1,6}/gs) ?? []).map((piece) => ({
      message: { role: "assistant", content: piece },
    }));
    if (answer.toolCalls !== undefined) {
      lines.push({
        message: {
          role: "assistant",
          content: "",
          tool_calls: answer.toolCalls.map(([name, args]) => ({
            function: { name, arguments: args },
          })),
        },
      } as (typeof lines)[number]);
    }
    lines.push({ message: { role: "assistant", content: "" } });
    return Promise.resolve(
      streamOf(lines.map((line) => JSON.stringify(line)).join("\n")),
    );
  }
  if (path === "/api/pull") {
    ollama.installed.push("gemma4:e2b");
    return Promise.resolve(
      streamOf(
        [
          '{"status":"pulling manifest"}',
          '{"total":1000,"completed":420}',
          '{"status":"success"}',
        ].join("\n"),
      ),
    );
  }
  return Promise.reject(new Error(`unexpected request to ${path}`));
}

/** The model calls these tools, then answers "ok" */
function modelCalls(...toolCalls: [string, Record<string, unknown>][]): void {
  ollama.answers = [{ toolCalls }, { content: "ok" }];
}

/** Opens the chat panel and gives helpers to use it like a researcher would */
async function openPanel(
  projectStore = makeProjectStore(),
  dataStore = makeDataStore(makeTables()),
) {
  const container = document.createElement("div");
  const unmount = aiChatPlugin.mount?.(container, {
    projectStore,
    dataStore,
    appStore: {} as never,
    settingsStore: {} as never,
  });
  const button = (text: string) =>
    [...container.querySelectorAll("button")].find(
      (b) => b.textContent === text,
    ) as HTMLButtonElement;
  const status = container.querySelector("span") as HTMLSpanElement;
  const select = container.querySelector("select") as HTMLSelectElement;
  const history = container.children[2] as HTMLElement;
  const examplesRow = container.children[3] as HTMLElement;
  await vi.waitFor(() =>
    expect(status.textContent).not.toBe("Ollama: checking..."),
  );
  // the tables are read
  await new Promise((resolve) => setTimeout(resolve, 20));
  const messages = () =>
    [...history.children].map((message) => ({
      who: message.children[0]?.textContent ?? "",
      text: message.children[1]?.textContent ?? "",
      element: message.children[1] as HTMLElement,
    }));
  return {
    projectStore,
    container,
    unmount,
    status: () => status.textContent,
    model: () => select.value,
    chooseModel: (name: string) => {
      select.value = name;
      select.dispatchEvent(new Event("change"));
    },
    button,
    messages,
    texts: (who: string) =>
      messages()
        .filter((m) => m.who === who)
        .map((m) => m.text),
    examples: () =>
      [...examplesRow.querySelectorAll("button")].map((b) => b.textContent),
    /** Clicks a button and waits until the model has answered */
    click: async (target: HTMLButtonElement) => {
      target.click();
      await vi.waitFor(() => expect(button("Send").textContent).toBe("Send"));
    },
    /** Sends a request and waits until the model has answered */
    say: async (text: string) => {
      (container.querySelector("textarea") as HTMLTextAreaElement).value = text;
      button("Send").click();
      await vi.waitFor(() => expect(button("Send").textContent).toBe("Send"));
    },
    /** Clicks "Show context" and returns the text it shows */
    context: () => {
      button("Show context").click();
      return messages().at(-1)?.text ?? "";
    },
  };
}

/** One object of the fake project, by id */
function objectOf(store: ProjectStoreApi, id: string): Labels | Points {
  const { labels, points } = store.getState();
  return [...labels, ...points].find((o) => o.id === id) as Labels | Points;
}

beforeEach(() => {
  Object.assign(ollama, {
    running: true,
    blocked: false,
    installed: ["gemma4:e2b"],
    answers: [],
    requests: [],
  });
  vi.stubGlobal("fetch", fakeFetch);
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  window.localStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("aiChatPlugin", () => {
  describe("showing and hiding", () => {
    it("shows hidden objects and tells the model which ones were already visible", async () => {
      const panel = await openPanel();
      modelCalls([
        "set_visibility",
        { ids: [transcripts, cellCircles], visible: true },
      ]);
      await panel.say("Show the transcripts and the cell circles.");
      expect(objectOf(panel.projectStore, transcripts).visibility).toBe(true);
      expect(panel.texts("Tool")[0]).toContain(
        "Done: now visible: transcripts. Already visible before: cell_circles.",
      );
    });

    it("says nothing changed when the object is already in that state", async () => {
      const panel = await openPanel();
      modelCalls(["set_visibility", { ids: [transcripts], visible: false }]);
      await panel.say("Hide the transcripts.");
      expect(panel.texts("Tool")[0]).toContain(
        "Nothing changed: transcripts already hidden.",
      );
    });

    it("shows only the objects of a category, whatever their ids look like", async () => {
      const panel = await openPanel();
      modelCalls(["show_only", { category: "point" }]);
      await panel.say("Show only the points.");
      const { images, labels, points, shapes } = panel.projectStore.getState();
      expect([...images, ...labels, ...shapes].some((o) => o.visibility)).toBe(
        false,
      );
      expect(points.every((o) => o.visibility)).toBe(true);
    });

    it("hides everything with show_only without ids", async () => {
      const panel = await openPanel();
      modelCalls(["show_only", {}]);
      await panel.say("Hide everything.");
      const { images, labels, points, shapes } = panel.projectStore.getState();
      expect(
        [...images, ...labels, ...points, ...shapes].some((o) => o.visibility),
      ).toBe(false);
    });

    it("changes nothing when one id is wrong, or is a table", async () => {
      const panel = await openPanel();
      modelCalls([
        "set_visibility",
        { ids: [transcripts, "images/nope"], visible: true },
      ]);
      await panel.say("Show them.");
      modelCalls(["show_only", { ids: ["tables/table"] }]);
      await panel.say("Show only the table.");
      expect(objectOf(panel.projectStore, transcripts).visibility).toBe(false);
      expect(panel.texts("Tool")[0]).toContain(
        'there is no object with id "images/nope"',
      );
      expect(panel.texts("Tool")[1]).toContain('"tables/table" is a table');
    });

    it("stops when the model repeats a call that failed", async () => {
      const panel = await openPanel();
      ollama.answers = [
        { toolCalls: [["show_only", { ids: ["images/nope"] }]] },
        { toolCalls: [["show_only", { ids: ["images/nope"] }]] },
        { content: "never reached" },
      ];
      await panel.say("Show only nope.");
      expect(panel.texts("Error")).toEqual([
        "Stopped: the model repeated a tool call that had already failed.",
      ]);
    });
  });

  describe("opacity and point size", () => {
    it("sets the opacity and shows a hidden object, since the change could not be seen", async () => {
      const panel = await openPanel();
      modelCalls(["set_opacity", { category: "images", opacity: 0.5 }]);
      await panel.say("Make the images 50% transparent.");
      const { images } = panel.projectStore.getState();
      expect(images.map((image) => [image.opacity, image.visibility])).toEqual([
        [0.5, true],
        [0.5, true],
      ]);
    });

    it("rejects an opacity in percent", async () => {
      const panel = await openPanel();
      modelCalls(["set_opacity", { ids: [heImage], opacity: 50 }]);
      await panel.say("Make the H&E image 50% transparent.");
      expect(panel.texts("Tool")[0]).toContain(
        "Error: opacity must be a number from 0 to 1",
      );
    });

    it("changes the point size factor by the percentage, also for sizes from a column", async () => {
      const panel = await openPanel();
      ollama.answers = [
        {
          toolCalls: [
            ["set_point_size", { ids: [transcripts], change_percent: 25 }],
          ],
        },
        {
          toolCalls: [
            ["set_point_size", { ids: [transcripts], change_percent: -10 }],
          ],
        },
        {
          toolCalls: [
            ["set_point_size", { ids: [cellCircles], change_percent: 100 }],
          ],
        },
        { content: "ok" },
      ];
      await panel.say(
        "25% bigger, then 10% smaller, and the circles twice as big.",
      );
      expect(
        (objectOf(panel.projectStore, transcripts) as Points).pointSizeFactor,
      ).toBeCloseTo(1.125);
      expect(
        (objectOf(panel.projectStore, cellCircles) as Points).pointSizeFactor,
      ).toBe(2);
    });

    it("rejects sizes that would make the points disappear, and non-points", async () => {
      const panel = await openPanel();
      ollama.answers = [
        {
          toolCalls: [
            ["set_point_size", { ids: [transcripts], change_percent: -100 }],
          ],
        },
        {
          toolCalls: [
            ["set_point_size", { ids: [heImage], change_percent: 50 }],
          ],
        },
        { content: "ok" },
      ];
      await panel.say("Make them disappear.");
      expect(panel.texts("Tool")[0]).toContain(
        "Error: change_percent must be a number above -100",
      );
      expect(panel.texts("Tool")[1]).toContain("Error:");
    });
  });

  describe("colors and filters", () => {
    it("colors by the groups of a column, with the usual palette", async () => {
      const panel = await openPanel();
      modelCalls(["color_by_column", { ids: [cellLabels], column: "leiden" }]);
      await panel.say("Color the cell labels by leiden.");
      const labels = objectOf(panel.projectStore, cellLabels) as Labels;
      expect(labels.labelColor).toEqual({
        groupBy: { column: "obs/leiden", palette: "distinct" },
      });
      expect(labels.visibility).toBe(true);
    });

    it("colors by a gene as numbers, and by numbers with a palette and a range", async () => {
      const panel = await openPanel();
      modelCalls(["color_by_column", { ids: [cellLabels], column: "EPCAM" }]);
      await panel.say("Color the cells by EPCAM.");
      expect(
        (objectOf(panel.projectStore, cellLabels) as Labels).labelColor,
      ).toEqual({
        from: { column: "X[EPCAM]", palette: "viridis" },
      });
      modelCalls([
        "color_by_column",
        {
          ids: [cellLabels],
          column: "total_counts",
          palette: "Batlow",
          range: [0, 500],
        },
      ]);
      await panel.say(
        "Color the cells by total_counts with batlow from 0 to 500.",
      );
      expect(
        (objectOf(panel.projectStore, cellLabels) as Labels).labelColor,
      ).toEqual({
        from: {
          column: "obs/total_counts",
          palette: "batlow",
          range: [0, 500],
        },
      });
    });

    it("rejects a palette for numbers on groups, and a range on groups", async () => {
      const panel = await openPanel();
      ollama.answers = [
        {
          toolCalls: [
            [
              "color_by_column",
              { ids: [cellLabels], column: "leiden", palette: "viridis" },
            ],
          ],
        },
        {
          toolCalls: [
            [
              "color_by_column",
              { ids: [cellLabels], column: "leiden", range: [0, 5] },
            ],
          ],
        },
        { content: "ok" },
      ];
      await panel.say("Color by leiden.");
      expect(panel.texts("Tool")[0]).toContain(
        '"viridis" is not a palette for groups',
      );
      expect(panel.texts("Tool")[1]).toContain(
        "has groups, not numbers, so it takes no range",
      );
    });

    it("finds a column in the table of the object's colors, which is another table", async () => {
      const panel = await openPanel();
      modelCalls(["color_by_column", { ids: [cellCircles], column: "leiden" }]);
      await panel.say("Color the cell circles by leiden.");
      expect(
        (objectOf(panel.projectStore, cellCircles) as Points).pointColor,
      ).toEqual({
        groupBy: {
          table: "tables/table",
          column: "obs/leiden",
          palette: "distinct",
        },
      });
    });

    it("skips the objects that do not have the column, and says why", async () => {
      const panel = await openPanel();
      modelCalls(["color_by_column", { category: "labels", column: "leiden" }]);
      await panel.say("Color the labels by leiden.");
      expect(
        (objectOf(panel.projectStore, cellLabels) as Labels).labelColor,
      ).toEqual({
        groupBy: { column: "obs/leiden", palette: "distinct" },
      });
      expect(panel.texts("Tool")[0]).toContain(
        "Not changed: nucleus_labels (the table of nucleus_labels is not loaded yet.",
      );
    });

    it("gives chosen colors to groups and light grey to the others, and keeps them for the next group", async () => {
      const panel = await openPanel();
      ollama.answers = [
        {
          toolCalls: [
            ["color_by_column", { ids: [cellLabels], column: "leiden" }],
            [
              "set_group_colors",
              { ids: [cellLabels], colors: { "cluster 3": "red" } },
            ],
          ],
        },
        { content: "ok" },
      ];
      await panel.say("Cluster 3 in red.");
      modelCalls([
        "set_group_colors",
        { ids: [cellLabels], colors: { 5: "Blue", others: "grey" } },
      ]);
      await panel.say("And cluster 5 in blue, the others grey.");
      const map = panel.projectStore
        .getState()
        .colorMaps.find((m) => m.id === `chat-colors/labels/${cellLabels}`);
      expect(map?.values).toEqual({
        3: { r: 255, g: 0, b: 0 },
        5: { r: 0, g: 90, b: 255 },
      });
      expect(map?.default).toEqual({ r: 128, g: 128, b: 128 });
      expect(
        (objectOf(panel.projectStore, cellLabels) as Labels).labelColor,
      ).toEqual({
        groupBy: {
          column: "obs/leiden",
          map: `chat-colors/labels/${cellLabels}`,
        },
      });
    });

    it("explains what to do with unknown colors, groups, and colors without groups", async () => {
      const panel = await openPanel();
      ollama.answers = [
        {
          toolCalls: [
            [
              "set_group_colors",
              { ids: [cellLabels], colors: { 3: "rosso" }, column: "leiden" },
            ],
          ],
        },
        {
          toolCalls: [
            [
              "set_group_colors",
              { ids: [cellLabels], colors: { 99: "red" }, column: "leiden" },
            ],
          ],
        },
        {
          toolCalls: [
            [
              "set_group_colors",
              { ids: [cellCircles], colors: {}, others: "green" },
            ],
          ],
        },
        { content: "ok" },
      ];
      await panel.say("Colors.");
      const [color, group, empty] = panel.texts("Tool");
      expect(color).toContain('unknown color "rosso"');
      expect(group).toContain(
        'obs/leiden has no group "99". Its groups: 0, 1, 2',
      );
      expect(empty).toContain(
        "To give one color to all the items, use set_color.",
      );
    });

    it("gives one color to all items", async () => {
      const panel = await openPanel();
      modelCalls(["set_color", { ids: [cellCircles], color: "light gray" }]);
      await panel.say("Color the cell circles light gray.");
      expect(
        (objectOf(panel.projectStore, cellCircles) as Points).pointColor,
      ).toEqual({
        constant: { value: { r: 211, g: 211, b: 211 } },
      });
    });

    it("shows only some groups, without column the groups of the colors, and shows all again", async () => {
      const panel = await openPanel();
      modelCalls([
        "show_only_groups",
        { ids: [transcripts], groups: ["epcam", "KRT7"] },
      ]);
      await panel.say("Show only the EPCAM and KRT7 transcripts.");
      const filter = (objectOf(panel.projectStore, transcripts) as Points)
        .pointVisibility;
      expect(filter).toEqual({
        groupBy: {
          column: "feature_name",
          map: `chat-filter/points/${transcripts}`,
        },
      });
      expect(panel.projectStore.getState().visibilityMaps[0]).toMatchObject({
        values: { EPCAM: true, KRT7: true },
        default: false,
      });
      modelCalls(["show_all_groups", { category: "points" }]);
      await panel.say("Show all the transcripts again.");
      expect(
        (objectOf(panel.projectStore, transcripts) as Points).pointVisibility,
      ).toEqual({ constant: { value: true } });
      expect(panel.texts("Tool")[1]).toContain(
        "No filter, left as they were: cell_circles.",
      );
    });

    it("does not filter by numbers", async () => {
      const panel = await openPanel();
      modelCalls([
        "show_only_groups",
        { ids: [cellLabels], groups: ["100"], column: "total_counts" },
      ]);
      await panel.say("Show the cells with 100 counts.");
      expect(panel.texts("Tool")[0]).toContain(
        'filters on numbers (e.g. "area > 50") are not possible',
      );
    });

    it("tells the user when another visible object shows the same items", async () => {
      // the same labels added a second time, like "cells" and "cells-epcam"
      const projectStore = makeProjectStore();
      const original = projectStore.getState().labels[1] as Labels;
      const copy = {
        ...structuredClone(original),
        id: "labels/cell_labels_epcam@global",
        name: "cell_labels_epcam",
        visibility: true,
      };
      projectStore.setState((draft) => void draft.labels.push(copy));
      const panel = await openPanel(projectStore);
      modelCalls([
        "show_only_groups",
        { ids: [cellLabels], groups: ["1", "3"], column: "leiden" },
      ]);
      await panel.say("Show only clusters 1 and 3 of cell_labels.");
      expect(panel.texts("Tool")[0]).toMatch(
        /^show_only_groups.*→ Done\. Tell the user: cell_labels_epcam is also visible/s,
      );
      expect(panel.texts("Note")).toEqual([
        "cell_labels_epcam is also visible and shows the same items as cell_labels, in the same place: it can cover this change. Hide cell_labels_epcam to see it.",
      ]);
    });
  });

  describe("undo", () => {
    it("takes back the last request, and then the one before", async () => {
      const panel = await openPanel();
      modelCalls(["set_opacity", { ids: [heImage], opacity: 0.5 }]);
      await panel.say("Make the H&E image 50% transparent.");
      modelCalls(
        ["color_by_column", { ids: [cellLabels], column: "leiden" }],
        ["set_group_colors", { ids: [cellLabels], colors: { 3: "red" } }],
      );
      await panel.say("Cluster 3 in red.");
      panel.button("Undo").click();
      const labels = objectOf(panel.projectStore, cellLabels) as Labels;
      expect(labels.labelColor).toEqual({ random: { palette: "distinct" } });
      expect(labels.visibility).toBe(false);
      panel.button("Undo").click();
      expect(panel.projectStore.getState().images[0]?.opacity).toBe(1);
      expect(panel.button("Undo").disabled).toBe(true);
      // the model is told, so that it does not trust the old tool results
      await panel.say("And now?");
      expect(ollama.requests.at(-1)?.messages.at(-2)?.content).toContain(
        'the user clicked "Undo"',
      );
    });

    it("notes values that were changed by hand, and undoes all after asking", async () => {
      const panel = await openPanel();
      modelCalls(["set_opacity", { ids: [heImage], opacity: 0.5 }]);
      await panel.say("Make the H&E image 50% transparent.");
      modelCalls(["set_visibility", { ids: [transcripts], visible: true }]);
      await panel.say("Show the transcripts.");
      panel.projectStore.getState().updateImage(heImage, { opacity: 0.8 });
      const confirm = vi
        .spyOn(window, "confirm")
        .mockReturnValueOnce(false)
        .mockReturnValueOnce(true);
      panel.button("Undo all").click();
      expect(objectOf(panel.projectStore, transcripts).visibility).toBe(true);
      panel.button("Undo all").click();
      expect(confirm).toHaveBeenCalledTimes(2);
      expect(objectOf(panel.projectStore, transcripts).visibility).toBe(false);
      expect(panel.projectStore.getState().images[0]?.opacity).toBe(1);
      expect(panel.texts("Undo all")[0]).toContain(
        "he_image opacity 0.8 → 1 (it had been changed by hand after the chat changed it)",
      );
    });
  });

  describe("project summary", () => {
    it("lists the useful columns of the objects' tables, and counts the others", async () => {
      const panel = await openPanel();
      const context = panel.context();
      expect(context).toContain(
        "obs/leiden (16 groups: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)",
      );
      expect(context).toContain(
        "obs/kmeans (12 groups: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)",
      );
      expect(context).toContain("obs/total_counts (numbers, 100 to 3098.5)");
      expect(context).toContain(
        "X (377 genes, e.g. X[EPCAM], X[KRT7], X[CD3E])",
      );
      // obs/_index, obs/cell_id, obs/region, obs/control_probe_counts, obs/broken, obsm/spatial
      expect(context).toContain("(+ 6 other columns)");
      expect(context).toContain(
        "columns you can use (another table, tables/table, with rows for the same items",
      );
      expect(context).toContain(
        'cluster columns: obs/leiden, obs/kmeans. If the user just says "clusters", do not call a tool: ask which one.',
      );
      expect(context).toContain(
        "columns: the table tables/nucleus_labels_table is not loaded yet",
      );
    });

    it("reads only the used columns of very big tables, and the new ones after a change", async () => {
      const tables = makeTables();
      const panel = await openPanel(makeProjectStore(), makeDataStore(tables));
      expect(tables["points/transcripts@global"]?.loaded).toEqual([
        "feature_name",
      ]);
      expect(tables["tables/table"]?.loaded).not.toContain("var/_index");
      modelCalls(["color_by_column", { ids: [transcripts], column: "qv" }]);
      await panel.say("Color the transcripts by qv.");
      expect(panel.context()).toContain("qv (numbers, 0 to 39.9)");
    });

    it("says when the gene names are not available, and the error explains the format", async () => {
      const panel = await openPanel(
        makeProjectStore(),
        makeDataStore(makeTables({ geneNames: false })),
      );
      expect(panel.context()).toContain(
        "X (377 genes, gene names not available (var/_index: Failed to fetch): only by number, X[0] to X[376])",
      );
      ollama.answers = [
        {
          toolCalls: [["color_by_column", { ids: [cellLabels], column: "X" }]],
        },
        {
          toolCalls: [
            ["color_by_column", { ids: [cellLabels], column: "EPCAM" }],
          ],
        },
        { content: "ok" },
      ];
      await panel.say("Color the cells by EPCAM.");
      const [matrix, gene] = panel.texts("Tool");
      expect(matrix).toContain(
        "give X[gene name], so a gene by number, X[0] to X[376]: the gene names are not available here",
      );
      expect(gene).toContain(
        "the gene names are not available in this project",
      );
    });

    it("asks Ollama for a context window that fits the summary", async () => {
      const panel = await openPanel();
      await panel.say("Hello");
      expect(ollama.requests[0]?.options.num_ctx).toBe(16384);
      expect(ollama.requests[0]?.messages[0]?.role).toBe("system");
    });
  });

  describe("examples", () => {
    it("proposes only requests for objects and columns of the open project, which work", async () => {
      const panel = await openPanel();
      expect(panel.examples()).toEqual([
        "Color the cell_labels by leiden",
        "Show only CD3E and EPCAM in transcripts",
        "Color the cell_labels by EPCAM expression",
        "What is shown now?",
      ]);
      modelCalls([
        "show_only_groups",
        { ids: [transcripts], groups: ["CD3E", "EPCAM"] },
      ]);
      await panel.click(
        panel.button("Show only CD3E and EPCAM in transcripts"),
      );
      expect(panel.texts("Tool")[0]).toContain(
        "Done: now shown transcripts: only CD3E, EPCAM of feature_name",
      );
      expect(panel.examples()).toEqual([]);
    });

    it("changes the examples at once when another project is opened", async () => {
      const panel = await openPanel();
      panel.projectStore.openOtherProject({
        name: "xenium-lung_h5ad_cells",
        points: [],
      });
      expect(panel.examples()).not.toContain(
        "Show only CD3E and EPCAM in transcripts",
      );
    });

    it("proposes no colors by gene without gene names", async () => {
      const panel = await openPanel(
        makeProjectStore(),
        makeDataStore(makeTables({ geneNames: false })),
      );
      expect(
        panel.examples().some((example) => example.includes("expression")),
      ).toBe(false);
    });
  });

  describe("chat", () => {
    it("shows the answer while it arrives", async () => {
      const panel = await openPanel();
      ollama.answers = [
        { content: "The **H&E image** and the cell circles are visible." },
      ];
      await panel.say("What is visible?");
      const answer = panel.messages().find((m) => m.who === "gemma4:e2b");
      expect(answer?.text).toBe(
        "The H&E image and the cell circles are visible.",
      );
      expect(answer?.element.querySelector("b")?.textContent).toBe("H&E image");
    });

    it("never turns the answer into HTML", async () => {
      const panel = await openPanel();
      ollama.answers = [
        {
          content:
            '<script>alert("hacked")</script> <img src=x onerror=alert(1)>',
        },
      ];
      await panel.say("Hi");
      expect(panel.container.querySelector("script, img")).toBeNull();
      expect(panel.texts("gemma4:e2b")[0]).toContain("<script>");
    });

    it("starts a new chat after asking, and keeps the undo list", async () => {
      const panel = await openPanel();
      modelCalls(["set_visibility", { ids: [transcripts], visible: true }]);
      await panel.say("Show the transcripts.");
      vi.spyOn(window, "confirm").mockReturnValue(true);
      panel.button("New chat").click();
      expect(panel.messages()).toEqual([]);
      expect(panel.button("Undo").disabled).toBe(false);
      await panel.say("Hello again");
      expect(ollama.requests.at(-1)?.messages.slice(1)).toEqual([
        { role: "user", content: "Hello again" },
      ]);
    });

    it("turns Ollama's errors into instructions", async () => {
      const panel = await openPanel();
      ollama.answers = [
        { error: "unauthorized" },
        { error: "model 'gemma4:e2b' not found" },
      ];
      await panel.say("Hello");
      await panel.say("Hello");
      expect(panel.texts("Error")).toEqual([
        "Online models need you to be signed in to Ollama. Open the Ollama app, go to Settings and sign in (a free account is enough for gemma4:cloud). Then try again.",
        'This model is not downloaded yet. Click "Download" next to the model menu.',
      ]);
    });

    it("remembers the chosen model for the next session", async () => {
      const first = await openPanel();
      expect(first.model()).toBe("gemma4:e2b");
      first.chooseModel("gemma4:cloud");
      const next = await openPanel();
      expect(next.model()).toBe("gemma4:cloud");
    });

    it("downloads a missing model and shows the progress", async () => {
      ollama.installed = [];
      const panel = await openPanel();
      expect(panel.status()).toBe(
        "Ollama: connected. gemma4:e2b is not downloaded yet.",
      );
      panel.button("Download").click();
      await vi.waitFor(() =>
        expect(panel.status()).toBe("Ollama: connected. gemma4:e2b is ready."),
      );
    });
  });

  describe("when Ollama does not answer", () => {
    it("says that Ollama is not running, on a page from this computer", async () => {
      ollama.running = false;
      const panel = await openPanel();
      expect(panel.status()).toMatch(/^Ollama is not running\./);
    });

    it("says that Ollama blocks a website, and how to allow it", async () => {
      ollama.blocked = true;
      vi.stubGlobal("location", {
        hostname: "tissuumaps.github.io",
        origin: "https://tissuumaps.github.io",
      });
      const panel = await openPanel();
      expect(panel.status()).toContain(
        "Ollama is running, but it blocks this website (https://tissuumaps.github.io): by default it only accepts pages opened from this computer. To allow it, set OLLAMA_ORIGINS to https://tissuumaps.github.io",
      );
    });
  });
});
