// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createStore } from "zustand";
import { immer } from "zustand/middleware/immer";

import {
  type AppStoreApi,
  type Color,
  type DataStoreApi,
  type Image,
  type ImageData,
  type Labels,
  type Points,
  type ProjectStore,
  type ProjectStoreApi,
  type SettingsStore,
  type SettingsStoreApi,
  type TableData,
  createImage,
  createLabels,
  createLayer,
  createPoints,
  createProject,
  createShapes,
} from "@tissuumaps/core";

import { aiChatPlugin } from ".";
import { thinkingWords } from "./persona";
import { storeTypes } from "./store-types.generated";

/** The ids of the objects of the fake project */
const cellLabels = "labels/cell_labels@global";
const nucleusLabels = "labels/nucleus_labels@global";
const cellCircles = "shapes/cell_circles@global";
const transcripts = "points/transcripts@global";
const heImage = "images/he_image@global";
const morphologyFocus = "images/morphology_focus@global";

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
  type ListKind =
    | "layers"
    | "images"
    | "labels"
    | "points"
    | "shapes"
    | "markerMaps"
    | "sizeMaps"
    | "colorMaps"
    | "visibilityMaps"
    | "opacityMaps";
  const store = createStore<ProjectStore>()(
    immer((set) => {
      const find = (draft: ProjectStore, kind: ListKind, id: string) => {
        const list = draft[kind] as { id: string }[];
        const index = list.findIndex((o) => o.id === id);
        if (index === -1) {
          throw new Error(`${kind} "${id}" does not exist`);
        }
        return { list, index };
      };
      const update = (kind: ListKind) => (id: string, updates: object) =>
        set((draft) => {
          const { list, index } = find(draft, kind, id);
          Object.assign(list[index] as object, updates);
        });
      const add = (kind: ListKind) => (entry: { id: string }) =>
        set((draft) => void (draft[kind] as { id: string }[]).push(entry));
      const remove = (kind: ListKind) => (id: string) =>
        set((draft) => {
          const { list, index } = find(draft, kind, id);
          list.splice(index, 1);
        });
      const move = (kind: ListKind) => (id: string, newIndex: number) =>
        set((draft) => {
          const { list, index } = find(draft, kind, id);
          const [entry] = list.splice(index, 1);
          list.splice(newIndex, 0, entry as { id: string });
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
            id: morphologyFocus,
            name: "morphology_focus",
            visibility: false,
            layer: "global",
            dataSource: { type: "ome-zarr", source: "focus.zarr" },
            channels: ["DAPI", "ATP1A1", "18S", "AlphaSMA"].map((name) => ({
              name,
              color: { r: 255, g: 255, b: 255 },
            })),
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
        markerMaps: [],
        sizeMaps: [],
        visibilityMaps: [],
        opacityMaps: [],
        setName: (name: string) => set({ name }),
        setViewerBackgroundColor: (viewerBackgroundColor: Color) =>
          set({ viewerBackgroundColor }),
        updateLayer: update("layers"),
        updateImage: update("images"),
        updateLabels: update("labels"),
        updatePoints: update("points"),
        updateShapes: update("shapes"),
        updateMarkerMap: update("markerMaps"),
        updateSizeMap: update("sizeMaps"),
        updateColorMap: update("colorMaps"),
        updateVisibilityMap: update("visibilityMaps"),
        updateOpacityMap: update("opacityMaps"),
        addMarkerMap: add("markerMaps"),
        addSizeMap: add("sizeMaps"),
        addColorMap: add("colorMaps"),
        addVisibilityMap: add("visibilityMaps"),
        addOpacityMap: add("opacityMaps"),
        deleteMarkerMap: remove("markerMaps"),
        deleteSizeMap: remove("sizeMaps"),
        deleteColorMap: remove("colorMaps"),
        deleteVisibilityMap: remove("visibilityMaps"),
        deleteOpacityMap: remove("opacityMaps"),
        moveLayer: move("layers"),
        moveImage: move("images"),
        moveLabels: move("labels"),
        movePoints: move("points"),
        moveShapes: move("shapes"),
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

/** A data store with the given loaded tables, and image data */
function makeDataStore(
  tables: Record<string, TableData>,
  images: Record<string, ImageData> = {},
): DataStoreApi {
  const loaded = <T>(data: Record<string, T>) =>
    new Map(
      Object.entries(data).map(([id, entry]) => [
        id,
        { status: "loaded", data: entry, promise: Promise.resolve(entry) },
      ]),
    );
  return createStore(() => ({
    tableDataRefs: loaded(tables),
    imageDataRefs: loaded(images),
  })) as unknown as DataStoreApi;
}

/**
 * Image data with named channels and no colors of its own, like the
 * morphology image of the real Xenium project
 */
function makeImageData(names: string[]): ImageData {
  return {
    getSizeC: () => names.length,
    getChannelName: (c: number) => names[c],
    getTileSource: () => "tiles.dzi",
  } as unknown as ImageData;
}

/** What the fake Ollama does, and what it was asked */
const ollama = {
  running: true,
  blocked: false,
  installed: ["gemma4:e2b"],
  /** When set, the next chat answer waits for it, as if the model were thinking */
  thinking: undefined as Promise<void> | undefined,
  /** The next answers of the model, in order */
  answers: [] as (
    | { content?: string; toolCalls?: [string, Record<string, unknown>][] }
    | { error: string }
  )[],
  /** The bodies of the chat requests */
  requests: [] as {
    messages: { role: string; content: string }[];
    options: { num_ctx: number };
    tools?: Record<string, unknown>[];
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

/** Fake API keys, which only look like real ones */
const openAiKey = "sk-test-openai-not-a-real-key-0001";
const anthropicKey = "sk-ant-test-not-a-real-key-0002";

/** The parts of a chat request to OpenAI or Anthropic that the tests look at */
type ServiceRequestBody = {
  model: string;
  instructions?: string;
  system?: { type: string; text: string; cache_control?: unknown }[];
  store?: boolean;
  max_tokens?: number;
  tools: Record<string, unknown>[];
  input?: Record<string, unknown>[];
  messages?: { role: string; content: Record<string, unknown>[] }[];
};

/** What the fake OpenAI and Anthropic do, and what they were asked */
const services = {
  online: true,
  /** The next answers of the model, in order */
  answers: [] as {
    content?: string;
    toolCalls?: [string, Record<string, unknown>][];
  }[],
  /** An error answer for every request, instead of the model's answer */
  error: undefined as { status: number; body: unknown } | undefined,
  /** Refuse the next request that sends the model's thinking back */
  refuseThinking: false,
  /** The chat requests, also the refused ones */
  requests: [] as { url: string; headers: Headers; body: ServiceRequestBody }[],
};

/** A short hash of a text, for the fake signatures */
function hash(text: string): string {
  let value = 5381;
  for (const character of text) {
    value = (value * 33 + character.charCodeAt(0)) | 0;
  }
  return (value >>> 0).toString(16);
}

/**
 * What a thinking block of Anthropic is bound to: the system prompt, the
 * tools and the messages before it, without the thinking blocks themselves
 * (removing them is allowed) and without cache markers
 */
function binding(body: ServiceRequestBody, before: number): string {
  const messages = (body.messages ?? []).slice(0, before).map((message) => ({
    role: message.role,
    content: message.content.filter(
      (block) =>
        block.type !== "thinking" && block.type !== "redacted_thinking",
    ),
  }));
  return `sig-${hash(
    JSON.stringify(
      [body.system, body.tools, messages],
      (key, value: unknown) => (key === "cache_control" ? undefined : value),
    ),
  )}`;
}

/**
 * Why the fake Anthropic refuses a request, as Anthropic does: a thinking
 * block whose beginning of the conversation changed since it was made
 */
function anthropicRefusal(body: ServiceRequestBody): string | undefined {
  for (const [index, message] of (body.messages ?? []).entries()) {
    for (const [position, block] of message.content.entries()) {
      if (
        block.type === "thinking" &&
        (services.refuseThinking || block.signature !== binding(body, index))
      ) {
        return `messages.${index}.content.${position}: Invalid \`signature\` in \`thinking\` block. The block is bound to a different conversation. Remove the block, or set \`thinking.block_binding.prefix_mismatch_behavior\` to "drop_block". That setting requires the \`thinking-binding-controls-2026-08-01\` value in the \`anthropic-beta\` header.`;
      }
    }
  }
  return undefined;
}

/**
 * Why the fake OpenAI refuses a request, as OpenAI does: a tool call sent
 * with its ID, but without the reasoning that came with it
 */
function openAiRefusal(body: ServiceRequestBody): string | undefined {
  const input = body.input ?? [];
  for (const item of input) {
    const id = typeof item.id === "string" ? item.id : undefined;
    if (item.type === "function_call" && id !== undefined) {
      const reasoning = id.replace(/^fc_(\d+)_.*$/, "rs_$1");
      if (
        services.refuseThinking ||
        !input.some((other) => other.id === reasoning)
      ) {
        return `Item '${id}' of type 'function_call' was provided without its required 'reasoning' item: '${reasoning}'.`;
      }
    }
  }
  return undefined;
}

/** Every request made by the panel, to find where a key was sent */
const network: { url: string; headers: Headers; body: string }[] = [];

/** An answer as server-sent events, each with its type, like OpenAI and Anthropic send */
function eventStreamOf(events: { type: string }[]): Response {
  return streamOf(
    events
      .map(
        (event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`,
      )
      .join(""),
  );
}

/** The fake OpenAI Responses API, which also sends back a reasoning item */
function fakeOpenAi(): Response {
  const answer = services.answers.shift() ?? { content: "ok" };
  const request = services.requests.length - 1;
  const output: object[] = [
    {
      type: "reasoning",
      id: `rs_${request}`,
      encrypted_content: `thinking-${request}`,
    },
  ];
  if (answer.content !== undefined) {
    output.push({
      type: "message",
      id: `msg_${request}`,
      role: "assistant",
      content: [{ type: "output_text", text: answer.content }],
    });
  }
  for (const [name, args] of answer.toolCalls ?? []) {
    output.push({
      type: "function_call",
      id: `fc_${request}_${name}`,
      call_id: `call_${request}_${name}`,
      name,
      arguments: JSON.stringify(args),
    });
  }
  return eventStreamOf([
    { type: "response.created" },
    ...(answer.content?.match(/.{1,6}/gs) ?? []).map((delta) => ({
      type: "response.output_text.delta",
      delta,
    })),
    {
      type: "response.completed",
      response: {
        output,
        // OpenAI counts the cached tokens also in input_tokens
        usage: {
          input_tokens: 9000,
          input_tokens_details: { cached_tokens: 6000, cache_write_tokens: 0 },
          output_tokens: 80,
        },
      },
    } as { type: string },
  ]);
}

/**
 * The fake Anthropic Messages API, which also sends the model's thinking,
 * signed with what it is bound to
 */
function fakeAnthropic(body: ServiceRequestBody): Response {
  const answer = services.answers.shift() ?? { content: "ok" };
  const signature = binding(body, body.messages?.length ?? 0);
  const events: object[] = [
    {
      type: "message_start",
      message: {
        role: "assistant",
        content: [],
        usage: {
          input_tokens: 3000,
          cache_read_input_tokens: 6000,
          cache_creation_input_tokens: 0,
          output_tokens: 1,
        },
      },
    },
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "thinking", thinking: "" },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "thinking_delta", thinking: "Let me see." },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "signature_delta", signature },
    },
    { type: "content_block_stop", index: 0 },
  ];
  let index = 1;
  if (answer.content !== undefined) {
    events.push({
      type: "content_block_start",
      index,
      content_block: { type: "text", text: "" },
    });
    for (const text of answer.content.match(/.{1,6}/gs) ?? []) {
      events.push({
        type: "content_block_delta",
        index,
        delta: { type: "text_delta", text },
      });
    }
    events.push({ type: "content_block_stop", index });
    index++;
  }
  for (const [name, args] of answer.toolCalls ?? []) {
    events.push({
      type: "content_block_start",
      index,
      content_block: {
        type: "tool_use",
        id: `toolu_${services.requests.length - 1}_${name}`,
        name,
        input: {},
      },
    });
    // the arguments arrive in pieces of JSON
    for (const piece of JSON.stringify(args).match(/.{1,10}/gs) ?? []) {
      events.push({
        type: "content_block_delta",
        index,
        delta: { type: "input_json_delta", partial_json: piece },
      });
    }
    events.push({ type: "content_block_stop", index });
    index++;
  }
  events.push(
    {
      type: "message_delta",
      delta: { stop_reason: "end_turn" },
      // the output tokens so far, which replace the earlier number
      usage: { output_tokens: 80 },
    },
    { type: "message_stop" },
  );
  return eventStreamOf(events as { type: string }[]);
}

/** The fake OpenAI and Anthropic */
function fakeService(url: URL, options?: RequestInit): Promise<Response> {
  if (!services.online) {
    return Promise.reject(new TypeError("Failed to fetch"));
  }
  if (services.error !== undefined) {
    return Promise.resolve(
      Response.json(services.error.body, { status: services.error.status }),
    );
  }
  if (url.pathname === "/v1/models") {
    return Promise.resolve(Response.json({ data: [] }));
  }
  const body = JSON.parse(options?.body as string) as ServiceRequestBody;
  services.requests.push({
    url: url.href,
    headers: new Headers(options?.headers),
    body,
  });
  const openAi = url.host === "api.openai.com";
  const refusal = openAi ? openAiRefusal(body) : anthropicRefusal(body);
  if (refusal !== undefined) {
    services.refuseThinking = false;
    return Promise.resolve(
      Response.json(
        { error: { type: "invalid_request_error", message: refusal } },
        { status: 400 },
      ),
    );
  }
  return Promise.resolve(openAi ? fakeOpenAi() : fakeAnthropic(body));
}

/** The fake services: OpenAI, Anthropic and the Ollama server */
function fakeFetch(
  url: string | URL,
  options?: RequestInit,
): Promise<Response> {
  const address = new URL(url);
  network.push({
    url: address.href,
    headers: new Headers(options?.headers),
    body: typeof options?.body === "string" ? options.body : "",
  });
  if (["api.openai.com", "api.anthropic.com"].includes(address.host)) {
    return fakeService(address, options);
  }
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
  const { thinking } = ollama;
  if (path === "/api/chat" && thinking !== undefined) {
    ollama.thinking = undefined;
    return thinking.then(() => fakeFetch(url, options));
  }
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
    lines.push({
      message: { role: "assistant", content: "" },
      prompt_eval_count: 5000,
      eval_count: 40,
    } as (typeof lines)[number]);
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

/** An app store with the state that only changes the view */
function makeAppStore(): AppStoreApi {
  return createStore<Record<string, unknown>>()((set) => ({
    interactionMode: "pan",
    imageChannelPreview: null,
    highlightedItemGroup: null,
    activePanelId: "plugin-ai-chat",
    expandedImageIds: [],
    expandedLabelsIds: [cellLabels],
    expandedPointsIds: [],
    expandedShapesIds: [],
    expandedTableIds: [],
    setHighlightedItemGroup: (highlightedItemGroup: unknown) =>
      set({ highlightedItemGroup }),
    setExpandedImageIds: (ids: string[]) => set({ expandedImageIds: ids }),
    setExpandedLabelsIds: (ids: string[]) => set({ expandedLabelsIds: ids }),
    setExpandedPointsIds: (ids: string[]) => set({ expandedPointsIds: ids }),
    setExpandedShapesIds: (ids: string[]) => set({ expandedShapesIds: ids }),
    setExpandedTableIds: (ids: string[]) => set({ expandedTableIds: ids }),
  })) as unknown as AppStoreApi;
}

/** A settings store like the application's, with the theme */
function makeSettingsStore(dark = false): SettingsStoreApi {
  return createStore<SettingsStore>()(
    immer((set) => ({ dark, setDark: (value) => set({ dark: value }) })),
  );
}

/** Waits until the status line no longer says that something is being checked */
async function checked(status: HTMLElement): Promise<void> {
  await vi.waitFor(() => expect(status.textContent).not.toMatch(/checking/));
}

/**
 * Opens the chat panel and gives helpers to use it like a researcher would;
 * chooses the local Ollama, unless told another service, or `null` to keep
 * the one the panel starts with
 */
async function openPanel(
  projectStore = makeProjectStore(),
  dataStore = makeDataStore(makeTables()),
  {
    provider = "ollama",
    settingsStore = makeSettingsStore(),
  }: { provider?: string | null; settingsStore?: SettingsStoreApi } = {},
) {
  const container = document.createElement("div");
  const appStore = makeAppStore();
  const unmount = aiChatPlugin.mount?.(container, {
    projectStore,
    dataStore,
    appStore,
    settingsStore,
  });
  const button = (text: string) =>
    [...container.querySelectorAll("button")].find(
      (b) => b.textContent === text,
    ) as HTMLButtonElement;
  const status = container.querySelector('[role="status"]') as HTMLElement;
  const summary = container.children[1] as HTMLElement;
  const settingsBox = container.children[2] as HTMLElement;
  const input = container.querySelector("textarea") as HTMLTextAreaElement;
  const providerSelect = container.querySelector(
    'select[title="Service"]',
  ) as HTMLSelectElement;
  const select = container.querySelector(
    'select[title="Model"]',
  ) as HTMLSelectElement;
  const keyInput = container.querySelector(
    'input[type="password"]',
  ) as HTMLInputElement;
  const rememberBox = container.querySelector(
    'input[type="checkbox"]',
  ) as HTMLInputElement;
  const history = container.children[3] as HTMLElement;
  const examplesRow = container.children[4] as HTMLElement;
  const chooseProvider = async (id: string) => {
    providerSelect.value = id;
    providerSelect.dispatchEvent(new Event("change"));
    await checked(status);
  };
  await checked(status);
  if (provider !== null) {
    await chooseProvider(provider);
  }
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
    appStore,
    container,
    unmount,
    status: () => status.textContent,
    provider: () => providerSelect.value,
    providers: () =>
      [...providerSelect.options].map((option) => option.textContent),
    chooseProvider,
    model: () => select.value,
    chooseModel: (name: string) => {
      select.value = name;
      select.dispatchEvent(new Event("change"));
    },
    keyInput,
    rememberBox,
    input,
    summary,
    settingsShown: () => settingsBox.style.display !== "none",
    /** Switches TissUUmaps between the light and the dark theme */
    setDark: (dark: boolean) => settingsStore.getState().setDark(dark),
    /** Presses a key in the text box */
    press: (key: string) =>
      input.dispatchEvent(
        new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }),
      ),
    /** Pastes a key, leaves the field, and waits for the check of the key */
    pasteKey: async (key: string) => {
      keyInput.value = key;
      keyInput.dispatchEvent(new Event("input"));
      keyInput.dispatchEvent(new Event("change"));
      await checked(status);
    },
    remember: (on: boolean) => {
      rememberBox.checked = on;
      rememberBox.dispatchEvent(new Event("change"));
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
    thinking: undefined,
    answers: [],
    requests: [],
  });
  Object.assign(services, {
    online: true,
    answers: [],
    error: undefined,
    refuseThinking: false,
    requests: [],
  });
  network.length = 0;
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
  document.body.replaceChildren();
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
      const mapIds = () =>
        panel.projectStore.getState().colorMaps.map(({ id }) => id);
      expect(mapIds()).toEqual(["map-1", `chat-colors/labels/${cellLabels}`]);
      panel.button("Undo").click();
      const labels = objectOf(panel.projectStore, cellLabels) as Labels;
      expect(labels.labelColor).toEqual({ random: { palette: "distinct" } });
      expect(labels.visibility).toBe(false);
      // the map the chat made for the colors is gone too
      expect(mapIds()).toEqual(["map-1"]);
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
      // a suggestion is sent without losing what the user is typing
      panel.input.value = "Make the cells";
      await panel.click(
        panel.button("Show only CD3E and EPCAM in transcripts"),
      );
      expect(panel.texts("Tool")[0]).toContain(
        "Done: now shown transcripts: only CD3E, EPCAM of feature_name",
      );
      expect(panel.texts("You")).toEqual([
        "Show only CD3E and EPCAM in transcripts",
      ]);
      expect(panel.input.value).toBe("Make the cells");
      // and the suggestions stay
      expect(panel.examples()).toHaveLength(4);
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
      const answer = panel.messages().find((m) => m.who === "Dia");
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
      expect(panel.texts("Dia")[0]).toContain("<script>");
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

    it("remembers the chosen service and model for the next session", async () => {
      const first = await openPanel(undefined, undefined, { provider: null });
      expect(first.model()).toBe("gpt-6-luna");
      await first.chooseProvider("anthropic");
      expect(first.model()).toBe("claude-haiku-5-5");
      first.chooseModel("claude-sonnet-5-5");
      const next = await openPanel(undefined, undefined, { provider: null });
      expect(next.provider()).toBe("anthropic");
      expect(next.model()).toBe("claude-sonnet-5-5");
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

  describe("call_action, for OpenAI and Anthropic", () => {
    /** A call of an action, as the model makes it */
    const act = (
      action: string,
      args: unknown[],
      store = "projectStore",
    ): [string, Record<string, unknown>] => [
      "call_action",
      { store, action, args },
    ];
    const image = (store: ProjectStoreApi, id: string) =>
      store.getState().images.find((i) => i.id === id) as Image;

    it("colors a channel, draws points as squares and rotates an image, shows each readably, and Undo takes them back", async () => {
      const panel = await openPanel(undefined, undefined, {
        provider: "anthropic",
      });
      await panel.pasteKey(anthropicKey);
      const red = { r: 255, g: 0, b: 0 };
      for (const [text, call] of [
        [
          "Make channel 1 of morphology_focus red.",
          act("updateImage", [
            morphologyFocus,
            { channels: [{}, { color: red }, {}, {}] },
          ]),
        ],
        [
          "Show the transcripts as squares.",
          act("updatePoints", [
            transcripts,
            { pointMarker: { constant: { value: 2 } } },
          ]),
        ],
        [
          "Rotate the H&E image by 90 degrees.",
          act("updateImage", [heImage, { transform: { rotation: 90 } }]),
        ],
      ] as const) {
        services.answers = [{ toolCalls: [call] }, { content: "Done." }];
        await panel.say(text);
      }
      const { projectStore } = panel;
      // only the channel asked for, the others as they were
      expect(image(projectStore, morphologyFocus).channels).toEqual([
        { name: "DAPI", color: { r: 255, g: 255, b: 255 } },
        { name: "ATP1A1", color: red },
        { name: "18S", color: { r: 255, g: 255, b: 255 } },
        { name: "AlphaSMA", color: { r: 255, g: 255, b: 255 } },
      ]);
      expect(
        (objectOf(projectStore, transcripts) as Points).pointMarker,
      ).toEqual({ constant: { value: 2 } });
      // only the rotation, the rest of the transform as it was
      expect(image(projectStore, heImage).transform).toMatchObject({
        rotation: 90,
        scale: 1,
        flip: false,
      });
      expect(panel.texts("Tool")).toEqual([
        'updateImage(morphology_focus, {channels[1].color: {"r":255,"g":0,"b":0}})\n→ Done. Changed in morphology_focus: channels[1] (ATP1A1).color white → red. morphology_focus is hidden: show it to see the change.',
        'updatePoints(transcripts, {pointMarker: {"constant":{"value":2}}})\n→ Done. Changed in transcripts: pointMarker {"constant":{"value":6}} → {"constant":{"value":2}}. transcripts is hidden: show it to see the change.',
        "updateImage(he_image, {transform.rotation: 90})\n→ Done. Changed in he_image: transform.rotation 0 → 90.",
      ]);
      for (let i = 0; i < 3; i++) {
        panel.button("Undo").click();
      }
      expect(image(projectStore, heImage).transform.rotation).toBe(0);
      expect(
        (objectOf(projectStore, transcripts) as Points).pointMarker,
      ).not.toEqual({ constant: { value: 2 } });
      expect(image(projectStore, morphologyFocus).channels?.[1]?.color).toEqual(
        { r: 255, g: 255, b: 255 },
      );
      expect(panel.texts("Undo")).toEqual([
        "Undone: he_image transform",
        "Undone: transcripts pointMarker",
        "Undone: morphology_focus channels",
      ]);
    });

    it("refuses what is not allowed or not right, says why, and changes nothing", async () => {
      const panel = await openPanel(undefined, undefined, {
        provider: "openai",
      });
      await panel.pasteKey(openAiKey);
      const before = JSON.stringify(panel.projectStore.getState());
      const viewBefore = JSON.stringify(panel.appStore.getState());
      services.answers = [
        {
          toolCalls: [
            act("deleteImage", [heImage]),
            act("addImage", [{ id: "images/new" }]),
            act("setActivePanelId", ["images"], "appStore"),
            act("setImageChannelPreview", [null], "appStore"),
            act("rotateImage", [heImage]),
            act("updateImage", [heImage, { opacity: 0.5 }], "appStore"),
            act("updateImage", ["images/nope", { opacity: 0.5 }]),
            act("updateImage", [heImage, { dataSource: { type: "x" } }]),
            act("updateImage", [heImage, { opacity: 2, sparkle: true }]),
            act("updateImage", [morphologyFocus, { channels: [{}] }]),
            act("updateLabels", [
              cellLabels,
              { labelColor: { groupBy: { column: "obs/nope" } } },
            ]),
            act("updateLabels", [
              cellLabels,
              { labelVisibility: { groupBy: { column: "obs/leiden" } } },
            ]),
          ],
        },
        { content: "I could not." },
      ];
      await panel.say("Try everything.");
      expect(JSON.stringify(panel.projectStore.getState())).toBe(before);
      expect(JSON.stringify(panel.appStore.getState())).toBe(viewBefore);
      const results = panel.texts("Tool").map((text) => text.split("→ ")[1]);
      expect(results).toEqual([
        "Error: deleteImage is not allowed: deleting cannot be taken back safely. Nothing changed.",
        "Error: addImage is not allowed: new objects would need their data from a file. Nothing changed.",
        "Error: setActivePanelId is not allowed: another panel could hide the chat. Nothing changed.",
        "Error: setImageChannelPreview is not allowed: to show one channel, call updateImage with channelViewMode and activeChannel. Nothing changed.",
        expect.stringMatching(
          /^Error: there is no action "rotateImage" that you can call\. You can call: updateLayer, updateImage,/,
        ) as string,
        'Error: updateImage is an action of projectStore: call it with store "projectStore".',
        'Error: there is no entry of images with id "images/nope". Use an id from the app state.',
        "Error: nothing changed: dataSource cannot be changed.",
        expect.stringMatching(
          /^Error: nothing changed: opacity must be a number from 0 to 1; Image has no "sparkle"; it has name, opacity,/,
        ) as string,
        "Error: nothing changed: morphology_focus has 4 channels (0 DAPI, 1 ATP1A1, 2 18S, 3 AlphaSMA): give 4 entries, {} for the channels that stay.",
        'Error: nothing changed: the table "tables/table" has no column "obs/nope": give its exact name, e.g. "obs/leiden".',
        'Error: nothing changed: labelVisibility.groupBy needs "map": the id of a map of visibilityMaps.',
      ]);
    });

    it("adds a map, moves an image, renames the project and sets the view, and Undo takes all of it back", async () => {
      const panel = await openPanel(undefined, undefined, {
        provider: "anthropic",
      });
      await panel.pasteKey(anthropicKey);
      const group = {
        annotatedObject: { labelsId: cellLabels },
        groupBy: { column: "obs/leiden" },
        group: "3",
      };
      services.answers = [
        {
          toolCalls: [
            act("addColorMap", [
              {
                id: "my-colors",
                name: "My colors",
                values: { 3: { r: 255, g: 0, b: 0 } },
                default: { r: 200, g: 200, b: 200 },
              },
            ]),
            act("updateLabels", [
              cellLabels,
              {
                labelColor: {
                  groupBy: { column: "obs/leiden", map: "my-colors" },
                },
              },
            ]),
            act("moveImage", [heImage, 1]),
            act("setName", ["xenium 2"]),
            act("setHighlightedItemGroup", [group], "appStore"),
            act("setExpandedLabelsIds", [[nucleusLabels]], "appStore"),
          ],
        },
        { content: "Done." },
      ];
      await panel.say("Do all of it.");
      const state = () => panel.projectStore.getState();
      const view = () => panel.appStore.getState();
      // each says exactly what changed, from what to what
      expect(panel.texts("Tool")).toEqual([
        'addColorMap({id: "my-colors", name: "My colors", values.3: {"r":255,"g":0,"b":0}, default: {"r":200,"g":200,"b":200}})\n→ Done. Added to colorMaps the map "My colors" (id my-colors), with 1 group: 3.',
        'updateLabels(cell_labels, {labelColor: {"groupBy":{"column":"obs/leiden","map":"my-colors"}}})\n→ Done. Changed in cell_labels: labelColor {"random":{"palette":"distinct"}} → {"groupBy":{"column":"obs/leiden","map":"my-colors"}}. cell_labels is hidden: show it to see the change.',
        "moveImage(he_image, 1)\n→ Done. Moved he_image from index 0 to 1.",
        'setName("xenium 2")\n→ Done. Changed in the project: name "xenium" → "xenium 2".',
        // readable, with the name of the labels instead of their id
        'setHighlightedItemGroup({annotatedObject.labelsId: cell_labels, groupBy.column: "obs/leiden", group: "3"})\n→ Done. Changed in the view: highlightedItemGroup none → group "3" of column "obs/leiden" of cell_labels.',
        "setExpandedLabelsIds([nucleus_labels])\n→ Done. Changed in the view: expandedLabelsIds [cell_labels] → [nucleus_labels].",
      ]);
      expect(state().colorMaps.map(({ id }) => id)).toEqual([
        "map-1",
        "my-colors",
      ]);
      expect(state().images.map(({ name }) => name)).toEqual([
        "morphology_focus",
        "he_image",
      ]);
      expect(state().name).toBe("xenium 2");
      expect(view().highlightedItemGroup).toEqual(group);
      expect(view().expandedLabelsIds).toEqual([nucleusLabels]);
      panel.button("Undo").click();
      expect(state().colorMaps.map(({ id }) => id)).toEqual(["map-1"]);
      expect(
        (objectOf(panel.projectStore, cellLabels) as Labels).labelColor,
      ).toEqual({ random: { palette: "distinct" } });
      expect(state().images.map(({ name }) => name)).toEqual([
        "he_image",
        "morphology_focus",
      ]);
      expect(state().name).toBe("xenium");
      expect(view().highlightedItemGroup).toBeNull();
      expect(view().expandedLabelsIds).toEqual([cellLabels]);
      expect(panel.texts("Undo")).toEqual([
        'Undone: the expanded entries, the highlighted group, the project name "xenium 2" → "xenium", he_image back at position 1, cell_labels colors, My colors (the map added by the chat is deleted)',
      ]);
    });

    /**
     * A panel with Anthropic and the morphology image of the real Xenium
     * project: visible, with one channel in the project, and four in its data
     */
    async function morphologyPanel() {
      const projectStore = makeProjectStore();
      projectStore.setState((draft) => {
        const morphology = draft.images.find(
          ({ id }) => id === morphologyFocus,
        ) as Image;
        morphology.visibility = true;
        morphology.channels = [{ color: { r: 255, g: 255, b: 255 } }];
      });
      const dataStore = makeDataStore(makeTables(), {
        [morphologyFocus]: makeImageData(["DAPI", "ATP1A1", "18S", "AlphaSMA"]),
      });
      const panel = await openPanel(projectStore, dataStore, {
        provider: "anthropic",
      });
      await panel.pasteKey(anthropicKey);
      return panel;
    }

    /** The results of the tools, after the arrow of each Tool row */
    const results = (panel: { texts: (who: string) => string[] }) =>
      panel.texts("Tool").map((text) => text.split("\n→ ")[1]);

    it("counts the channels of the image data like the Images panel, also when the project lists fewer", async () => {
      const panel = await morphologyPanel();
      // all four, with their indices and as the panel shows them
      expect(panel.context()).toContain(
        "4 channels, composite (the visible channels blended): 0 DAPI (visible, white), 1 ATP1A1 (visible, rgb(0, 255, 0)), 2 18S (visible, rgb(0, 0, 255)), 3 AlphaSMA (visible, rgb(255, 255, 0))",
      );
      const hideOthers = act("updateImage", [
        morphologyFocus,
        { channels: [{}, ...repeat([{ visibility: false }], 3)] },
      ]);
      services.answers = [
        // one entry, as the project's list: the four are asked for, by name
        {
          toolCalls: [
            act("updateImage", [
              morphologyFocus,
              { channels: [{ visibility: true }] },
            ]),
          ],
        },
        { toolCalls: [hideOthers] },
        { content: "Only DAPI is shown now." },
      ];
      await panel.say("Show only channel 0 of morphology.");
      expect(results(panel)).toEqual([
        "Error: nothing changed: morphology_focus has 4 channels (0 DAPI, 1 ATP1A1, 2 18S, 3 AlphaSMA): give 4 entries, {} for the channels that stay.",
        "Done. Changed in morphology_focus: channels[1] (ATP1A1).visibility true → false; channels[2] (18S).visibility true → false; channels[3] (AlphaSMA).visibility true → false.",
      ]);
      // lengthened like the panel does, the first channel as it was
      expect(image(panel.projectStore, morphologyFocus).channels).toEqual([
        { color: { r: 255, g: 255, b: 255 } },
        { visibility: false },
        { visibility: false },
        { visibility: false },
      ]);
      expect(panel.context()).toContain(
        "0 DAPI (visible, white), 1 ATP1A1 (hidden, rgb(0, 255, 0)), 2 18S (hidden, rgb(0, 0, 255)), 3 AlphaSMA (hidden, rgb(255, 255, 0))",
      );
      // the same again changes nothing, and says so
      services.answers = [{ toolCalls: [hideOthers] }, { content: "Already." }];
      await panel.say("Hide the other channels.");
      expect(results(panel)[2]).toBe(
        "Nothing changed: in morphology_focus already channels[1] (ATP1A1).visibility = false; channels[2] (18S).visibility = false; channels[3] (AlphaSMA).visibility = false.",
      );
      panel.button("Undo").click();
      expect(image(panel.projectStore, morphologyFocus).channels).toEqual([
        { color: { r: 255, g: 255, b: 255 } },
      ]);
    });

    it("shows one channel with the color view mode, and says which one is shown", async () => {
      const panel = await morphologyPanel();
      services.answers = [
        {
          toolCalls: [
            act("updateImage", [morphologyFocus, { activeChannel: 4 }]),
            act("updateImage", [
              morphologyFocus,
              { channelViewMode: "color", activeChannel: 1 },
            ]),
          ],
        },
        { content: "Only ATP1A1 is shown." },
      ];
      await panel.say("Show only ATP1A1.");
      expect(results(panel)).toEqual([
        "Error: nothing changed: activeChannel must be the number of a channel, from 0 to 3.",
        'Done. Changed in morphology_focus: channelViewMode "composite" → "color"; activeChannel 0 → 1. In the color view mode, only channel 1 (ATP1A1) is shown, whatever the visibility of the channels.',
      ]);
      expect(panel.context()).toContain(
        "4 channels, color (only channel 1, ATP1A1, is shown): 0 DAPI",
      );
      panel.button("Undo").click();
      expect(image(panel.projectStore, morphologyFocus)).toMatchObject({
        channelViewMode: "composite",
        activeChannel: 0,
      });
    });

    it("says what already was so, and nothing changed when all of it was", async () => {
      const panel = await openPanel(undefined, undefined, {
        provider: "openai",
      });
      await panel.pasteKey(openAiKey);
      const before = JSON.stringify(panel.projectStore.getState());
      const viewBefore = JSON.stringify(panel.appStore.getState());
      services.answers = [
        {
          toolCalls: [
            act("updateImage", [heImage, { opacity: 1, visibility: true }]),
            act("setName", ["xenium"]),
            act("setHighlightedItemGroup", [null], "appStore"),
            act("moveImage", [heImage, 0]),
          ],
        },
        { content: "It was all like that already." },
      ];
      await panel.say("Make sure all of it is as it is.");
      expect(results(panel)).toEqual([
        "Nothing changed: in he_image already opacity = 1; visibility = true.",
        'Nothing changed: the project name already is "xenium".',
        "Nothing changed: highlightedItemGroup already is none.",
        "Nothing changed: he_image is already at index 0.",
      ]);
      expect(JSON.stringify(panel.projectStore.getState())).toBe(before);
      expect(JSON.stringify(panel.appStore.getState())).toBe(viewBefore);
      services.answers = [
        {
          toolCalls: [
            act("updateImage", [heImage, { opacity: 0.5, visibility: true }]),
          ],
        },
        { content: "he_image is half transparent." },
      ];
      await panel.say("Make the H&E image half transparent.");
      expect(results(panel)[4]).toBe(
        "Done. Changed in he_image: opacity 1 → 0.5. Already so: visibility = true.",
      );
      // only what changed is taken back
      panel.button("Undo").click();
      expect(panel.texts("Undo").at(-1)).toBe(
        "Undone: he_image opacity 0.5 → 1",
      );
    });

    it("tells every model that it cannot zoom yet, and how to do it by hand", async () => {
      const panel = await openPanel(undefined, undefined, {
        provider: "anthropic",
      });
      await panel.pasteKey(anthropicKey);
      await panel.say("Zoom out.");
      const hint = 'top right of the viewer ("Reset view")';
      expect(services.requests.at(-1)?.body?.system?.[0]?.text).toContain(hint);
      await (await openPanel()).say("Zoom out.");
      expect(ollama.requests.at(-1)?.messages[0]?.content).toContain(
        "mouse wheel zooms, dragging the image moves it",
      );
    });

    it("tells only OpenAI and Anthropic about call_action, and what they may change", async () => {
      const names = (tools: Record<string, unknown>[] = []) =>
        tools.map((tool) =>
          String(
            (tool.function as { name?: string } | undefined)?.name ?? tool.name,
          ),
        );
      for (const provider of ["openai", "anthropic"]) {
        const panel = await openPanel(undefined, undefined, { provider });
        await panel.pasteKey(provider === "openai" ? openAiKey : anthropicKey);
        await panel.say("Hello");
        const body = services.requests.at(-1)?.body;
        expect(names(body?.tools)).toContain("call_action");
        const fixed =
          provider === "openai" ? body?.instructions : body?.system?.[0]?.text;
        expect(fixed).toContain("Rename something");
        expect(fixed).not.toContain("You cannot change anything else");
      }
      const panel = await openPanel();
      modelCalls(act("updateImage", [heImage, { opacity: 0.5 }]));
      await panel.say("Hello");
      const request = ollama.requests[0];
      expect(names(request?.tools)).not.toContain("call_action");
      expect(request?.messages[0]?.content).toContain(
        "You cannot change anything else",
      );
      // and if a small model calls it anyway, it does nothing
      expect(panel.texts("Tool")[0]).toContain(
        'Error: there is no tool called "call_action".',
      );
      expect(panel.projectStore.getState().images[0]?.opacity).toBe(1);
    });
  });

  describe("the whole app for OpenAI and Anthropic", () => {
    /** The text of the user's message that starts the turn of a request */
    function turnText(provider: string, body?: ServiceRequestBody): string {
      const first =
        provider === "openai"
          ? body?.input?.[0]?.content
          : body?.messages?.[0]?.content[0]?.text;
      return typeof first === "string" ? first : "";
    }

    it.each(["openai", "anthropic"])(
      "tells %s the types with the fixed instructions, and the state with each turn",
      async (provider) => {
        const panel = await openPanel(undefined, undefined, { provider });
        await panel.pasteKey(provider === "openai" ? openAiKey : anthropicKey);
        await panel.say("What is the opacity of cell_labels?");
        const body = services.requests[0]?.body;
        const fixed =
          provider === "openai" ? body?.instructions : body?.system?.[0]?.text;
        expect(fixed).toContain("# The app state of TissUUmaps");
        expect(fixed).toContain(storeTypes);
        // "what can I change" is about TissUUmaps, and no made-up controls
        expect(fixed).toContain(
          "answer about TissUUmaps: list all the settings in the types",
        );
        expect(fixed).toContain("do not describe buttons");
        const [, json = ""] =
          turnText(provider, body)
            .split("[End of the context]")[0]
            ?.split("# The current app state, as JSON\n") ?? [];
        const state = JSON.parse(json) as {
          projectStore: ProjectStore;
          appStore: { expandedLabelsIds: string[] };
        };
        expect(state.projectStore.name).toBe("xenium");
        expect(state.projectStore.labels[1]).toMatchObject({
          id: cellLabels,
          opacity: 1,
        });
        expect(state.appStore.expandedLabelsIds).toEqual([cellLabels]);
        // without the actions and the long viewer options
        expect(json).not.toContain("updateLabels");
        expect(json).not.toContain("osOptions");
      },
    );

    it("tells Ollama no more than before", async () => {
      const panel = await openPanel();
      await panel.say("Hello");
      const system = ollama.requests[0]?.messages[0]?.content ?? "";
      expect(system).not.toContain("# The app state of TissUUmaps");
      expect(system).not.toContain("type Labels=");
    });

    it.each(["openai", "anthropic"])(
      "shows the tokens and about how much each turn with %s costs",
      async (provider) => {
        const panel = await openPanel(undefined, undefined, { provider });
        await panel.pasteKey(provider === "openai" ? openAiKey : anthropicKey);
        services.answers = [
          {
            toolCalls: [
              ["set_visibility", { ids: [transcripts], visible: true }],
            ],
          },
          { content: "The transcripts are visible." },
        ];
        await panel.say("Show the transcripts.");
        await panel.say("Thanks!");
        const lines = [...panel.container.querySelectorAll("[data-usage]")].map(
          (line) => line.textContent,
        );
        expect(lines).toEqual([
          "Tokens (2 requests): 6,000 input · 12,000 from cache · 0 written to cache · 160 output · ≈ $0.0008 · this session ≈ $0.0008",
          "Tokens: 3,000 input · 6,000 from cache · 0 written to cache · 80 output · ≈ $0.0004 · this session ≈ $0.0012",
        ]);
        // under the answers, not in their text
        expect(panel.texts("Dia")).toEqual([
          "The transcripts are visible.",
          "ok",
        ]);
      },
    );

    it("shows the tokens of Ollama, which are free", async () => {
      const panel = await openPanel();
      await panel.say("Hello");
      expect(panel.container.querySelector("[data-usage]")?.textContent).toBe(
        "Tokens: 5,000 input · 40 output · free",
      );
    });
  });

  describe("the model's thinking with OpenAI and Anthropic", () => {
    /** The user's message that starts the turn of a request, and its context */
    function turnOf(provider: string, body: ServiceRequestBody) {
      const items =
        provider === "openai"
          ? (body.input ?? []).map((item) => JSON.stringify(item))
          : (body.messages ?? []).map((message) => JSON.stringify(message));
      const index = items.findLastIndex((item) =>
        item.includes("[Context from the TissUUmaps chat panel"),
      );
      return { index, items, context: items[index] ?? "" };
    }

    it("the fake Anthropic refuses thinking whose conversation changed, as Anthropic does", async () => {
      const request = (system: string): ServiceRequestBody => ({
        model: "claude-haiku-5-5",
        system: [{ type: "text", text: system }],
        tools: [],
        messages: [{ role: "user", content: [{ type: "text", text: "Hi" }] }],
      });
      const signature = binding(request("fixed"), 1);
      const next = (system: string) => ({
        ...request(system),
        messages: [
          ...(request(system).messages ?? []),
          {
            role: "assistant",
            content: [
              { type: "thinking", thinking: "", signature },
              { type: "text", text: "Hello" },
            ],
          },
          { role: "user", content: [{ type: "text", text: "And?" }] },
        ],
      });
      const send = (body: ServiceRequestBody) =>
        fakeFetch("https://api.anthropic.com/v1/messages", {
          method: "POST",
          body: JSON.stringify(body),
        });
      expect((await send(next("fixed"))).status).toBe(200);
      const refused = await send(next("changed"));
      expect(refused.status).toBe(400);
      expect(JSON.stringify(await refused.json())).toContain(
        "The block is bound to a different conversation",
      );
    });

    it.each(["openai", "anthropic"])(
      "keeps a long conversation with %s going: questions, tools, a changing summary and theme",
      async (provider) => {
        const projectStore = makeProjectStore();
        const panel = await openPanel(projectStore, undefined, { provider });
        await panel.pasteKey(provider === "openai" ? openAiKey : anthropicKey);
        const turn = async (
          text: string,
          answers: (typeof services)["answers"],
        ) => {
          services.answers = answers;
          await panel.say(text);
        };
        await turn("What is shown now?", [
          { content: "The H&E image and the cell circles." },
        ]);
        await turn("Show the transcripts.", [
          {
            toolCalls: [
              ["set_visibility", { ids: [transcripts], visible: true }],
            ],
          },
          { content: "The transcripts are visible." },
        ]);
        await turn("How many leiden clusters are there?", [{ content: "16." }]);
        panel.setDark(true);
        await turn("Make the H&E image 50% transparent and the dots bigger.", [
          { toolCalls: [["set_opacity", { ids: [heImage], opacity: 0.5 }]] },
          {
            toolCalls: [
              ["set_point_size", { ids: [transcripts], change_percent: 20 }],
            ],
          },
          { content: "Done." },
        ]);
        panel.setDark(false);
        // a change by hand, between two questions
        projectStore.getState().updateImage(heImage, { opacity: 0.8 });
        await turn("Which images are visible?", [{ content: "he_image." }]);
        await turn("Hide all points.", [
          {
            toolCalls: [
              ["set_visibility", { category: "points", visible: false }],
            ],
          },
          { content: "Hidden." },
        ]);
        await turn("Thanks!", [{ content: "You are welcome." }]);
        await turn("What is shown now?", [{ content: "The H&E image." }]);

        expect(panel.texts("Error")).toEqual([]);
        expect(panel.texts("Aid")).toEqual(["Done."]);
        const { requests } = services;
        expect(requests).toHaveLength(12);
        // the beginning of every request is the same
        const beginning = (body: ServiceRequestBody) =>
          JSON.stringify([body.instructions, body.system, body.tools]);
        expect(new Set(requests.map(({ body }) => beginning(body))).size).toBe(
          1,
        );
        // the summary changes between the turns, but not within a turn
        const contexts = requests.map(
          ({ body }) => turnOf(provider, body).context,
        );
        expect(new Set(contexts).size).toBeGreaterThan(3);
        expect(contexts[1]).toBe(contexts[2]);
        expect(contexts[4]).toBe(contexts[6]);
        // the thinking goes back only within the turn
        for (const { body } of requests) {
          const { index, items } = turnOf(provider, body);
          items.forEach((item, position) => {
            if (/"type":"(thinking|reasoning)"/.test(item)) {
              expect(position).toBeGreaterThan(index);
            }
          });
        }
      },
    );

    it.each(["openai", "anthropic"])(
      "tries once more without the thinking when %s refuses it",
      async (provider) => {
        const panel = await openPanel(undefined, undefined, { provider });
        await panel.pasteKey(provider === "openai" ? openAiKey : anthropicKey);
        services.answers = [
          {
            toolCalls: [
              ["set_visibility", { ids: [transcripts], visible: true }],
            ],
          },
          { content: "The transcripts are visible." },
        ];
        services.refuseThinking = true;
        await panel.say("Show the transcripts.");
        expect(panel.texts("Error")).toEqual([]);
        expect(panel.texts("Dia")).toEqual(["The transcripts are visible."]);
        // the first step, the refused one, and the one sent again
        expect(services.requests).toHaveLength(3);
        const again = JSON.stringify(services.requests[2]?.body);
        expect(again).not.toMatch(/"type":"(thinking|reasoning)"/);
        // with the tool call and its result
        expect(again).toContain("_0_set_visibility");
        expect(again).toMatch(/function_call_output|tool_result/);
      },
    );

    it("switches from Anthropic to OpenAI and back in the middle of a conversation", async () => {
      const panel = await openPanel(undefined, undefined, {
        provider: "anthropic",
      });
      await panel.pasteKey(anthropicKey);
      await panel.chooseProvider("openai");
      await panel.pasteKey(openAiKey);
      for (const [provider, ids] of [
        ["anthropic", [transcripts]],
        ["openai", [cellCircles]],
        ["anthropic", [heImage]],
      ] as const) {
        await panel.chooseProvider(provider);
        services.answers = [
          { toolCalls: [["set_visibility", { ids, visible: false }]] },
          { content: "Hidden." },
        ];
        await panel.say("Hide it.");
      }
      expect(panel.texts("Error")).toEqual([]);
      expect(panel.texts("Dia")).toEqual(["Hidden.", "Hidden.", "Hidden."]);
    });
  });

  describe("look and feel", () => {
    it("goes through the sent messages with the arrows, like a terminal", async () => {
      const panel = await openPanel();
      await panel.say("first");
      await panel.say("second");
      panel.input.value = "new text";
      panel.press("ArrowUp");
      expect(panel.input.value).toBe("second");
      panel.press("ArrowUp");
      expect(panel.input.value).toBe("first");
      // no further back than the first message
      panel.press("ArrowUp");
      expect(panel.input.value).toBe("first");
      panel.press("ArrowDown");
      panel.press("ArrowDown");
      expect(panel.input.value).toBe("new text");
      // in a longer text, the arrows move between its lines
      panel.input.value = "line 1\nline 2";
      panel.input.setSelectionRange(10, 10);
      panel.press("ArrowUp");
      expect(panel.input.value).toBe("line 1\nline 2");
    });

    it("quotes a selected part of an answer in the next message", async () => {
      const panel = await openPanel();
      // the browser only selects text that is on the page
      document.body.append(panel.container);
      ollama.answers = [{ content: "Cluster 3 is the biggest one." }];
      await panel.say("Which cluster is the biggest?");
      const quote = panel.button("Quote");
      expect(quote.style.display).toBe("none");
      const answer = panel.messages().find((m) => m.who === "Dia");
      const range = document.createRange();
      range.selectNodeContents(answer?.element as HTMLElement);
      window.getSelection()?.removeAllRanges();
      window.getSelection()?.addRange(range);
      document.dispatchEvent(new Event("selectionchange"));
      expect(quote.style.display).toBe("");
      panel.input.value = "Color it red.";
      quote.click();
      expect(panel.input.value).toBe(
        "> Cluster 3 is the biggest one.\n\nColor it red.",
      );
      // the questions of the user cannot be quoted
      const question = panel.messages().find((m) => m.who === "You");
      range.selectNodeContents(question?.element as HTMLElement);
      window.getSelection()?.removeAllRanges();
      window.getSelection()?.addRange(range);
      document.dispatchEvent(new Event("selectionchange"));
      expect(quote.style.display).toBe("none");
    });

    it("shows an Italian word while the model thinks, with its translation, which stays while pointed at", async () => {
      const panel = await openPanel();
      let answer = () => {};
      ollama.thinking = new Promise((resolve) => (answer = resolve));
      panel.input.value = "Hello";
      panel.button("Send").click();
      await vi.waitFor(() =>
        expect(panel.container.querySelector('[role="tooltip"]')).not.toBe(
          null,
        ),
      );
      const tooltip = panel.container.querySelector(
        '[role="tooltip"]',
      ) as HTMLElement;
      const word = tooltip.previousElementSibling as HTMLElement;
      const words = new Map(
        thinkingWords.map(({ word, translation }) => [word, translation]),
      );
      expect(words.has(word.textContent)).toBe(true);
      expect(tooltip.style.display).toBe("none");
      word.dispatchEvent(new MouseEvent("mouseenter"));
      expect(tooltip.style.display).toBe("");
      expect(tooltip.textContent).toBe(words.get(word.textContent));
      const pointed = word.textContent;
      await new Promise((resolve) => setTimeout(resolve, 2200));
      expect(word.textContent).toBe(pointed);
      answer();
      await vi.waitFor(() =>
        expect(panel.button("Send").textContent).toBe("Send"),
      );
      expect(panel.container.querySelector('[role="tooltip"]')).toBe(null);
      expect(panel.texts("Dia")).toEqual(["ok"]);
    }, 10_000);

    it("is Dia with the light theme and Aid with the dark one, at once, with the same rules", async () => {
      const panel = await openPanel();
      await panel.say("Hello");
      expect(panel.texts("Dia")).toEqual(["ok"]);
      expect(ollama.requests[0]?.messages[0]?.content).toContain(
        "Your name is Dia. Your tone is gentle",
      );
      panel.setDark(true);
      await panel.say("Hello again");
      expect(panel.texts("Aid")).toEqual(["ok"]);
      const system = ollama.requests[1]?.messages[0]?.content ?? "";
      expect(system).toContain("Your name is Aid. Your tone is serious");
      expect(system).toContain(
        "The personality only changes your tone. Every rule of your instructions still applies",
      );
      expect(panel.context()).toContain("Your name is Aid.");
      expect(system).toContain(
        'Speak of yourself as\n"I", never by your name in the third person.',
      );
    });

    it("says who made it", async () => {
      const panel = await openPanel();
      expect(panel.container.textContent).toContain(
        "Made by Alessia Giustolisi",
      );
    });
  });

  describe("OpenAI and Anthropic", () => {
    it("offers OpenAI and Anthropic first, and asks for a key before sending", async () => {
      const panel = await openPanel(undefined, undefined, { provider: null });
      expect(panel.providers()).toEqual([
        "OpenAI",
        "Anthropic",
        "Ollama (local)",
        "Ollama (cloud)",
      ]);
      expect(panel.provider()).toBe("openai");
      expect(panel.model()).toBe("gpt-6-luna");
      expect(panel.keyInput.type).toBe("password");
      expect(panel.status()).toBe(
        'OpenAI: paste your API key below. No key yet? Click "Help".',
      );
      await panel.say("Hello");
      expect(panel.texts("Error")).toEqual([
        'Paste your OpenAI API key into the "API key" field first. No key yet? Click "Help".',
      ]);
      expect(network).toEqual([]);
    });

    it("runs the tools with OpenAI, and sends back the results and the model's thinking", async () => {
      const panel = await openPanel(undefined, undefined, { provider: null });
      await panel.pasteKey(openAiKey);
      expect(panel.status()).toMatch(/^OpenAI: key accepted\./);
      services.answers = [
        {
          toolCalls: [
            ["set_visibility", { ids: [transcripts], visible: true }],
          ],
        },
        { content: "The transcripts are **visible** now." },
      ];
      await panel.say("Show the transcripts.");
      expect(objectOf(panel.projectStore, transcripts).visibility).toBe(true);
      expect(panel.texts("Dia")).toEqual(["The transcripts are visible now."]);
      const [first, second] = services.requests;
      expect(first?.url).toBe("https://api.openai.com/v1/responses");
      expect(first?.headers.get("Authorization")).toBe(`Bearer ${openAiKey}`);
      expect(first?.body.model).toBe("gpt-6-luna");
      expect(first?.body.store).toBe(false);
      expect(first?.body.instructions).toContain(
        "You are an assistant inside TissUUmaps",
      );
      expect(first?.body.tools[0]).toMatchObject({
        type: "function",
        name: "set_visibility",
        strict: false,
      });
      expect(second?.body.input).toEqual(
        expect.arrayContaining([
          { type: "reasoning", id: "rs_0", encrypted_content: "thinking-0" },
          expect.objectContaining({
            type: "function_call",
            call_id: "call_0_set_visibility",
          }),
          {
            type: "function_call_output",
            call_id: "call_0_set_visibility",
            output: expect.stringContaining("Done") as string,
          },
        ]),
      );
    });

    it("runs the tools with Anthropic, with the header that lets a browser page use it", async () => {
      const panel = await openPanel(undefined, undefined, {
        provider: "anthropic",
      });
      await panel.pasteKey(anthropicKey);
      services.answers = [
        { toolCalls: [["set_opacity", { ids: [heImage], opacity: 0.5 }]] },
        { content: "The H&E image is 50% transparent now." },
      ];
      await panel.say("Make the H&E image 50% transparent.");
      expect(panel.projectStore.getState().images[0]?.opacity).toBe(0.5);
      expect(panel.texts("Dia")).toEqual([
        "The H&E image is 50% transparent now.",
      ]);
      const [first, second] = services.requests;
      expect(first?.url).toBe("https://api.anthropic.com/v1/messages");
      expect(first?.headers.get("Authorization")).toBe(
        `Bearer ${anthropicKey}`,
      );
      expect(
        first?.headers.get("anthropic-dangerous-direct-browser-access"),
      ).toBe("true");
      expect(first?.headers.get("anthropic-version")).toBe("2023-06-01");
      // the fixed instructions, marked for prompt caching
      expect(first?.body.system?.[0]?.text).toMatch(
        /^You are an assistant inside TissUUmaps/,
      );
      expect(first?.body.system?.[0]?.cache_control).toEqual({
        type: "ephemeral",
      });
      expect(first?.body.max_tokens).toBeGreaterThan(0);
      expect(first?.body.tools[0]).toMatchObject({
        name: "set_visibility",
        input_schema: { type: "object" },
      });
      // the thinking goes back as it came, and the results go in a user message
      expect(second?.body.messages?.slice(1)).toEqual([
        {
          role: "assistant",
          content: [
            {
              type: "thinking",
              thinking: "Let me see.",
              signature: binding(first?.body as ServiceRequestBody, 1),
            },
            {
              type: "tool_use",
              id: "toolu_0_set_opacity",
              name: "set_opacity",
              input: { ids: [heImage], opacity: 0.5 },
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "toolu_0_set_opacity",
              content: expect.stringContaining("Done") as string,
            },
          ],
        },
      ]);
    });

    it("keeps the conversation when the user switches to another service", async () => {
      const panel = await openPanel(undefined, undefined, { provider: null });
      await panel.pasteKey(openAiKey);
      services.answers = [
        {
          toolCalls: [
            ["set_visibility", { ids: [transcripts], visible: true }],
          ],
        },
        { content: "ok" },
      ];
      await panel.say("Show the transcripts.");
      await panel.chooseProvider("anthropic");
      await panel.pasteKey(anthropicKey);
      await panel.say("And now?");
      const { messages } = services.requests.at(-1)?.body ?? {};
      expect(messages?.map((message) => message.role)).toEqual([
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
      ]);
      expect(messages?.[1]?.content).toEqual([
        {
          type: "tool_use",
          id: "call_0_set_visibility",
          name: "set_visibility",
          input: { ids: [transcripts], visible: true },
        },
      ]);
      expect(messages?.[2]?.content[0]).toMatchObject({
        type: "tool_result",
        tool_use_id: "call_0_set_visibility",
      });
    });

    it("remembers the key only when asked, and forgets it", async () => {
      const { localStorage } = window;
      const stored = () =>
        Array.from({ length: localStorage.length }, (_, i) =>
          localStorage.getItem(localStorage.key(i) ?? ""),
        );
      const panel = await openPanel(undefined, undefined, { provider: null });
      await panel.pasteKey(openAiKey);
      expect(stored()).not.toContain(openAiKey);
      panel.remember(true);
      expect(stored()).toContain(openAiKey);
      const next = await openPanel(undefined, undefined, { provider: null });
      expect(next.keyInput.value).toBe(openAiKey);
      expect(next.rememberBox.checked).toBe(true);
      // Anthropic does not get the OpenAI key
      await next.chooseProvider("anthropic");
      expect(next.keyInput.value).toBe("");
      await next.chooseProvider("openai");
      next.button("Forget key").click();
      expect(next.keyInput.value).toBe("");
      expect(next.rememberBox.checked).toBe(false);
      expect(stored()).not.toContain(openAiKey);
      expect(next.status()).toBe(
        'OpenAI: paste your API key below. No key yet? Click "Help".',
      );
    });

    it("never shows or logs the key, and sends it only to its own service", async () => {
      const logged = vi.fn();
      for (const method of ["log", "info", "warn", "error", "debug"] as const) {
        vi.spyOn(console, method).mockImplementation(logged);
      }
      const panel = await openPanel(undefined, undefined, { provider: null });
      await panel.pasteKey(openAiKey);
      await panel.say("Hello");
      await panel.chooseProvider("anthropic");
      await panel.pasteKey(anthropicKey);
      await panel.say("Hello");
      // an error that repeats the key
      services.error = {
        status: 429,
        body: {
          error: {
            type: "rate_limit_error",
            message: `Rate limit reached for ${anthropicKey}`,
          },
        },
      };
      await panel.say("Hello");
      await panel.chooseProvider("ollama");
      await panel.say("Hello");
      panel.context();
      panel.button("Help").click();
      expect(panel.texts("Error")).toEqual([
        "Anthropic: too many requests, or no credit left. Wait a moment and try again, or check the billing of your Anthropic account. (Rate limit reached for [key hidden])",
      ]);
      for (const key of [openAiKey, anthropicKey]) {
        expect(panel.container.textContent).not.toContain(key);
        expect(panel.container.innerHTML).not.toContain(key);
        expect(JSON.stringify(logged.mock.calls)).not.toContain(key);
      }
      for (const { url, headers, body } of network) {
        const sent = JSON.stringify([...headers]) + body;
        if (sent.includes(openAiKey)) {
          expect(url).toMatch(/^https:\/\/api\.openai\.com\//);
        }
        if (sent.includes(anthropicKey)) {
          expect(url).toMatch(/^https:\/\/api\.anthropic\.com\//);
        }
        expect(sent.includes(openAiKey) && sent.includes(anthropicKey)).toBe(
          false,
        );
      }
      expect(
        network.some(({ url }) => url.startsWith("http://localhost:11434/")),
      ).toBe(true);
    });

    it("explains a key that is not accepted, without showing any part of it", async () => {
      services.error = {
        status: 401,
        body: {
          error: {
            code: "invalid_api_key",
            message: `Incorrect API key provided: ${openAiKey.slice(0, 8)}****${openAiKey.slice(-4)}.`,
          },
        },
      };
      const panel = await openPanel(undefined, undefined, { provider: null });
      await panel.pasteKey(openAiKey);
      const explanation =
        'OpenAI did not accept the API key. Check that you copied all of it and that it has not expired or been deleted, or make a new key (see "Help").';
      expect(panel.status()).toBe(explanation);
      await panel.say("Hello");
      expect(panel.texts("Error")).toEqual([explanation]);
      expect(panel.container.textContent).not.toContain(openAiKey.slice(-4));
    });

    it("explains an Anthropic key that is not tied to one workspace", async () => {
      services.error = {
        status: 400,
        body: {
          error: {
            type: "invalid_request_error",
            message:
              "This API key is not scoped to a workspace, so this request must include the anthropic-workspace-id header with the ID of the workspace to use.",
          },
        },
      };
      const panel = await openPanel(undefined, undefined, {
        provider: "anthropic",
      });
      await panel.pasteKey(anthropicKey);
      expect(panel.status()).toMatch(
        /^This Anthropic key is not tied to one workspace/,
      );
    });

    it("folds the settings into one line while the key works, and opens them again", async () => {
      const panel = await openPanel(undefined, undefined, {
        provider: "anthropic",
      });
      expect(panel.settingsShown()).toBe(true);
      expect(panel.button("⚙").style.display).toBe("none");
      await panel.pasteKey(anthropicKey);
      expect(panel.settingsShown()).toBe(false);
      expect(panel.summary.textContent).toBe(
        "Anthropic · claude-haiku-5-5 · key ✓",
      );
      // the first time, the chat says in full what using the key means
      expect(panel.texts("Anthropic")).toEqual([
        "Anthropic: key accepted. Each request is billed to your Anthropic account, and your messages and the project summary are sent to Anthropic. The key settings are under ⚙.",
      ]);
      panel.button("⚙").click();
      expect(panel.settingsShown()).toBe(true);
      expect(panel.status()).toMatch(/^Anthropic: key accepted\./);
      panel.button("⚙").click();
      expect(panel.settingsShown()).toBe(false);
      // a key that stops working opens them by themselves
      services.error = {
        status: 401,
        body: { error: { type: "authentication_error", message: "invalid" } },
      };
      await panel.say("Hello");
      expect(panel.settingsShown()).toBe(true);
      expect(panel.status()).toMatch(/^Anthropic did not accept the API key/);
    });

    it("says when the service cannot be reached", async () => {
      const panel = await openPanel(undefined, undefined, { provider: null });
      await panel.pasteKey(openAiKey);
      services.online = false;
      await panel.say("Hello");
      expect(panel.texts("Error")).toEqual([
        'OpenAI does not answer. Check the internet connection, then click "Check again".',
      ]);
      expect(panel.texts("You")).toEqual(["Hello"]);
    });
  });
});
