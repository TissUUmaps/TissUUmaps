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
    { type: "response.completed", response: { output } } as { type: string },
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
    { type: "message_start", message: { role: "assistant", content: [] } },
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
    { type: "message_delta", delta: { stop_reason: "end_turn" } },
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
  const unmount = aiChatPlugin.mount?.(container, {
    projectStore,
    dataStore,
    appStore: {} as never,
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
