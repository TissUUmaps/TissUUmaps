// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Plugin, PluginRegistry, PluginStores } from "@tissuumaps/core";

import {
  type ChatPluginEntry,
  type PluginHost,
  type PluginStorage,
  makeChatPlugins,
  pluginFileText,
  pluginId,
  readPluginFile,
} from "./chat-plugins";
import type { ScriptQuestion, ScriptRun } from "./sandbox";

/** The fake sandbox of JavaScript: asks what code would ask, then answers */
const sandbox = vi.hoisted(() => {
  const state: {
    runs: ScriptRun[];
    questions: ScriptQuestion[];
    outcome: { result?: string; error?: string; logs: string[] };
  } = {
    runs: [],
    questions: [],
    outcome: { result: '{"z": 1.5}', logs: ["hi"] },
  };
  return state;
});
vi.mock("./sandbox", () => ({
  runScript: async (
    run: ScriptRun,
    answer: (question: ScriptQuestion) => Promise<unknown>,
  ) => {
    sandbox.runs.push(run);
    for (const question of sandbox.questions) {
      await answer(question);
    }
    return sandbox.outcome;
  },
}));
vi.mock("./summary", () => ({ stateJson: () => '{"name":"p"}' }));

/**
 * The modules of the plugins of the tests, by their code: jsdom cannot import
 * modules, so each code shown to the user has its module written here
 */
const modules = new Map<string, unknown>();

/** Imports a plugin as the page would, from {@link modules} */
const load = (code: string) => Promise.resolve(modules.get(code) ?? {});

/** A registry like the one of TissUUmaps, which runs setup, mount and teardown */
function makeRegistry() {
  const registered = new Map<
    string,
    {
      plugin: Plugin;
      teardown: (() => void) | void;
      unmount?: (() => void) | void;
    }
  >();
  const stores = { name: "the stores" } as unknown as PluginStores;
  const registry: PluginRegistry = {
    registerPlugin: (plugin) => {
      registry.unregisterPlugin(plugin.id);
      try {
        registered.set(plugin.id, { plugin, teardown: plugin.setup?.(stores) });
      } catch {
        // logged by TissUUmaps
      }
    },
    mountPlugin: (id) => {
      const entry = registered.get(id);
      if (entry !== undefined && entry.unmount === undefined) {
        try {
          entry.unmount = entry.plugin.mount?.(
            document.createElement("div"),
            stores,
          );
        } catch {
          // logged by TissUUmaps
        }
      }
    },
    unmountPlugin: () => {},
    unregisterPlugin: (id) => {
      const entry = registered.get(id);
      registered.delete(id);
      entry?.unmount?.();
      entry?.teardown?.();
    },
  };
  return { registry, registered };
}

/** A fake chat: no project, a Python per slot, one skill from GitHub */
function makeHost() {
  const kernels: { source: string; stopped: boolean; runs: unknown[] }[] = [];
  const saved: unknown[] = [];
  const skill = {
    name: "anhood",
    description: "Neighborhood enrichment",
    files: new Map([["anhood/__init__.py", new Blob(["x = 1"])]]),
  };
  const host = {
    stores: {
      projectStore: {
        getState: () => ({ instanceId: "p1" }),
        subscribe: () => () => {},
      },
      appStore: { getState: () => ({ plugins: new Map([["ai-chat", {}]]) }) },
    },
    readData: (question: ScriptQuestion) =>
      Promise.resolve(
        question.kind === "tables" ? [{ id: "t", name: "t" }] : [],
      ),
    saveColumn: (question: ScriptQuestion) => {
      saved.push(question);
      return Promise.resolve({ result: { table: "t (chat)", column: "c" } });
    },
    makeKernelSlot: () => {
      let current: (typeof kernels)[number] | undefined;
      return {
        kernel: (source: string) => {
          if (current === undefined || current.source !== source) {
            if (current !== undefined) {
              current.stopped = true;
            }
            current = { source, stopped: false, runs: [] };
            kernels.push(current);
          }
          const kernel = current;
          return Promise.resolve({
            run: (
              run: unknown,
              _answer: unknown,
              options: { onRunning?: () => void },
            ) => {
              kernel.runs.push(run);
              options.onRunning?.();
              return Promise.resolve({ result: "[1, 2]", logs: [] });
            },
          });
        },
        stop: () => {
          if (current !== undefined) {
            current.stopped = true;
          }
        },
      };
    },
    skillList: {
      activeSkills: () => [skill],
      originOf: () => ({
        kind: "github",
        github: { repo: "lab/anhood", path: "", commit: "abc", license: "MIT" },
      }),
    },
    pythonDownload: () => undefined,
    onPythonDownload: () => () => {},
    signal: new AbortController().signal,
  } as unknown as PluginHost;
  return { host, kernels, saved };
}

/** A storage like the browser's, in memory */
function makeStorage(entries: ChatPluginEntry[] = []) {
  const kept = new Map(entries.map((entry) => [entry.id, entry]));
  const storage: PluginStorage = {
    load: () => Promise.resolve([...kept.values()]),
    save: (entry) => {
      kept.set(entry.id, entry);
      return Promise.resolve();
    },
    remove: (id) => {
      kept.delete(id);
      return Promise.resolve();
    },
  };
  return { storage, kept };
}

/** A plugin made by the chat, with this code */
function entryOf(code: string, id = "chat-test"): ChatPluginEntry {
  return { id, name: "Test", description: "A test.", code, made: "2026-11-18" };
}

/** What the plugins of the tests kept, the last one first */
type Seen = {
  chat: Record<string, (...args: unknown[]) => Promise<unknown>>;
  mounted?: unknown;
  container?: HTMLElement;
  unmounted?: boolean;
};
let lastSeen: Seen | undefined;

/** The code of a plugin that keeps its `chat` and stores in `seen` */
const keeping = `export default function createPlugin(chat) {
  return { mount(container, stores) { ... } };
}`;
modules.set(keeping, {
  default: (chat: Seen["chat"]) => {
    const mine: Seen = { chat };
    lastSeen = mine;
    return {
      mount: (container: HTMLElement, stores: unknown) => {
        mine.mounted = stores;
        mine.container = container;
        return () => {
          mine.unmounted = true;
        };
      },
    };
  },
});
modules.set("export default () => 5", { default: () => 5 });
modules.set("setup throws", {
  default: () => ({
    setup: () => {
      throw new Error("no data");
    },
  }),
});
modules.set("mount throws", {
  default: () => ({
    mount: () => {
      throw new Error("no panel");
    },
  }),
});

/** What the last plugin kept */
const seen = () => lastSeen!;

afterEach(() => {
  sandbox.runs.length = 0;
  sandbox.questions.length = 0;
  sandbox.outcome = { result: '{"z": 1.5}', logs: ["hi"] };
});

describe("pluginId", () => {
  it("makes ids with the prefix chat-, of small letters, digits and dashes", () => {
    expect(pluginId("anhood", "x")).toBe("chat-anhood");
    expect(pluginId("Chat-Anhood!", "x")).toBe("chat-anhood");
    expect(pluginId(undefined, "Neighborhood enrichment")).toBe(
      "chat-neighborhood-enrichment",
    );
    expect(pluginId("Café", "x")).toBe("chat-cafe");
    expect(pluginId("chat-", "")).toEqual({
      error:
        'give the plugin an id of letters, digits and dashes, e.g. "anhood"',
    });
  });
});

describe("plugin files", () => {
  it("exports a plugin with a header that it reads back", () => {
    const entry: ChatPluginEntry = {
      ...entryOf(keeping, "chat-anhood"),
      name: "Neighborhood enrichment",
      origin: {
        skill: "anhood",
        github: { repo: "lab/anhood", path: "", commit: "abc", license: "MIT" },
      },
    };
    const text = pluginFileText(entry);
    expect(text).toContain(
      "// based on the skill: anhood (github.com/lab/anhood, commit abc, license: MIT)",
    );
    expect(text.endsWith(keeping)).toBe(true);
    expect(readPluginFile(text, "x.js")).toEqual({
      id: "chat-anhood",
      name: "Neighborhood enrichment",
      description: "A test.",
      code: text,
      made: "2026-11-18",
    });
  });

  it("reads a module without a header, and refuses a file that is not a plugin", () => {
    expect(readPluginFile(keeping, "My tool.js")).toMatchObject({
      id: "chat-my-tool",
      name: "My tool",
      description: "From the file My tool.js.",
    });
    expect(readPluginFile("console.log(1)", "a.js")).toEqual({
      error: 'a.js is not a plugin for the AI chat: it has no "export default"',
    });
  });
});

describe("makeChatPlugins", () => {
  it("runs a plugin the user said yes to, opens it, and gives it chat", async () => {
    const { host, saved } = makeHost();
    const { registry, registered } = makeRegistry();
    const plugins = makeChatPlugins(host, makeStorage().storage, {
      load,
      registry: () => registry,
    });
    await plugins.install(entryOf(keeping), { remember: false });
    expect(registered.get("chat-test")?.plugin.name).toBe("Test");
    expect(plugins.get("chat-test")).toMatchObject({
      running: true,
      remembered: false,
    });
    expect(plugins.isConfirmed("chat-test")).toBe(true);
    await plugins.open("chat-test");
    expect(seen().mounted).toEqual({ name: "the stores" });
    // the chat signs the panel of the plugins it makes
    expect(seen().container?.lastElementChild?.textContent).toBe(
      "This plugin was made with the AID of DIA, the AI agent of TissUUmaps",
    );

    const { chat } = seen();
    expect(await chat["tables"]!()).toEqual([{ id: "t", name: "t" }]);
    sandbox.questions.push({ kind: "download", name: "z.csv", text: "a,b\n" });
    const result = await chat["run"]!({
      code: "export default () => 1",
      args: { k: 5 },
    });
    expect(result).toEqual({
      result: { z: 1.5 },
      text: '{"z": 1.5}',
      logs: ["hi"],
      charts: [],
      files: [{ name: "z.csv", text: "a,b\n" }],
      saved: [],
    });
    expect(sandbox.runs[0]).toEqual({
      code: "export default () => 1",
      args: { k: 5 },
      project: { name: "p" },
    });
    // a column saved by the plugin goes through the chat, without Undo
    await chat["saveColumn"]!("t", "c", [1, 2], { ids: ["a", "b"] });
    expect(saved).toEqual([
      {
        kind: "save_column",
        table: "t",
        column: "c",
        values: [1, 2],
        ids: ["a", "b"],
      },
    ]);
  });

  it("gives each plugin a Python of its own, with the files of a skill that is on", async () => {
    const { host, kernels } = makeHost();
    const { registry } = makeRegistry();
    const plugins = makeChatPlugins(host, makeStorage().storage, {
      load,
      registry: () => registry,
    });
    await plugins.install(entryOf(keeping, "chat-one"), { remember: false });
    const one = seen().chat;
    await plugins.install(entryOf(keeping, "chat-two"), { remember: false });
    const two = seen().chat;
    const progress: string[] = [];
    const ran = await one["run"]!({
      language: "python",
      skill: "anhood",
      code: "async def main(tm): ...",
      onProgress: (text: string) => progress.push(text),
    });
    expect(ran).toMatchObject({ result: [1, 2] });
    expect(progress).toEqual([
      "starting Python",
      "loading the Python packages",
      "running",
    ]);
    await two["run"]!({ language: "python", code: "async def main(tm): ..." });
    expect(kernels.map(({ source }) => source)).toEqual([
      "plugin:chat-one:github:lab/anhood/@abc",
      "plugin:chat-two:",
    ]);
    expect(kernels[0]?.runs[0]).toMatchObject({
      files: { "anhood/__init__.py": expect.any(ArrayBuffer) as unknown },
    });
    await expect(
      two["run"]!({ language: "python", skill: "nope", code: "x" }),
    ).rejects.toThrow('the skill "nope" is not on in the AI chat');
    await expect(two["run"]!({ language: "r", code: "x" })).rejects.toThrow(
      'chat.run runs "javascript" or "python", not "r"',
    );

    // removing a plugin stops its Python and its panel
    await plugins.remove("chat-one");
    expect(kernels[0]?.stopped).toBe(true);
    expect(kernels[1]?.stopped).toBe(false);
    await expect(one["run"]!({ code: "x" })).rejects.toThrow("was closed");
  });

  it("refuses code in the other language, and gives the colors of the theme", async () => {
    const { host } = makeHost();
    const { registry } = makeRegistry();
    const plugins = makeChatPlugins(host, makeStorage().storage, {
      load,
      registry: () => registry,
    });
    await plugins.install(entryOf(keeping), { remember: false });
    const { chat } = seen();
    expect(Object.keys(chat)).toEqual([
      "pluginId",
      "run",
      "tables",
      "columns",
      "read",
      "saveColumn",
      "skills",
      "theme",
      "onThemeChange",
      "heatmap",
      "bars",
      "scatter",
      "legend",
      "showShapes",
      "hideShapes",
    ]);
    await expect(
      chat["run"]!({
        language: "python",
        code: "export default async function (tm) { return 1; }",
      }),
    ).rejects.toThrow(
      'this looks like JavaScript, not Python: pass language "javascript", or write Python with async def main(tm): ...',
    );
    expect(sandbox.runs).toEqual([]);

    // colors a canvas understands, which follow the theme
    const root = document.documentElement;
    root.style.setProperty("--background", "rgb(255, 255, 255)");
    root.style.setProperty("--foreground", "rgb(10, 10, 10)");
    const theme = chat["theme"] as unknown as () => Record<string, unknown>;
    expect(theme()).toMatchObject({
      background: "rgb(255, 255, 255)",
      foreground: "rgb(10, 10, 10)",
      dark: false,
    });
    const changes: unknown[] = [];
    const onThemeChange = chat["onThemeChange"] as unknown as (
      listener: (colors: Record<string, unknown>) => void,
    ) => () => void;
    const stop = onThemeChange((colors) => changes.push(colors["dark"]));
    root.style.setProperty("--background", "rgb(20, 20, 20)");
    root.classList.add("dark");
    await vi.waitFor(() => expect(changes).toEqual([true]));
    stop();
    root.classList.remove("dark");
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(changes).toEqual([true]);
    root.removeAttribute("style");

    // the ready-made charts go away with the plugin
    const place = document.createElement("div");
    const heatmap = chat["heatmap"] as unknown as (
      element: HTMLElement,
      options: object,
    ) => unknown;
    heatmap(place, {
      rows: ["0", "1"],
      values: [
        [1, -1],
        [-1, 1],
      ],
    });
    expect(place.querySelector("[data-chat-chart]")).not.toBeNull();
    await plugins.remove("chat-test");
    expect(place.querySelector("[data-chat-chart]")).toBeNull();
  });

  it("throws what failed in the code, with its logs", async () => {
    const { host } = makeHost();
    const { registry } = makeRegistry();
    const plugins = makeChatPlugins(host, makeStorage().storage, {
      load,
      registry: () => registry,
    });
    await plugins.install(entryOf(keeping), { remember: false });
    sandbox.outcome = { error: "x is not defined", logs: ["before"] };
    const error = (await seen().chat["run"]!({ code: "x" }).catch(
      (problem: unknown) => problem,
    )) as Error & { logs?: string[] };
    expect(error.message).toBe("x is not defined");
    expect(error.logs).toEqual(["before"]);
  });

  it("lists the remembered plugins, but runs none before the user says yes", async () => {
    const { host } = makeHost();
    const { registry, registered } = makeRegistry();
    const { storage, kept } = makeStorage([entryOf(keeping)]);
    const plugins = makeChatPlugins(host, storage, {
      load,
      registry: () => registry,
    });
    await vi.waitFor(() => expect(plugins.list()).toHaveLength(1));
    expect(plugins.get("chat-test")).toMatchObject({
      remembered: true,
      running: false,
    });
    expect(registered.size).toBe(0);
    await expect(plugins.open("chat-test")).rejects.toThrow(
      'the user has not said yes to the plugin "Test" since the page was loaded',
    );
    // after the yes, it runs and stays remembered
    await plugins.install(plugins.get("chat-test")!.entry, { remember: false });
    expect(registered.has("chat-test")).toBe(true);
    expect(plugins.get("chat-test")?.remembered).toBe(true);
    // forgotten, then removed
    expect(await plugins.setRemembered("chat-test", false)).toBeUndefined();
    expect(kept.size).toBe(0);
    await plugins.remove("chat-test");
    expect(registered.size).toBe(0);
    expect(plugins.list()).toEqual([]);
  });

  it("replaces an earlier version, and refuses code that is not a plugin", async () => {
    const { host } = makeHost();
    const { registry, registered } = makeRegistry();
    const plugins = makeChatPlugins(host, makeStorage().storage, {
      load,
      registry: () => registry,
    });
    await plugins.install(entryOf(keeping), { remember: false });
    await plugins.open("chat-test");
    const first = seen();
    await plugins.install(
      { ...entryOf(keeping), name: "Test 2" },
      { remember: false },
    );
    expect(first.unmounted).toBe(true);
    await expect(first.chat["run"]!({ code: "x" })).rejects.toThrow(
      "was closed",
    );
    expect(registered.get("chat-test")?.plugin.name).toBe("Test 2");

    await expect(
      plugins.install(entryOf("const x = 1;", "chat-bad"), { remember: false }),
    ).rejects.toThrow(
      'the code must "export default" a function createPlugin(chat)',
    );
    expect(plugins.get("chat-bad")).toMatchObject({ running: false });
    expect(plugins.get("chat-bad")?.error).toContain("export default");
    await expect(
      plugins.install(entryOf("export default () => 5", "chat-five"), {
        remember: false,
      }),
    ).rejects.toThrow("createPlugin must return an object with mount");
    await expect(
      plugins.install(entryOf("setup throws", "chat-setup"), {
        remember: false,
      }),
    ).rejects.toThrow("its setup failed: no data");
    await plugins.install(entryOf("mount throws", "chat-mount"), {
      remember: false,
    });
    await expect(plugins.open("chat-mount")).rejects.toThrow(
      "its panel could not open: no panel",
    );
    // the id of a plugin not made by the chat is never taken
    await expect(
      plugins.install(entryOf(keeping, "ai-chat"), { remember: false }),
    ).rejects.toThrow(
      'another plugin, not made by the chat, has the id "ai-chat"',
    );

    plugins.dispose();
    expect(registered.size).toBe(0);
  });
});
