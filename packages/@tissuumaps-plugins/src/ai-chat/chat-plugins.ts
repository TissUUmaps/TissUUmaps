import type { Plugin, PluginRegistry, PluginStores } from "@tissuumaps/core";

import { checkPng } from "./charts";
import { makeChatShapes } from "./chat-shapes";
import { languageProblem } from "./code-language";
import { downloadFileName, downloadTypes, maxDownloadBytes } from "./downloads";
import type { GitHubOrigin } from "./github-skills";
import {
  type ChartHandle,
  type ThemeColors,
  bars,
  heatmap,
  legend,
  scatter,
  themeColors,
  watchTheme,
} from "./plugin-charts";
import { importModule } from "./plugin-import";
import { skillPackageName } from "./python-imports";
import type { PythonRun } from "./python-sandbox";
import { type ScriptOutcome, type ScriptQuestion, runScript } from "./sandbox";
import type { ChatServices } from "./services";
import type { Skill } from "./skills";
import { stateJson } from "./summary";

// The chat can make plugins for TissUUmaps: JavaScript modules that the
// model writes and the user runs, after reading their code and what they do
// (see plugin-confirm.ts). A plugin made by the chat is a real plugin: it
// runs in the page, with the stores of TissUUmaps, as plugins loaded from a
// file do. It also gets an object `chat`, with which it can run JavaScript
// and Python in the sandbox of the chat and read the tables, as the code of
// the chat does: the sandbox gets no more powers from it. Each plugin has a
// Python of its own. The ids of these plugins start with "chat-", so that
// they never replace another plugin. The user can keep a plugin in the
// browser; its code runs again only once the user says yes, once for each
// plugin each time the page is loaded.

/** Where a plugin made by the chat comes from */
export type PluginOrigin = {
  /** The skill whose code or method it uses, e.g. "anhood" */
  skill: string;
  /** Where that skill comes from, for a skill from GitHub */
  github?: GitHubOrigin;
  /** For a skill of the project, where the project keeps it */
  project?: string;
};

/** A plugin made by the chat, as the browser can remember it */
export type ChatPluginEntry = {
  /** Its id, which starts with "chat-", e.g. "chat-anhood" */
  id: string;
  /** Its name, shown in the plugins menu and on its panel */
  name: string;
  /** What it does, in a sentence or two */
  description: string;
  /** Its code: an ES module whose default export is createPlugin(chat) */
  code: string;
  /** When it was made, as an ISO date */
  made: string;
  /** What it is based on, if anything */
  origin?: PluginOrigin;
};

/** Where the plugins that the user keeps are remembered */
export type PluginStorage = {
  load: () => Promise<ChatPluginEntry[]>;
  save: (entry: ChatPluginEntry) => Promise<void>;
  remove: (id: string) => Promise<void>;
};

/** A plugin made by the chat, as the chat knows it now */
export type ChatPluginState = {
  entry: ChatPluginEntry;
  /** Whether the browser remembers it for the next sessions */
  remembered: boolean;
  /** Whether it is registered with TissUUmaps now */
  running: boolean;
  /** Why it could not run, the last time it was tried */
  error?: string;
};

/** What a plugin gets back from chat.run */
export type ChatRunResult = {
  /**
   * What the code returned: parsed from JSON when it can be (objects, lists,
   * numbers), otherwise the text
   */
  result: unknown;
  /** What the code returned, as text */
  text: string;
  /** What it logged or printed */
  logs: string[];
  /** The matplotlib figures that Python drew, as checked PNG images */
  charts: {
    png: ArrayBuffer;
    width: number;
    height: number;
    description: string;
  }[];
  /** The files that the code offered with tm.download */
  files: { name: string; text: string }[];
  /** The columns that the code saved with tm.saveColumn */
  saved: unknown[];
};

/** The plugins made by the chat */
export type ChatPlugins = {
  /** The plugins, in the order they were made */
  list: () => ChatPluginState[];
  /** One plugin, by id */
  get: (id: string) => ChatPluginState | undefined;
  /**
   * Calls a function whenever the plugins change
   *
   * @returns A function that stops calling it
   */
  onChange: (listener: () => void) => () => void;
  /** Whether the user said yes to this plugin since the page was loaded */
  isConfirmed: (id: string) => boolean;
  /**
   * Runs a plugin the user said yes to, in place of one with the same id,
   * and remembers it if asked; its panel is not opened
   *
   * @throws If its code cannot run, or another plugin has its id
   */
  install: (
    entry: ChatPluginEntry,
    options: { remember: boolean },
  ) => Promise<void>;
  /**
   * Opens the panel of a plugin the user said yes to, running it first if
   * needed
   *
   * @throws If the user has not said yes to it since the page was loaded,
   * or it cannot run or open
   */
  open: (id: string) => Promise<void>;
  /**
   * Remembers a plugin in the browser, or forgets it there
   *
   * @returns What went wrong, if anything, for the user
   */
  setRemembered: (id: string, remember: boolean) => Promise<string | undefined>;
  /** Unregisters a plugin, and forgets it */
  remove: (id: string) => Promise<void>;
  /** Why the remembered plugins could not be read, if they could not */
  storageProblem: () => string | undefined;
  /** Unregisters all of them, e.g. when the AI chat is unregistered */
  dispose: () => void;
};

/** What the chat gives the plugins it makes */
export type PluginHost = Pick<
  ChatServices,
  | "stores"
  | "readData"
  | "saveColumn"
  | "makeKernelSlot"
  | "skillList"
  | "pythonDownload"
  | "onPythonDownload"
  | "signal"
>;

/** The prefix of the ids of the plugins made by the chat */
export const pluginIdPrefix = "chat-";

/**
 * The line under the panel of every plugin made by the chat: Dia and Aid
 * are the two personalities of the chat, each the other read backwards
 */
export const pluginCredit =
  "This plugin was made with the AID of DIA, the AI agent of TissUUmaps";

/** The longest id of a plugin made by the chat */
const maxIdLength = 64;

/** The time that code of a plugin may run, by default and at most, in seconds */
const defaultTimeLimit = 120;
const maxTimeLimit = 600;

/**
 * Makes the id of a plugin made by the chat from what the model gave: small
 * letters, digits and dashes, with the prefix "chat-"
 *
 * @param given - The id the model gave, e.g. "anhood" or "chat-anhood"
 * @param name - The name of the plugin, used when no id is given
 * @returns The id, e.g. "chat-anhood", or why there is none
 */
export function pluginId(
  given: unknown,
  name: string,
): string | { error: string } {
  const raw = (typeof given === "string" && given.trim() !== "" ? given : name)
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+/, "");
  const bare = (
    raw.startsWith(pluginIdPrefix) ? raw.slice(pluginIdPrefix.length) : raw
  ).replace(/-+$/, "");
  if (bare === "") {
    return {
      error:
        'give the plugin an id of letters, digits and dashes, e.g. "anhood"',
    };
  }
  return `${pluginIdPrefix}${bare}`.slice(0, maxIdLength).replace(/-+$/, "");
}

/**
 * Where the code of a skill comes from, for the Python of a plugin: code
 * from elsewhere gets a new Python
 *
 * @param host - The chat
 * @param skill - The skill
 * @returns e.g. "github:wahlby-lab/analytical-enrichment-test@a7e9…"
 */
function skillSource(host: PluginHost, skill: Skill): string {
  const origin = host.skillList.originOf(skill);
  if (origin.kind === "github") {
    const { repo, path, commit } = origin.github;
    return `github:${repo}/${path}@${commit}`;
  }
  if (origin.kind === "project") {
    return `project:${host.stores.projectStore.getState().instanceId}:${origin.where}/${skill.name}`;
  }
  return `added:${skill.name}`;
}

/**
 * The text of the file that "Export .js" saves: the code, after a few lines
 * that say what it is, which "Add plugin file…" reads back
 *
 * @param entry - The plugin
 * @returns The text of the file
 */
export function pluginFileText(entry: ChatPluginEntry): string {
  const oneLine = (text: string) => text.replace(/\s+/g, " ").trim();
  const github = entry.origin?.github;
  return [
    "// TissUUmaps AI chat plugin",
    `// id: ${entry.id}`,
    `// name: ${oneLine(entry.name)}`,
    `// description: ${oneLine(entry.description)}`,
    `// made: ${entry.made}`,
    `// ${pluginCredit}.`,
    ...(entry.origin === undefined
      ? []
      : [
          `// based on the skill: ${entry.origin.skill}` +
            (github === undefined
              ? ""
              : ` (github.com/${github.repo}${github.path === "" ? "" : `/${github.path}`}, commit ${github.commit}, license: ${github.license ?? "none"})`),
        ]),
    "// Made by the AI chat of TissUUmaps. It needs the AI chat: to use it again, open",
    '// the settings (⚙) of the AI chat and click "Add plugin file…" under "Plugins made',
    '// by the chat". Run only plugins whose code you trust.',
    "",
    entry.code,
  ].join("\n");
}

/**
 * Reads a file that "Export .js" saved, or another module of a plugin
 * made for the chat
 *
 * @param text - The text of the file
 * @param fileName - The name of the file, for the id when the file has none
 * @returns The plugin, not yet confirmed, or why the file is not one
 */
export function readPluginFile(
  text: string,
  fileName: string,
): ChatPluginEntry | { error: string } {
  const header = new Map<string, string>();
  for (const line of text.split(/\r?\n/).slice(0, 20)) {
    const field = /^\/\/ (id|name|description|made): (.*)$/.exec(line);
    if (field !== null) {
      header.set(field[1]!, field[2]!.trim());
    }
  }
  if (!/\bexport\s+default\b/.test(text)) {
    return {
      error: `${fileName} is not a plugin for the AI chat: it has no "export default"`,
    };
  }
  const name = header.get("name") ?? fileName.replace(/\.(m?js)$/i, "").trim();
  const id = pluginId(header.get("id"), name || "plugin");
  if (typeof id !== "string") {
    return { error: id.error };
  }
  return {
    id,
    name: name || id,
    description: header.get("description") ?? `From the file ${fileName}.`,
    code: text,
    made: header.get("made") ?? new Date().toISOString(),
  };
}

/**
 * The plugin registry of TissUUmaps, which it puts on `window`
 *
 * @returns The registry, if TissUUmaps has started
 */
function windowRegistry(): PluginRegistry | undefined {
  return (window as { tissuumaps?: PluginRegistry }).tissuumaps;
}

/**
 * The thin bar with {@link pluginCredit} at the bottom of the panel of a
 * plugin, in the style of the credits of the chat: it stays at the bottom
 * while the panel scrolls
 *
 * @returns The bar
 */
function creditLine(): HTMLElement {
  const bar = document.createElement("div");
  bar.dataset.chatCredit = "true";
  bar.textContent = pluginCredit;
  bar.style.cssText =
    "margin-top: auto; position: sticky; bottom: 0; flex-shrink: 0; padding: 2px 4px;" +
    "border-top: 1px solid var(--border); background: var(--background);" +
    "color: var(--muted-foreground); font-size: 11px; text-align: right;";
  return bar;
}

/**
 * The message of an error, or what was thrown as text
 *
 * @param error - What was thrown
 * @returns The message
 */
function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Parses what code returned, when it is JSON
 *
 * @param text - What it returned, as text
 * @returns The value, or the text itself
 */
function parseResult(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

/**
 * Makes the plugins made by the chat
 *
 * @param host - What the chat gives them
 * @param storage - Where the plugins the user keeps are remembered
 * @param options - For the tests: how the code is imported, and the
 * registry of the plugins
 * @returns The plugins
 */
export function makeChatPlugins(
  host: PluginHost,
  storage: PluginStorage,
  options?: {
    load?: (code: string) => Promise<unknown>;
    registry?: () => PluginRegistry | undefined;
  },
): ChatPlugins {
  const load = options?.load ?? importModule;
  const registry = options?.registry ?? windowRegistry;
  const { projectStore, appStore } = host.stores;
  const states = new Map<string, ChatPluginState>();
  /** The plugins that the user said yes to, since the page was loaded */
  const confirmed = new Set<string>();
  /** What stops each running plugin's runs and Python */
  const running = new Map<string, () => void>();
  const listeners = new Set<() => void>();
  let problem: string | undefined;

  const changed = () => {
    for (const listener of listeners) {
      listener();
    }
  };

  /**
   * Makes the object `chat` of one plugin
   *
   * @param entry - The plugin
   * @returns The object, and what stops its runs and its Python
   */
  function makeBridge(entry: ChatPluginEntry): {
    bridge: object;
    dispose: () => void;
  } {
    const stopAll = new AbortController();
    const python = host.makeKernelSlot();
    /** What stops watching the theme for the plugin, stopped with it */
    const themeWatches = new Set<() => void>();
    /** The charts of the plugin, removed with it */
    const charts = new Set<ChartHandle>();
    /** The shapes that it shows on the image, removed with it */
    const shapes = makeChatShapes(projectStore, entry.id);
    const forget = (chart: ChartHandle) => charts.delete(chart);
    const chart =
      (make: typeof heatmap) => (element: unknown, options: unknown) => {
        const made = make(element, options, forget);
        charts.add(made);
        return made;
      };
    // one Python run at a time: its kernel runs one code at a time
    let pythonQueue: Promise<unknown> = Promise.resolve();

    async function run(given: unknown): Promise<ChatRunResult> {
      stopAll.signal.throwIfAborted();
      const options = (
        typeof given === "object" && given !== null ? given : {}
      ) as Record<string, unknown>;
      const language = options["language"] ?? "javascript";
      if (language !== "javascript" && language !== "python") {
        throw new Error(
          `chat.run runs "javascript" or "python", not ${JSON.stringify(language)}`,
        );
      }
      const code = options["code"];
      if (typeof code !== "string" || code.trim() === "") {
        throw new Error(
          language === "python"
            ? "give the code: Python with a function async def main(tm): ..."
            : "give the code: a module, export default async function (tm) { ... }",
        );
      }
      const mismatch = languageProblem(code, language);
      if (mismatch !== undefined) {
        throw new Error(mismatch);
      }
      let skill: Skill | undefined;
      if (options["skill"] !== undefined) {
        const wanted =
          typeof options["skill"] === "string"
            ? options["skill"]
            : JSON.stringify(options["skill"]);
        skill = host.skillList
          .activeSkills()
          .find(({ name }) => name === wanted);
        if (skill === undefined) {
          throw new Error(
            `the skill "${wanted}" is not on in the AI chat: add it (e.g. from GitHub) or turn it on in the settings (⚙) of the AI chat`,
          );
        }
      }
      const seconds = Number(options["timeLimit"] ?? defaultTimeLimit);
      const timeLimit =
        Math.min(
          maxTimeLimit,
          Math.max(1, Number.isFinite(seconds) ? seconds : defaultTimeLimit),
        ) * 1000;
      const outer = options["signal"];
      const signal =
        outer instanceof AbortSignal
          ? AbortSignal.any([outer, stopAll.signal])
          : stopAll.signal;
      const onProgress =
        typeof options["onProgress"] === "function"
          ? (options["onProgress"] as (text: string) => void)
          : undefined;
      const files: ChatRunResult["files"] = [];
      const saved: unknown[] = [];

      const answer = async (question: ScriptQuestion): Promise<unknown> => {
        signal.throwIfAborted();
        switch (question.kind) {
          case "tables":
          case "columns":
          case "values":
            return host.readData(question, { signal });
          case "file": {
            const file = skill?.files.get(question.path);
            if (file === undefined) {
              throw new Error(
                skill === undefined
                  ? "only code run with a skill has files to read"
                  : `the skill has no file "${question.path}"`,
              );
            }
            return file.text();
          }
          case "save_column": {
            const column = await host.saveColumn(question, { signal });
            saved.push(column.result);
            return column.result;
          }
          case "download": {
            const checked = downloadFileName(question.name);
            if ("error" in checked) {
              throw new Error(checked.error);
            }
            if (typeof question.text !== "string") {
              throw new Error(
                "give the text of the file, e.g. the CSV as a string",
              );
            }
            const ending = /\.[^.]+$/.exec(checked.name)![0].toLowerCase();
            const bytes = new Blob([question.text], {
              type: downloadTypes[ending]!,
            }).size;
            if (bytes > maxDownloadBytes) {
              throw new Error(
                `the file is too big (${(bytes / 1e6).toFixed(1)} MB, at most ${maxDownloadBytes / 1e6} MB)`,
              );
            }
            files.push({ name: checked.name, text: question.text });
            return { name: checked.name, bytes };
          }
        }
      };
      const project = JSON.parse(
        stateJson(projectStore.getState(), appStore.getState()),
      ) as unknown;
      const args = options["args"] ?? {};

      let outcome: ScriptOutcome;
      if (language === "python") {
        const runPython = async (): Promise<ScriptOutcome> => {
          const skillFiles =
            skill === undefined
              ? undefined
              : Object.fromEntries(
                  await Promise.all(
                    [...skill.files].map(
                      async ([path, content]) =>
                        [path, await content.arrayBuffer()] as const,
                    ),
                  ),
                );
          const megabytes = (bytes: number) =>
            bytes < 10e6 ? (bytes / 1e6).toFixed(1) : Math.round(bytes / 1e6);
          const stopListening = host.onPythonDownload(() => {
            const download = host.pythonDownload();
            if (download !== undefined) {
              onProgress?.(
                `Downloading Python, ${megabytes(download.done)} of ${megabytes(download.total)} MB`,
              );
            }
          });
          try {
            onProgress?.("starting Python");
            const kernel = await python.kernel(
              `plugin:${entry.id}:${skill === undefined ? "" : skillSource(host, skill)}`,
              { signal },
            );
            onProgress?.("loading the Python packages");
            const pythonRun: PythonRun = {
              code,
              args,
              project,
              ...(skillFiles === undefined ? {} : { files: skillFiles }),
              ...(skill === undefined || skillPackageName(skill) === undefined
                ? {}
                : { packageName: skillPackageName(skill)! }),
            };
            return await kernel.run(pythonRun, answer, {
              timeLimit,
              signal,
              ...(onProgress === undefined ? {} : { onProgress }),
              onRunning: () => onProgress?.("running"),
            });
          } finally {
            stopListening();
          }
        };
        const queued = pythonQueue.then(runPython, runPython);
        pythonQueue = queued.catch(() => undefined);
        outcome = await queued;
      } else {
        outcome = await runScript({ code, args, project }, answer, {
          timeLimit,
          signal,
          ...(onProgress === undefined ? {} : { onProgress }),
        });
      }
      if (outcome.error !== undefined) {
        const error = new Error(outcome.error) as Error & { logs?: string[] };
        error.logs = outcome.logs;
        throw error;
      }
      const charts: ChatRunResult["charts"] = [];
      for (const chart of outcome.charts ?? []) {
        const checked = checkPng(chart.png);
        if (!("error" in checked)) {
          charts.push({ ...chart, ...checked });
        }
      }
      return {
        result: parseResult(outcome.result ?? ""),
        text: outcome.result ?? "",
        logs: outcome.logs,
        charts,
        files,
        saved,
      };
    }

    const bridge = Object.freeze({
      pluginId: entry.id,
      run,
      tables: () => host.readData({ kind: "tables" }),
      columns: (table: string, matrix?: string) =>
        host.readData({
          kind: "columns",
          table,
          ...(matrix === undefined ? {} : { matrix }),
        }),
      read: (table: string, column: string) =>
        host.readData({ kind: "values", table, column }),
      saveColumn: async (
        table: string,
        column: string,
        values: unknown,
        more?: { ids?: unknown },
      ) =>
        (
          await host.saveColumn({
            kind: "save_column",
            table,
            column,
            values,
            ...(more?.ids === undefined ? {} : { ids: more.ids }),
          })
        ).result,
      skills: () =>
        host.skillList
          .activeSkills()
          .map(({ name, description }) => ({ name, description })),
      theme: themeColors,
      onThemeChange: (listener: (colors: ThemeColors) => void) => {
        const stop = watchTheme(listener);
        themeWatches.add(stop);
        return () => {
          stop();
          themeWatches.delete(stop);
        };
      },
      heatmap: chart(heatmap),
      bars: chart(bars),
      scatter: chart(scatter),
      legend,
      showShapes: (options: unknown) => shapes.show(options),
      hideShapes: (id?: unknown) => shapes.hide(id),
    });
    return {
      bridge,
      dispose: () => {
        stopAll.abort(new Error(`the plugin "${entry.name}" was closed`));
        python.stop();
        for (const stop of themeWatches) {
          stop();
        }
        themeWatches.clear();
        for (const made of [...charts]) {
          made.destroy();
        }
        shapes.dispose();
      },
    };
  }

  /**
   * Imports the code of a plugin and registers it with TissUUmaps, in place
   * of an earlier version of it
   *
   * @param entry - The plugin
   * @throws If its code cannot run, or another plugin has its id
   */
  async function register(entry: ChatPluginEntry): Promise<void> {
    const plugins = registry();
    if (plugins === undefined) {
      throw new Error(
        "TissUUmaps has not started its plugins yet (window.tissuumaps)",
      );
    }
    const taken = appStore.getState().plugins?.has(entry.id) ?? false;
    if (taken && !running.has(entry.id)) {
      throw new Error(
        `another plugin, not made by the chat, has the id "${entry.id}"`,
      );
    }
    const module = (await load(entry.code)) as { default?: unknown } | null;
    const create = module?.default;
    if (typeof create !== "function") {
      throw new Error(
        'the code must "export default" a function createPlugin(chat) that returns the plugin, e.g. export default function createPlugin(chat) { return { mount(container, stores) { ... } }; }',
      );
    }
    const { bridge, dispose } = makeBridge(entry);
    let made: unknown;
    try {
      made = await (create as (chat: object) => unknown)(bridge);
    } catch (error) {
      dispose();
      throw new Error(`createPlugin failed: ${messageOf(error)}`, {
        cause: error,
      });
    }
    const parts = (typeof made === "object" && made !== null ? made : {}) as {
      setup?: unknown;
      mount?: unknown;
    };
    const { setup, mount } = parts;
    if (
      (setup !== undefined && typeof setup !== "function") ||
      (mount !== undefined && typeof mount !== "function") ||
      (setup === undefined && mount === undefined)
    ) {
      dispose();
      throw new Error(
        "createPlugin must return an object with mount(container, stores) for a panel, and/or setup(stores); each may return a function that cleans up",
      );
    }
    let setupError: unknown;
    const plugin: Plugin = {
      id: entry.id,
      name: entry.name,
      setup: (stores: PluginStores) => {
        let teardown: unknown;
        try {
          teardown = (setup as Plugin["setup"])?.(stores);
        } catch (error) {
          setupError = error;
          dispose();
          throw error;
        }
        return () => {
          running.delete(entry.id);
          try {
            if (typeof teardown === "function") {
              (teardown as () => void)();
            }
          } finally {
            dispose();
            const state = states.get(entry.id);
            if (state !== undefined && state.entry === entry) {
              state.running = false;
              changed();
            }
          }
        };
      },
      ...(mount === undefined
        ? {}
        : {
            mount: (container: HTMLElement, stores: PluginStores) => {
              let unmount: (() => void) | void;
              try {
                unmount = (mount as NonNullable<Plugin["mount"]>)(
                  container,
                  stores,
                );
              } catch (error) {
                mountErrors.set(entry.id, error);
                throw error;
              }
              container.append(creditLine());
              return unmount;
            },
          }),
    };
    plugins.registerPlugin(plugin);
    if (setupError !== undefined) {
      throw new Error(`its setup failed: ${messageOf(setupError)}`, {
        cause: setupError,
      });
    }
    running.set(entry.id, dispose);
  }

  /** Why the mount of each plugin failed, the last time */
  const mountErrors = new Map<string, unknown>();

  async function install(
    entry: ChatPluginEntry,
    { remember }: { remember: boolean },
  ): Promise<void> {
    confirmed.add(entry.id);
    const state: ChatPluginState = states.get(entry.id) ?? {
      entry,
      remembered: false,
      running: false,
    };
    state.entry = entry;
    states.set(entry.id, state);
    try {
      await register(entry);
      state.running = true;
      delete state.error;
    } catch (error) {
      state.running = false;
      state.error = error instanceof Error ? error.message : String(error);
      changed();
      throw error;
    }
    if (remember || state.remembered) {
      await setRemembered(entry.id, true);
    }
    changed();
  }

  async function open(id: string): Promise<void> {
    const state = states.get(id);
    if (state === undefined) {
      throw new Error(`there is no plugin "${id}" made by the chat`);
    }
    if (!confirmed.has(id)) {
      throw new Error(
        `the user has not said yes to the plugin "${state.entry.name}" since the page was loaded`,
      );
    }
    if (!running.has(id)) {
      await install(state.entry, { remember: state.remembered });
    }
    mountErrors.delete(id);
    registry()?.mountPlugin(id);
    if (mountErrors.has(id)) {
      const error = mountErrors.get(id);
      throw new Error(
        `its panel could not open: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async function setRemembered(
    id: string,
    remember: boolean,
  ): Promise<string | undefined> {
    const state = states.get(id);
    if (state === undefined) {
      return undefined;
    }
    try {
      if (remember) {
        await storage.save(state.entry);
      } else {
        await storage.remove(id);
      }
      state.remembered = remember;
      return undefined;
    } catch (error) {
      state.remembered = false;
      return `${state.entry.name} is kept for this session only: ${error instanceof Error ? error.message : String(error)}`;
    } finally {
      changed();
    }
  }

  async function remove(id: string): Promise<void> {
    const state = states.get(id);
    if (running.has(id)) {
      registry()?.unregisterPlugin(id);
      // without the registry, at least its runs and Python stop
      running.get(id)?.();
      running.delete(id);
    }
    states.delete(id);
    confirmed.delete(id);
    changed();
    if (state?.remembered) {
      await storage.remove(id);
    }
  }

  // the remembered plugins are listed, but their code runs only once the
  // user says yes
  storage.load().then(
    (entries) => {
      for (const entry of entries) {
        if (!states.has(entry.id)) {
          states.set(entry.id, { entry, remembered: true, running: false });
        }
      }
      changed();
    },
    (error: unknown) => {
      problem = `The remembered plugins could not be read: ${error instanceof Error ? error.message : String(error)}`;
      changed();
    },
  );

  return {
    list: () => [...states.values()],
    get: (id) => states.get(id),
    onChange: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    isConfirmed: (id) => confirmed.has(id),
    install,
    open,
    setRemembered,
    remove,
    storageProblem: () => problem,
    dispose: () => {
      for (const id of [...running.keys()]) {
        registry()?.unregisterPlugin(id);
        running.get(id)?.();
      }
      running.clear();
      listeners.clear();
    },
  };
}
