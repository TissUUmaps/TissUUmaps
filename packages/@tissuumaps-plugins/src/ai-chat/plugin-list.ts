import {
  type ChatPluginEntry,
  type ChatPluginState,
  type ChatPlugins,
  pluginFileText,
  readPluginFile,
} from "./chat-plugins";
import { makeButton, quietTextStyle, smallButtonStyle } from "./ui";

// The section "Plugins made by the chat" of the settings: each plugin with
// Open, Show code, Remember, Remove and Export .js, and "Add plugin file…"
// for a plugin exported before. The plugins themselves live in the setup of
// the chat (see chat-plugins.ts); this list only shows them, while the panel
// is open.

/** What the buttons of the list do that needs the chat */
export type PluginListActions = {
  /**
   * Asks the user in the chat whether to run a plugin, and runs and opens
   * it on yes
   *
   * @param entry - The plugin
   * @param kind - A plugin made before, or one from a file
   */
  askToRun: (entry: ChatPluginEntry, kind: "remembered" | "file") => void;
  /** Shows the code of a plugin in the chat */
  showCode: (entry: ChatPluginEntry) => void;
};

/**
 * Says what a plugin is doing now, in a few words
 *
 * @param state - The plugin
 * @param confirmed - Whether the user said yes to it since the page was loaded
 * @returns e.g. "running"
 */
function stateText(state: ChatPluginState, confirmed: boolean): string {
  if (state.error !== undefined) {
    return `did not run: ${state.error}`;
  }
  if (state.running) {
    return "running";
  }
  return confirmed ? "not running" : "not running: Open asks you first";
}

/**
 * Saves a text as a file, through a link that the browser downloads
 *
 * @param name - The name of the file
 * @param text - Its text
 */
function saveText(name: string, text: string): void {
  const url = URL.createObjectURL(
    new Blob([text], { type: "text/javascript" }),
  );
  const link = document.createElement("a");
  link.download = name;
  link.href = url;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

/**
 * Creates the section "Plugins made by the chat" of the settings
 *
 * @param plugins - The plugins made by the chat
 * @param actions - What needs the chat
 * @returns The section, and a function that stops updating it
 */
export function makePluginList(
  plugins: ChatPlugins,
  actions: PluginListActions,
): { element: HTMLElement; dispose: () => void } {
  const title = document.createElement("strong");
  title.textContent = "Plugins made by the chat";
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = ".js,.mjs,text/javascript";
  fileInput.style.display = "none";
  const addButton = makeButton(
    "Add plugin file…",
    () => fileInput.click(),
    smallButtonStyle,
  );
  addButton.title =
    'Run a plugin that was exported with "Export .js"; you see its code first';
  const header = document.createElement("div");
  header.style.cssText =
    "display: flex; flex-wrap: wrap; gap: 8px; align-items: center;";
  header.append(title, addButton, fileInput);

  const list = document.createElement("div");
  list.style.cssText = "display: flex; flex-direction: column; gap: 6px;";
  const status = document.createElement("div");
  status.setAttribute("role", "status");
  status.style.cssText = `${quietTextStyle}white-space: pre-line;`;
  const element = document.createElement("div");
  element.style.cssText = "display: flex; flex-direction: column; gap: 6px;";
  element.append(header, list, status);

  /**
   * Says what happened, under the list
   *
   * @param text - What happened
   */
  function report(text: string): void {
    status.textContent = text;
  }

  /**
   * Creates the row of a plugin
   *
   * @param state - The plugin
   * @returns The row
   */
  function makeRow(state: ChatPluginState): HTMLElement {
    const { entry } = state;
    const name = document.createElement("span");
    name.textContent = entry.name;
    name.title = entry.id;
    name.style.fontWeight = "600";
    const what = document.createElement("span");
    what.textContent = stateText(state, plugins.isConfirmed(entry.id));
    what.style.cssText = quietTextStyle;

    const openButton = makeButton(
      "Open",
      () => {
        if (!plugins.isConfirmed(entry.id)) {
          report(`${entry.name}: answer the question in the chat to run it.`);
          actions.askToRun(entry, "remembered");
          return;
        }
        plugins.open(entry.id).then(
          () => report(`${entry.name} is open.`),
          (error: unknown) =>
            report(
              `${entry.name} could not open: ${error instanceof Error ? error.message : String(error)}`,
            ),
        );
      },
      smallButtonStyle,
    );
    openButton.title = "Open the panel of the plugin";
    const codeButton = makeButton(
      "Show code",
      () => actions.showCode(entry),
      smallButtonStyle,
    );
    codeButton.title = "Show the code of the plugin in the chat";

    const rememberBox = document.createElement("input");
    rememberBox.type = "checkbox";
    rememberBox.checked = state.remembered;
    rememberBox.style.accentColor = "var(--primary)";
    rememberBox.addEventListener("change", () => {
      const on = rememberBox.checked;
      void plugins
        .setRemembered(entry.id, on)
        .then((problem) =>
          report(
            problem ??
              (on
                ? `${entry.name} is remembered by this browser; you are asked before it runs in another session.`
                : `${entry.name} is kept for this session only.`),
          ),
        );
    });
    const rememberLabel = document.createElement("label");
    rememberLabel.style.cssText = `display: flex; gap: 4px; align-items: center;${quietTextStyle}`;
    rememberLabel.title =
      "Keep the plugin in this browser for the next sessions; you are asked again before it runs";
    rememberLabel.append(rememberBox, "Remember");

    const removeButton = makeButton(
      "Remove",
      () => {
        const question = `Remove the plugin "${entry.name}"? Its panel closes${state.remembered ? ", and the browser forgets it" : ""}. What it changed in the project stays.`;
        if (!window.confirm(question)) {
          return;
        }
        plugins.remove(entry.id).then(
          () => report(`${entry.name} removed.`),
          (error: unknown) =>
            report(
              `${entry.name} removed, but the browser could not forget it: ${error instanceof Error ? error.message : String(error)}`,
            ),
        );
      },
      smallButtonStyle,
    );
    removeButton.setAttribute("aria-label", `Remove ${entry.name}`);
    const exportButton = makeButton(
      "Export .js",
      () => {
        saveText(`${entry.id}.js`, pluginFileText(entry));
        report(
          `${entry.id}.js saved. "Add plugin file…" runs it again, here or in another browser.`,
        );
      },
      smallButtonStyle,
    );
    exportButton.title =
      "Save the plugin as a .js file, to keep it or to share it";

    const top = document.createElement("div");
    top.style.cssText =
      "display: flex; flex-wrap: wrap; gap: 6px 8px; align-items: center;";
    top.append(
      name,
      what,
      openButton,
      codeButton,
      rememberLabel,
      removeButton,
      exportButton,
    );
    const description = document.createElement("div");
    description.textContent = entry.description;
    description.style.cssText = quietTextStyle;
    const row = document.createElement("div");
    row.dataset.plugin = entry.id;
    row.style.cssText = "display: flex; flex-direction: column; gap: 2px;";
    row.append(top, description);
    return row;
  }

  /** Shows the list again */
  function render(): void {
    const all = plugins.list();
    const problem = plugins.storageProblem();
    if (all.length === 0) {
      const empty = document.createElement("div");
      empty.style.cssText = quietTextStyle;
      empty.textContent =
        'No plugins yet. Ask the chat to make one, e.g. "make me a plugin with a slider for the opacity of the cells".';
      list.replaceChildren(empty);
    } else {
      list.replaceChildren(...all.map(makeRow));
    }
    if (problem !== undefined && status.textContent === "") {
      report(problem);
    }
  }

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    fileInput.value = "";
    if (file === undefined) {
      return;
    }
    void file.text().then((text) => {
      const entry = readPluginFile(text, file.name);
      if ("error" in entry) {
        report(entry.error);
        return;
      }
      report(`${entry.name}: answer the question in the chat to run it.`);
      actions.askToRun(entry, "file");
    });
  });

  const stop = plugins.onChange(render);
  render();
  return { element, dispose: stop };
}
