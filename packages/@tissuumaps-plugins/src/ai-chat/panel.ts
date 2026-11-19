import {
  type ProjectStore,
  SourceUtils,
  type TableDataSource,
} from "@tissuumaps/core";

import { callActionDefinition, describeAction } from "./call-action";
import { chartFileName, checkPng, maxCharts, toBase64 } from "./charts";
import type { ChatPluginEntry, PluginOrigin } from "./chat-plugins";
import {
  codeInstructions,
  lookAtChartDefinition,
  runCodeDefinition,
  saveComputedTableDefinition,
} from "./code-tool";
import { overlapWarning } from "./color-tools";
import {
  type ComputedTable,
  type ComputedValues,
  computedTableCsv,
  computedTableType,
  csvFileName,
  idsSurviveCsv,
  suggestCsvName,
  versionOf,
} from "./computed-tables";
import { downloadFileName, downloadTypes, maxDownloadBytes } from "./downloads";
import { type GitHubPreview, readGitHubSkill } from "./github-skills";
import { aboutText, helpText } from "./help";
import {
  actionLimits,
  instructions,
  toolLimits,
  wholeAppInstructions,
} from "./instructions";
import { renderMarkdown } from "./markdown";
import { findById, isMapKind, same } from "./objects";
import { personaFor } from "./persona";
import { type StoreActions, actionsOf, readPluginCode } from "./plugin-code";
import { type PluginQuestionKind, confirmPlugin } from "./plugin-confirm";
import { makePluginList } from "./plugin-list";
import {
  describePlugins,
  pluginInstructions,
  proposePluginDefinition,
  readPluginDefinition,
} from "./plugin-tools";
import {
  type ChatAnswer,
  type ChatMessage,
  type OllamaPullUpdate,
  type Provider,
  type ProviderId,
  type Source,
  type Usage,
  explainNotReachable,
  hideKey,
  notRunningMessage,
  ollamaUrl,
  providers,
  readJsonLines,
} from "./providers";
import { countPythonFiles, forgetPythonFiles, lockFile } from "./python-files";
import {
  type PyodidePackages,
  pyodidePackages,
  skillPackageName,
} from "./python-imports";
import { type ScriptOutcome, type ScriptQuestion, runScript } from "./sandbox";
import type { ChatServices } from "./services";
import { loadSetting, saveSetting } from "./settings";
import { confirmGitHubSkill, confirmSkillMd } from "./skill-confirm";
import {
  describeSkills,
  skillInstructions,
  skillToolDefinitions,
} from "./skill-tools";
import type { Skill } from "./skills";
import { buildContext, exampleRequests, stateJson } from "./summary";
import {
  describeMatrix,
  describeValues,
  findCoordinates,
  findUsedColumns,
  linkedTables,
  listColumns,
  maxRowsToRead,
  objectTables,
} from "./tables";
import { imageTools, toolDefinitions, tools, wholeAppTools } from "./tools";
import type {
  Change,
  ColumnInfo,
  ItemsObject,
  RunOptions,
  TableInfo,
  TableKnowledge,
  ToolArguments,
} from "./types";
import {
  type MessageKind,
  controlStyle,
  credits,
  describeCall,
  dollars,
  firstLine,
  makeAvatar,
  makeButton,
  makeCopyButton,
  makeThinkingIndicator,
  messageStyles,
  notReachable,
  quietButtonStyle,
  quietTextStyle,
  smallButtonStyle,
  toolIcons,
  usesOllama,
} from "./ui";
import { currentValue, describeValueChange, handNote, takeBack } from "./undo";
import {
  describeSearches,
  makeSourcesRow,
  webInstructions,
  webSearchState,
} from "./web";

/**
 * How many times in a row the strong models of OpenAI and Anthropic may call
 * tools in one request before they must answer: a skill often takes several
 * steps, e.g. to load it, read its files, find the columns and run the code
 */
const maxSteps = 15;

/** The same for the small local models of Ollama */
const maxStepsLocal = 5;

/** Told to the model with the results of its last step of tools */
const lastStepNote =
  "(Note from the chat: this was the last step of tools for this request. Now answer the user without calling tools, with what you found: the results, what is still missing, and what the user can ask next to go on.)";

/** The longest time a script of a skill may run, in milliseconds */
const scriptTimeLimit = 120_000;

/** The longest result of a script that the model gets, in characters */
const maxScriptText = 20_000;

/**
 * Mounts the chat panel
 *
 * @param container - The element to mount the panel into
 * @param services - What the chat keeps while TissUUmaps runs, with the
 * stores of the application
 * @returns A callback that unmounts the panel
 */
export function mountAiChat(
  container: HTMLElement,
  services: ChatServices,
): () => void {
  const { projectStore, settingsStore, appStore } = services.stores;
  // with the tables computed by the chat, which the tools must find at once
  const { dataStore, skillList, pythonFiles } = services;
  // aborted on unmount, which stops the requests to the model and the table reads
  const abortController = new AbortController();
  const { signal } = abortController;

  /** The conversation so far, without the system message; sent in full each time */
  const messages: ChatMessage[] = [];

  /** The changes made by the chat, one list per request, the newest last */
  const undoStack: Change[][] = [];

  const tables: TableKnowledge = { info: new Map(), remembered: new Map() };
  // tables are read one after the other, not all at once
  let readingQueue = Promise.resolve();

  // what Ollama said when it was last asked
  let connected = false;
  let installed: string[] = [];
  let unreachable = notRunningMessage;

  /**
   * The API keys of OpenAI and Anthropic, by service: typed in this session or
   * remembered in the browser. They are only sent to their own service, and
   * never shown or logged.
   */
  const keys = new Map<ProviderId, string>();
  for (const provider of providers) {
    const key = loadSetting(`key.${provider.id}`);
    if (key !== null) {
      keys.set(provider.id, key);
    }
  }
  /** What the last check of each service's key found, for the status line */
  const keyStatus = new Map<ProviderId, string>();
  /** The services whose key was accepted by the last check */
  const acceptedKeys = new Set<ProviderId>();
  /** The services for which the chat already said what a key means */
  const announcedKeys = new Set<ProviderId>();
  /** Whether the settings are shown although the key works */
  let settingsOpen = false;

  /** The personality of the assistant, which follows the theme */
  let persona = personaFor(settingsStore.getState().dark);
  /** The indicator shown in the chat while the model thinks */
  let thinking: { element: HTMLElement; stop: () => void } | null = null;
  /** The message added last to the history, e.g. an answer of the model */
  let lastFrame: HTMLElement | null = null;

  /** About how much the answers cost since the panel was opened, in dollars */
  let sessionCost = 0;

  /** The messages sent so far, for the up and down arrows */
  const sent: string[] = [];
  /** Which sent message the text box shows; `sent.length` for the new text */
  let sentIndex = 0;
  /** The new text, kept while the arrows show sent messages */
  let draft = "";

  const providerSelect = document.createElement("select");
  providerSelect.title = "Service";
  providerSelect.style.cssText = controlStyle;
  for (const provider of providers) {
    const option = document.createElement("option");
    option.value = provider.id;
    option.textContent = provider.name;
    providerSelect.append(option);
  }
  const savedProvider = loadSetting("provider");
  if (providers.some((provider) => provider.id === savedProvider)) {
    providerSelect.value = savedProvider ?? "";
  }
  const modelSelect = document.createElement("select");
  modelSelect.title = "Model";
  modelSelect.style.cssText = controlStyle;
  const downloadButton = makeButton("Download", () => void download());
  const serviceRow = document.createElement("div");
  serviceRow.style.cssText = "display: flex; flex-wrap: wrap; gap: 8px;";
  serviceRow.append(providerSelect, modelSelect, downloadButton);

  // small buttons, so that they fit in one row of a narrow panel
  const newChatButton = makeButton("New chat", newChat, smallButtonStyle);
  const contextButton = makeButton(
    "Show context",
    () =>
      show(
        "Context sent to the model",
        `${instructionsFor(currentProvider())}\n\n${contextFor(currentProvider())}`,
        "info",
        true,
      ),
    smallButtonStyle,
  );
  const helpButton = makeButton(
    "Help",
    () => show("Help", helpText),
    smallButtonStyle,
  );
  const aboutButton = makeButton("About", () => showAbout(), smallButtonStyle);
  aboutButton.title = "Where the names Dia and Aid come from";
  const undoButton = makeButton("Undo", undo, smallButtonStyle);
  const undoAllButton = makeButton("Undo all", undoAll, smallButtonStyle);
  const settingsButton = makeButton("⚙", toggleSettings, smallButtonStyle);
  settingsButton.title = "Settings";
  settingsButton.setAttribute("aria-label", "Settings");
  settingsButton.style.marginLeft = "auto";
  const topRow = document.createElement("div");
  topRow.style.cssText =
    "display: flex; flex-wrap: wrap; gap: 6px; align-items: center;";
  topRow.append(
    newChatButton,
    contextButton,
    helpButton,
    aboutButton,
    undoButton,
    undoAllButton,
    settingsButton,
  );

  // the short line shown instead of the settings, while the key works
  const summary = document.createElement("div");
  summary.style.cssText = quietTextStyle;

  // a password field, so that the key is not shown on the screen
  const keyInput = document.createElement("input");
  keyInput.type = "password";
  keyInput.autocomplete = "off";
  keyInput.spellcheck = false;
  keyInput.style.cssText = `${controlStyle}flex: 1; min-width: 12em;`;
  const keyLabel = document.createElement("label");
  keyLabel.style.cssText =
    "display: flex; flex: 1; gap: 6px; align-items: center;";
  keyLabel.append("API key", keyInput);
  const rememberBox = document.createElement("input");
  rememberBox.type = "checkbox";
  rememberBox.style.accentColor = "var(--primary)";
  const rememberLabel = document.createElement("label");
  rememberLabel.append(rememberBox, " Remember key on this computer");
  const forgetButton = makeButton("Forget key", forgetKey);
  const keyRow = document.createElement("div");
  keyRow.style.cssText =
    "display: flex; flex-wrap: wrap; gap: 8px; align-items: center;";
  keyRow.append(keyLabel, rememberLabel, forgetButton);

  /** Whether the rows of the tools are open, remembered by the browser */
  let showToolDetails = loadSetting("toolDetails") === "true";
  const toolDetailsBox = document.createElement("input");
  toolDetailsBox.type = "checkbox";
  toolDetailsBox.checked = showToolDetails;
  toolDetailsBox.style.accentColor = "var(--primary)";
  toolDetailsBox.addEventListener("change", () => {
    showToolDetails = toolDetailsBox.checked;
    saveSetting("toolDetails", showToolDetails ? "true" : null);
    updateToolRows();
  });
  const toolDetailsLabel = document.createElement("label");
  toolDetailsLabel.title =
    "Show the calls and the results of the tools, instead of one line each";
  toolDetailsLabel.append(toolDetailsBox, " Show tool details");

  /**
   * Makes a checkbox of the settings that is off unless the user turns it
   * on, remembered by the browser
   *
   * @param setting - Its name in the browser's memory
   * @param text - Its label
   * @param title - Its tooltip
   * @param onChange - Called with the new value
   * @returns The label with the checkbox
   */
  function makeOption(
    setting: string,
    text: string,
    title: string,
    onChange: (on: boolean) => void,
  ): HTMLLabelElement {
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = loadSetting(setting) === "true";
    box.style.accentColor = "var(--primary)";
    box.addEventListener("change", () => {
      saveSetting(setting, box.checked ? "true" : null);
      onChange(box.checked);
    });
    const label = document.createElement("label");
    label.title = title;
    label.append(box, ` ${text}`);
    return label;
  }

  /**
   * Whether the model gets the image of every chart that its code draws,
   * remembered by the browser; without it, only when it looks at one
   */
  let seeCharts = loadSetting("seeCharts") === "true";
  const seeChartsLabel = makeOption(
    "seeCharts",
    "Let the model see charts",
    "With OpenAI and Anthropic: send the image of every chart to the model, so that it can say what it shows. " +
      "Each image costs about a thousand tokens (see Usage). When this is off, the model sees a chart only when you ask it to look at one.",
    (on) => {
      seeCharts = on;
    },
  );
  /** Whether the model may search the web, remembered by the browser */
  let webSearch = loadSetting("webSearch") === "true";
  const webSearchLabel = makeOption(
    "webSearch",
    "Allow web search",
    "With OpenAI and Anthropic: the model may search the web, e.g. for papers or documentation. " +
      "Each search costs about 1 cent (see Usage), and the sources are shown under the answer. The code in the sandbox never gets the internet.",
    (on) => {
      webSearch = on;
      updateLayout();
    },
  );
  const optionsRow = document.createElement("div");
  optionsRow.style.cssText =
    "display: flex; flex-wrap: wrap; gap: 4px 16px; align-items: center;";
  optionsRow.append(toolDetailsLabel, seeChartsLabel, webSearchLabel);

  const status = document.createElement("span");
  status.setAttribute("role", "status");
  status.style.cssText = "flex: 1;";
  const checkButton = makeButton(
    "Check again",
    () => void check(),
    smallButtonStyle,
  );
  const statusRow = document.createElement("div");
  statusRow.style.cssText = "display: flex; gap: 8px; align-items: center;";
  statusRow.append(status, checkButton);

  // the skills of the open project, and those added by hand, remembered by
  // the browser if the user wants: kept by the services, shown here
  services.hooks.skillsChanged = updateLayout;
  services.hooks.addSkillFromGitHub = () => void addGitHubSkillFromSettings();
  /** The skills of the project whose scripts the user allowed, by project and name */
  const allowedScripts = new Set<string>();

  /** Shows the download, in the row of the tool that waits for Python */
  let showPythonDownload: (() => void) | null = null;
  const stopListeningToDownload = services.onPythonDownload(() =>
    showPythonDownload?.(),
  );
  /**
   * Python, kept for the next runs of this chat; code from elsewhere gets a
   * new one
   */
  const pythonKernel = services.makeKernelSlot();

  /** Stops Python: the next Python code starts it again */
  function stopPython(): void {
    pythonKernel.stop();
  }

  const pythonFilesInfo = document.createElement("span");
  pythonFilesInfo.style.cssText = quietTextStyle;
  const forgetPythonButton = makeButton(
    "Forget Python files",
    () => void forgetPython(),
    smallButtonStyle,
  );
  forgetPythonButton.title =
    "Remove the files of Python that this browser keeps; they are downloaded again when Python is needed";
  const pythonRow = document.createElement("div");
  pythonRow.style.cssText =
    "display: flex; flex-wrap: wrap; gap: 8px; align-items: center;";
  pythonRow.append(pythonFilesInfo, forgetPythonButton);

  /** Shows how many files of Python this browser keeps */
  async function updatePythonFiles(): Promise<void> {
    const count = await countPythonFiles();
    pythonFilesInfo.textContent =
      count === 0
        ? "Python: no files downloaded"
        : `Python: ${count} files kept in this browser`;
    forgetPythonButton.disabled = count === 0;
  }

  /** Forgets the files of Python that this browser keeps */
  async function forgetPython(): Promise<void> {
    const count = await forgetPythonFiles();
    pythonFilesInfo.textContent = `Python: ${count} files forgotten; they are downloaded again when Python is needed`;
    forgetPythonButton.disabled = true;
  }
  // the plugins made by the chat, which live in its setup, shown here
  const pluginList = makePluginList(services.plugins, {
    askToRun: (entry, kind) => {
      void askToRunPlugin(entry, kind, { signal });
    },
    showCode: (entry) =>
      showScript(
        `🧩 ${entry.name}`,
        `The code of the plugin ${entry.id}, made by the chat`,
        entry.code,
      )(),
  });

  const settingsBox = document.createElement("div");
  settingsBox.style.cssText =
    "display: flex; flex-direction: column; gap: 8px;";
  settingsBox.append(
    serviceRow,
    keyRow,
    statusRow,
    optionsRow,
    skillList.element,
    pluginList.element,
    pythonRow,
  );

  const history = document.createElement("div");
  history.style.cssText =
    "flex: 1; min-height: 100px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px;" +
    "padding: 8px; border: 1px solid var(--border); border-radius: 6px;";

  // above the text box; the suggestions go on to a new line instead of
  // scrolling sideways, and stay quiet
  const examplesRow = document.createElement("div");
  examplesRow.style.cssText = "display: flex; flex-wrap: wrap; gap: 4px 6px;";

  const input = document.createElement("textarea");
  input.rows = 3;
  input.placeholder =
    "Type a message... (Enter to send, Shift+Enter for a new line, ↑ ↓ for the messages sent before)";
  input.style.cssText = `${controlStyle}flex: 1; resize: none;`;
  const sendButton = makeButton("Send", () => void send());
  sendButton.style.cssText +=
    "background: var(--primary); color: var(--primary-foreground); border-color: var(--primary);";
  const bottomRow = document.createElement("div");
  bottomRow.style.cssText = "display: flex; gap: 8px;";
  bottomRow.append(input, sendButton);

  const footer = document.createElement("div");
  footer.style.cssText = `${quietTextStyle}font-size: 11px; text-align: right;`;
  // "The AID of DIA" opens About; the name of the author links to GitHub
  const showAbout = () => show(persona.name, aboutText, "info", true);
  const aboutLink = document.createElement("button");
  aboutLink.type = "button";
  aboutLink.textContent = credits.title;
  aboutLink.title = "About: where the names Dia and Aid come from";
  aboutLink.style.cssText =
    "padding: 0; border: none; background: none; color: inherit; font: inherit;" +
    "cursor: pointer; text-decoration: none;";
  aboutLink.addEventListener("click", showAbout);
  const creditsLink = document.createElement("a");
  creditsLink.textContent = credits.author;
  creditsLink.href = credits.url;
  creditsLink.target = "_blank";
  creditsLink.rel = "noopener noreferrer";
  creditsLink.title = "View on GitHub";
  creditsLink.style.cssText = "color: inherit; text-decoration: none;";
  // underlined only under the pointer, so that it stays quiet
  for (const link of [aboutLink, creditsLink]) {
    link.addEventListener("mouseenter", () => {
      link.style.textDecoration = "underline";
    });
    link.addEventListener("mouseleave", () => {
      link.style.textDecoration = "none";
    });
  }
  footer.append(aboutLink, " · Made by ", creditsLink);

  // shown under a part of an answer that the user selects
  const quoteButton = makeButton("Quote", quoteSelection, smallButtonStyle);
  quoteButton.style.cssText +=
    "position: absolute; display: none; z-index: 2;" +
    "background: var(--primary); color: var(--primary-foreground); border-color: var(--primary);";
  // a click on the button would otherwise end the selection first
  quoteButton.addEventListener("mousedown", (event) => event.preventDefault());

  /**
   * Finds the useful columns of one table: groups (e.g. clusters), numbers and
   * gene matrices; the others are only counted
   */
  async function readTable(tableId: string, projectId: string): Promise<void> {
    try {
      const tableRef = dataStore.getState().tableDataRefs.get(tableId);
      if (tableRef === undefined) {
        throw new Error("the table is not loaded");
      }
      const data = await tableRef.promise;
      const rows = data.getSize();
      const tooBig = rows > maxRowsToRead;
      const used = findUsedColumns(projectStore.getState(), tableId);
      const useful: ColumnInfo[] = [];
      let other = 0;
      const listed = await listColumns(data, { signal });
      // also in hidden groups such as obsm/: for spatial analyses
      const coordinates = await findCoordinates(data, listed, {
        signal,
      }).catch(() => {
        signal.throwIfAborted();
        return undefined;
      });
      for (const { name, matrix, hidden } of listed) {
        if (hidden || (tooBig && !used.has(name))) {
          other++;
          continue;
        }
        // each column on its own: if one cannot be read, the others still are
        try {
          const described = matrix
            ? await describeMatrix(data, name, { signal })
            : describeValues(await data.loadValues(name, { signal }));
          if (described === undefined) {
            other++;
          } else {
            useful.push({ name, ...described });
          }
        } catch {
          signal.throwIfAborted();
          other++;
        }
      }
      const computed =
        projectStore.getState().tables.find(({ id }) => id === tableId)
          ?.dataSource.type === computedTableType;
      saveTableInfo(tableId, projectId, {
        status: "ready",
        ...(computed
          ? {
              note: "computed by the chat; temporary: lost when the page is reloaded, unless kept with save_computed_table",
            }
          : {}),
        rows,
        tooBig,
        useful,
        other,
        ...(coordinates === undefined ? {} : { coordinates }),
      });
    } catch (error) {
      if (signal.aborted) {
        return;
      }
      saveTableInfo(tableId, projectId, {
        status: "error",
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  /** Keeps what was read about a table, if the same project is still open */
  function saveTableInfo(
    tableId: string,
    projectId: string,
    info: TableInfo,
  ): void {
    if (projectStore.getState().instanceId !== projectId) {
      return;
    }
    tables.info.set(tableId, info);
    updateExamples();
  }

  /**
   * Starts reading the tables of labels and points that TissUUmaps has loaded
   * and that were not read yet; runs again whenever the data store changes,
   * since tables can finish loading later
   */
  function readTables(): void {
    for (const tableId of linkedTables(projectStore.getState(), tables)) {
      if (
        dataStore.getState().tableDataRefs.has(tableId) &&
        !tables.info.has(tableId)
      ) {
        tables.info.set(tableId, { status: "reading" });
        const projectId = projectStore.getState().instanceId;
        readingQueue = readingQueue.then(() => readTable(tableId, projectId));
      }
    }
  }

  /**
   * Reads the tables of changed labels and points again: in a very big table
   * only the columns used now are read, and the colors may use others now
   */
  function rereadTables(someChanges: Change[]): void {
    const project = projectStore.getState();
    for (const { kind, id } of someChanges) {
      if (kind !== "labels" && kind !== "points") {
        continue;
      }
      const object = findById(project, kind, id) as ItemsObject | undefined;
      for (const tableId of object === undefined
        ? []
        : objectTables(object, tables)) {
        if (tables.info.get(tableId)?.status !== "reading") {
          tables.info.delete(tableId);
        }
      }
    }
    readTables();
  }

  /**
   * Shows example requests above the text box, also while the user types; a
   * click sends one, and leaves the text being typed as it is
   */
  function updateExamples(): void {
    const examples = exampleRequests(projectStore.getState(), tables);
    examplesRow.replaceChildren(
      ...examples.map((text) => {
        const button = makeButton(
          text,
          () => void send(text),
          `${smallButtonStyle}max-width: 100%; text-align: left; border-radius: 12px;` +
            "color: var(--muted-foreground); border-color: var(--border);",
        );
        button.title = "Send this request";
        return button;
      }),
    );
    examplesRow.style.display = examples.length > 0 ? "flex" : "none";
  }

  /**
   * Adds the frame of a message to the history: a name, with the avatar for
   * the assistant, and an empty body
   */
  function addFrame(who: string, kind: MessageKind): HTMLElement {
    const message = document.createElement("div");
    message.dataset.who = who;
    message.style.cssText =
      "padding: 6px 8px; border: 1px solid; border-radius: 6px; white-space: pre-wrap;" +
      messageStyles[kind];
    const header = document.createElement("div");
    header.style.cssText =
      "display: flex; align-items: center; gap: 6px; font-weight: bold;";
    if (kind === "error") {
      // the red alone is too light on the light theme's error background
      header.style.color =
        "color-mix(in oklch, var(--destructive) 70%, var(--foreground))";
    }
    if (kind === "assistant") {
      header.append(makeAvatar(persona));
    }
    header.append(who);
    const body = document.createElement("div");
    message.append(header, body);
    addToHistory(message);
    return body;
  }

  /**
   * Adds a message at the end of the history, but above the indicator that
   * the model is thinking, which stays the last one while the model works
   */
  function addToHistory(message: HTMLElement): void {
    history.insertBefore(message, thinking?.element ?? null);
    history.scrollTop = history.scrollHeight;
    lastFrame = message;
  }

  /**
   * The rows of the tools and the scripts, which open and close with "Show
   * tool details", except those kept open
   */
  const toolRows: { details: HTMLDetailsElement; keepOpen: boolean }[] = [];

  /**
   * Shows a tool call as one folded row: an icon and the first line of the
   * result, e.g. "🌫 Done: opacity of cell_labels 1 → 0.5"; opened, the call
   * and the whole result. Failed calls are always open.
   *
   * @param name - The tool
   * @param shownCall - The call, as it reads
   * @param result - The result for the model
   * @param before - The message to show it above, e.g. the answer that the
   * web searches were for; without it, at the end
   */
  function showTool(
    name: string,
    shownCall: string,
    result: string,
    before?: Element | null,
  ): void {
    const failed = result.startsWith("Error");
    const message = document.createElement("div");
    message.dataset.who = "Tool";
    message.style.cssText =
      "padding: 4px 8px; border: 1px solid; border-radius: 6px;" +
      messageStyles.tool;
    const details = document.createElement("details");
    details.open = failed || showToolDetails;
    const line = document.createElement("summary");
    line.style.cssText =
      "cursor: pointer; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;";
    line.textContent = `${failed ? "⚠️" : (toolIcons[name] ?? "🔧")} ${firstLine(result)}`;
    line.title = line.textContent;
    const body = document.createElement("div");
    body.dataset.details = "true";
    body.style.cssText = "white-space: pre-wrap; margin-top: 4px;";
    body.textContent = `${shownCall}\n→ ${result}`;
    details.append(line, body);
    message.append(details);
    if (before?.parentElement === history) {
      history.insertBefore(message, before);
    } else {
      addToHistory(message);
    }
    toolRows.push({ details, keepOpen: failed });
  }

  /** Opens or folds the rows of the tools, as "Show tool details" says */
  function updateToolRows(): void {
    for (const { details, keepOpen } of toolRows) {
      details.open = keepOpen || showToolDetails;
    }
  }

  /**
   * Adds a message to the history and returns a function that sets its text,
   * so that an answer can be shown while it arrives
   */
  function addMessage(
    who: string,
    kind: MessageKind = "info",
    markdown = false,
  ): (text: string) => void {
    const body = addFrame(who, kind);
    let shownText = "";
    if (kind === "assistant") {
      // a part of the answers can be quoted
      body.dataset.quotable = "true";
      // the answer as the model wrote it, e.g. with its Markdown
      const copyButton = makeCopyButton(() => shownText, "the answer");
      copyButton.style.marginLeft = "auto";
      body.previousElementSibling?.append(copyButton);
    }
    return (text) => {
      shownText = text;
      body.replaceChildren(markdown ? renderMarkdown(text) : text);
      history.scrollTop = history.scrollHeight;
    };
  }

  /**
   * Adds a message with its final text to the history; without a kind, it
   * follows from the name ("You", "Tool", "Note", "Error")
   */
  function show(
    who: string,
    text: string,
    kind?: MessageKind,
    markdown?: boolean,
  ): void {
    const kinds: Record<string, MessageKind> = {
      You: "user",
      Tool: "tool",
      Note: "note",
      Error: "error",
    };
    addMessage(who, kind ?? kinds[who] ?? "info", markdown)(text);
  }

  /** Shows that the model is thinking, at the end of the history */
  function startThinking(): void {
    stopThinking();
    const indicator = makeThinkingIndicator();
    const body = addFrame(persona.name, "assistant");
    body.append(indicator.element);
    thinking = {
      element: body.parentElement ?? body,
      stop: indicator.stop,
    };
  }

  /** Removes the indicator that the model is thinking, if it is shown */
  function stopThinking(): void {
    thinking?.stop();
    thinking?.element.remove();
    thinking = null;
  }

  /**
   * The text the user selected in one answer of the model, or "" if nothing
   * or something else is selected
   */
  function selectedAnswerText(): { text: string; range?: Range } {
    const selection = window.getSelection();
    if (
      selection === null ||
      selection.isCollapsed ||
      selection.rangeCount === 0
    ) {
      return { text: "" };
    }
    const range = selection.getRangeAt(0);
    const node = range.commonAncestorContainer;
    const element = node instanceof Element ? node : node.parentElement;
    const answer = element?.closest("[data-quotable]");
    if (answer === null || answer === undefined || !history.contains(answer)) {
      return { text: "" };
    }
    return { text: selection.toString().trim(), range };
  }

  /** Shows "Quote" under the selected part of an answer, or hides it */
  function updateQuoteButton(): void {
    const { text, range } = selectedAnswerText();
    if (text === "" || range === undefined) {
      quoteButton.style.display = "none";
      return;
    }
    // not every browser engine can measure a selection (e.g. in tests)
    if (typeof range.getBoundingClientRect === "function") {
      const selected = range.getBoundingClientRect();
      const panel = container.getBoundingClientRect();
      quoteButton.style.left = `${Math.max(0, selected.left - panel.left)}px`;
      quoteButton.style.top = `${Math.max(0, selected.bottom - panel.top + 4)}px`;
    }
    quoteButton.style.display = "";
  }

  /** Puts the selected part of an answer as a quote into the text box */
  function quoteSelection(): void {
    const { text } = selectedAnswerText();
    if (text === "") {
      return;
    }
    const quote = text
      .split("\n")
      .map((line) => `> ${line}`)
      .join("\n");
    input.value = `${quote}\n\n${input.value}`;
    window.getSelection()?.removeAllRanges();
    quoteButton.style.display = "none";
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }

  /**
   * Up and down arrows in the text box: shows the messages sent before, like a
   * terminal. Only on the first or the last line of the text, so that in a
   * longer text the arrows still move between its lines.
   */
  function browseSent(event: KeyboardEvent): void {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
      return;
    }
    const up = event.key === "ArrowUp";
    const { value, selectionStart, selectionEnd } = input;
    const onEdge = up
      ? !value.slice(0, selectionStart).includes("\n")
      : !value.slice(selectionEnd).includes("\n");
    const next = sentIndex + (up ? -1 : 1);
    if (!onEdge || next < 0 || next > sent.length) {
      return;
    }
    event.preventDefault();
    if (sentIndex === sent.length) {
      draft = value;
    }
    sentIndex = next;
    input.value = next === sent.length ? draft : (sent[next] ?? "");
    input.setSelectionRange(input.value.length, input.value.length);
  }

  /** Starts a new conversation, after asking; the undo list stays, since it is about the project */
  function newChat(): void {
    if (sendButton.disabled) {
      return;
    }
    const question =
      "Start a new chat? The model forgets this conversation. Undo still works for the changes made so far.";
    if (messages.length > 0 && !window.confirm(question)) {
      return;
    }
    messages.length = 0;
    history.replaceChildren();
    for (const url of chartUrls.splice(0)) {
      URL.revokeObjectURL(url);
    }
    charts.length = 0;
    toolRows.length = 0;
    updateExamples();
    // a new chat gets a new Python
    stopPython();
  }

  /** Enables "Undo" and "Undo all" when there is something to undo and the model is not working */
  function updateUndoButtons(): void {
    const enabled = undoStack.length > 0 && !sendButton.disabled;
    for (const button of [undoButton, undoAllButton]) {
      button.disabled = !enabled;
      button.style.opacity = enabled ? "1" : "0.5";
    }
  }

  /**
   * Puts back the values from before one request, from its last change to
   * its first, and returns one line per change for the chat
   */
  function undoRequest(changes: Change[]): string[] {
    const lines: string[] = [];
    for (const change of [...changes].reverse()) {
      const now = currentValue(change, projectStore, appStore);
      if (now === undefined) {
        const kept =
          change.type === "computed" &&
          projectStore.getState().tables.some(({ id }) => id === change.id);
        lines.push(
          kept
            ? `${change.name} was kept as a CSV file, so it stays`
            : `${change.name} is not in the project any more, so it was skipped`,
        );
        continue;
      }
      takeBack(change, projectStore, appStore);
      // the changes of the maps are only named, e.g. "Chat colors – cell_labels"
      let line =
        isMapKind(change.kind) && change.type === undefined
          ? change.name
          : describeValueChange(change, now.value, change.before);
      if (!same(now.value, change.after)) {
        line += handNote;
      }
      lines.push(line);
    }
    forgetGoneComputedTables();
    rereadTables(changes);
    return [...new Set(lines)];
  }

  /**
   * Tells the model what was undone, so that it does not trust the old tool
   * results; as a "user" message, since not every model accepts a system
   * message in the middle of a conversation
   */
  function tellModel(text: string): void {
    messages.push({
      role: "user",
      text: `(Note from the chat panel, not typed by the user: ${text} The project summary shows the current state.)`,
    });
  }

  /** Takes back the changes of the last request */
  function undo(): void {
    const changes = undoStack.pop();
    if (changes === undefined) {
      return;
    }
    const lines = undoRequest(changes);
    show("Undo", `Undone: ${lines.join(", ")}`);
    tellModel(`the user clicked "Undo". Undone: ${lines.join(", ")}.`);
    updateUndoButtons();
  }

  /** Takes back all the changes made by the chat, after asking */
  function undoAll(): void {
    const question =
      "Undo all the changes made by the chat? Settings you changed by hand that the chat never touched stay as they are.";
    if (!window.confirm(question)) {
      return;
    }
    const touched = new Map<string, { change: Change; from: unknown }>();
    for (const change of undoStack.flat()) {
      const key = `${change.type ?? ""} ${change.kind} ${change.id} ${change.property}`;
      if (!touched.has(key)) {
        touched.set(key, {
          change,
          from: currentValue(change, projectStore, appStore)?.value,
        });
      }
    }
    const notes: string[] = [];
    for (
      let changes = undoStack.pop();
      changes !== undefined;
      changes = undoStack.pop()
    ) {
      for (const line of undoRequest(changes)) {
        if (line.endsWith(handNote) || line.endsWith("skipped")) {
          notes.push(line);
        }
      }
    }
    const restored: string[] = [];
    for (const { change, from } of touched.values()) {
      const to = currentValue(change, projectStore, appStore)?.value;
      if (change.type === "added") {
        restored.push(describeValueChange(change, from, to));
      } else if (
        from !== undefined &&
        !same(from, to) &&
        !isMapKind(change.kind)
      ) {
        restored.push(describeValueChange(change, from, to));
      }
    }
    let text = `Restored: ${restored.join(", ") || "nothing, the values were already as before"}.`;
    if (notes.length > 0) {
      text += `\nNotes: ${notes.join("; ")}.`;
    }
    show("Undo all", text);
    tellModel(
      `the user clicked "Undo all", which took back every change made by the chat. ${text}`,
    );
    updateUndoButtons();
  }

  /**
   * Keeps the examples up to date, and when another project is opened (it
   * gets a new instanceId), forgets the old project's tables and undo list
   * and asks whether to clear the conversation
   */
  function onProjectChange(
    state: ProjectStore,
    previousState: ProjectStore,
  ): void {
    const objectsChanged =
      state.images !== previousState.images ||
      state.labels !== previousState.labels ||
      state.points !== previousState.points;
    if (objectsChanged) {
      updateExamples();
    }
    if (state.instanceId === previousState.instanceId) {
      if (state.tables !== previousState.tables) {
        linkComputedTables(previousState);
      }
      return;
    }
    // Python may hold data of the other project
    stopPython();
    tables.info.clear();
    tables.remembered.clear();
    readTables();
    updateExamples();
    if (undoStack.length > 0) {
      undoStack.length = 0;
      show(
        "Undo",
        "Another project was opened, so there is nothing to undo any more.",
      );
      updateUndoButtons();
    }
    if (messages.length === 0) {
      return;
    }
    show(
      "Chat",
      "Another project was opened. The earlier messages talk about objects of the old project.",
    );
    // asked a moment later, so that TissUUmaps can finish opening the project first
    setTimeout(() => {
      const question =
        "Another project was opened. Clear the conversation with the model, so that it starts fresh with the new project?";
      if (window.confirm(question)) {
        messages.length = 0;
        show(
          "Chat",
          "Conversation cleared: the model only knows about the new project now.",
        );
        updateExamples();
      } else {
        show(
          "Chat",
          "Conversation kept: the model still remembers the earlier messages.",
        );
      }
    }, 0);
  }

  /** Fills the model menu with the models of the chosen service */
  function fillModelMenu(): void {
    const provider = currentProvider();
    const chosen = loadSetting(`model.${provider.id}`) ?? modelSelect.value;
    const models = usesOllama(provider)
      ? ollamaModels(provider)
      : provider.models;
    modelSelect.replaceChildren();
    for (const { name, note } of models) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = `${name} (${note})`;
      modelSelect.append(option);
    }
    if (models.some(({ name }) => name === chosen)) {
      modelSelect.value = chosen;
    }
  }

  /**
   * The models of an Ollama service: the installed ones of that kind (local or
   * online) first, then the suggested ones
   */
  function ollamaModels(provider: Provider): { name: string; note: string }[] {
    const online = provider.id === "ollama-cloud";
    const names = [
      ...installed.filter((name) => name.endsWith("cloud") === online),
      ...provider.models.map(({ name }) => name),
    ];
    return [...new Set(names)].map((name) => {
      let note = connected ? "not downloaded" : "local";
      if (online) {
        note = "online";
      } else if (installed.includes(name)) {
        note = "local";
      }
      return { name, note };
    });
  }

  /** The service chosen in the menu */
  function currentProvider(): Provider {
    return providers.find(
      (provider) => provider.id === providerSelect.value,
    ) as Provider;
  }

  /**
   * Writes the status line, and shows the key field and "Download" only when
   * they are needed
   */
  function updateStatus(): void {
    const provider = currentProvider();
    const name = modelSelect.value;
    downloadButton.style.display = "none";
    keyRow.style.display = provider.needsKey ? "flex" : "none";
    if (provider.needsKey) {
      status.textContent =
        keyStatus.get(provider.id) ??
        (keys.has(provider.id)
          ? `${provider.name}: the key is not checked yet. Click "Check again".`
          : `${provider.name}: paste your API key below. No key yet? Click "Help".`);
    } else if (!connected) {
      status.textContent = unreachable;
    } else if (provider.id === "ollama-cloud") {
      status.textContent = `Ollama: connected. ${name} runs online: you need to be signed in (see Help).`;
    } else if (installed.includes(name)) {
      status.textContent = `Ollama: connected. ${name} is ready.`;
    } else {
      status.textContent = `Ollama: connected. ${name} is not downloaded yet.`;
      downloadButton.style.display = "";
    }
    updateLayout();
  }

  /**
   * Folds the settings into one short line while the key of the chosen
   * service works, unless the user opened them with ⚙
   */
  function updateLayout(): void {
    const provider = currentProvider();
    const accepted = acceptedKeys.has(provider.id);
    const folded = accepted && !settingsOpen;
    settingsBox.style.display = folded ? "none" : "flex";
    summary.style.display = folded ? "" : "none";
    const skills = skillList.activeSkills().length;
    const searching = webSearch && provider.searchPrice !== undefined;
    summary.textContent = `${provider.name} · ${modelSelect.value} · key ✓${skills > 0 ? ` · ${skills} skill${skills === 1 ? "" : "s"}` : ""}${searching ? " · web search" : ""}`;
    settingsButton.style.display = accepted ? "" : "none";
    settingsButton.setAttribute("aria-expanded", String(!folded));
  }

  /** ⚙: shows or hides the settings */
  function toggleSettings(): void {
    settingsOpen = !settingsOpen;
    updateLayout();
  }

  /**
   * Something went wrong with OpenAI or Anthropic: the status line says what,
   * and the settings open, so that the key can be checked or changed
   */
  function showServiceProblem(provider: Provider, text: string): void {
    if (usesOllama(provider)) {
      return;
    }
    acceptedKeys.delete(provider.id);
    keyStatus.set(provider.id, text);
    settingsOpen = true;
    updateStatus();
  }

  /** Shows the key of the chosen service, and whether it is remembered */
  function showKey(): void {
    const provider = currentProvider();
    keyInput.value = keys.get(provider.id) ?? "";
    keyInput.placeholder = `Paste your ${provider.name} API key`;
    rememberBox.checked = loadSetting(`key.${provider.id}`) !== null;
  }

  /** Keeps the key as it is typed, and remembers it if the user wants */
  function onKeyInput(): void {
    const provider = currentProvider();
    const key = keyInput.value.trim();
    if (key === "") {
      keys.delete(provider.id);
    } else {
      keys.set(provider.id, key);
    }
    if (rememberBox.checked) {
      saveSetting(`key.${provider.id}`, key === "" ? null : key);
    }
    keyStatus.delete(provider.id);
    acceptedKeys.delete(provider.id);
    updateStatus();
  }

  /** Remembers the key in the browser, or forgets it there */
  function onRememberChange(): void {
    const provider = currentProvider();
    saveSetting(
      `key.${provider.id}`,
      rememberBox.checked ? (keys.get(provider.id) ?? null) : null,
    );
  }

  /** Forgets the key of the chosen service, also in the browser */
  function forgetKey(): void {
    const provider = currentProvider();
    keys.delete(provider.id);
    keyStatus.delete(provider.id);
    acceptedKeys.delete(provider.id);
    saveSetting(`key.${provider.id}`, null);
    keyInput.value = "";
    rememberBox.checked = false;
    updateStatus();
  }

  /**
   * Checks the key of the chosen service, with a request that costs nothing.
   * A working key folds the settings; the first time a pasted key works, the
   * chat also says what using it means.
   *
   * @param pasted - Whether the user has just pasted the key
   */
  async function checkKey(pasted = false): Promise<void> {
    const provider = currentProvider();
    const key = keys.get(provider.id);
    keyStatus.delete(provider.id);
    acceptedKeys.delete(provider.id);
    if (key === undefined || provider.checkKey === undefined) {
      updateStatus();
      return;
    }
    keyStatus.set(provider.id, `${provider.name}: checking the key...`);
    updateStatus();
    let error: string | undefined;
    try {
      error = await provider.checkKey(key, { signal });
    } catch {
      if (signal.aborted) {
        return;
      }
      error = notReachable(provider);
    }
    // the key may have been changed in the meantime
    if (keys.get(provider.id) !== key) {
      return;
    }
    if (error !== undefined) {
      showServiceProblem(provider, error);
      return;
    }
    const text = `${provider.name}: key accepted. Each request is billed to your ${provider.name} account, and your messages and the project summary are sent to ${provider.name}.`;
    keyStatus.set(provider.id, text);
    acceptedKeys.add(provider.id);
    settingsOpen = false;
    updateStatus();
    if (pasted && !announcedKeys.has(provider.id)) {
      announcedKeys.add(provider.id);
      show(provider.name, `${text} The key settings are under ⚙.`, "note");
    }
  }

  /** "Check again": asks Ollama for its models, or checks the key */
  async function check(): Promise<void> {
    await (usesOllama(currentProvider()) ? checkOllama() : checkKey());
  }

  /** Nothing answers at the Ollama address: finds out why, for the status line */
  async function lostOllama(): Promise<string> {
    connected = false;
    unreachable = await explainNotReachable({ signal });
    updateStatus();
    return unreachable;
  }

  /** Asks Ollama which models are installed */
  async function checkOllama(): Promise<void> {
    status.textContent = "Ollama: checking...";
    try {
      const response = await fetch(`${ollamaUrl}/api/tags`, { signal });
      const data = (await response.json()) as { models: { name: string }[] };
      installed = data.models.map((model) => model.name);
      connected = true;
    } catch {
      if (signal.aborted) {
        return;
      }
      connected = false;
      unreachable = await explainNotReachable({ signal });
    }
    fillModelMenu();
    updateStatus();
  }

  /** Downloads the chosen model, with the progress in the status line */
  async function download(): Promise<void> {
    const name = modelSelect.value;
    downloadButton.disabled = true;
    status.textContent = `Downloading ${name}...`;
    try {
      const response = await fetch(`${ollamaUrl}/api/pull`, {
        method: "POST",
        body: JSON.stringify({ model: name }),
        signal,
      });
      await readJsonLines<OllamaPullUpdate>(
        response,
        (update) => {
          if (update.error !== undefined) {
            throw new Error(update.error);
          }
          status.textContent =
            update.total === undefined
              ? `Downloading ${name}: ${update.status ?? ""}`
              : `Downloading ${name}: ${Math.round((100 * (update.completed ?? 0)) / update.total)}%`;
        },
        { signal },
      );
      downloadButton.disabled = false;
      await checkOllama();
    } catch (error) {
      if (signal.aborted) {
        return;
      }
      downloadButton.disabled = false;
      // fetch fails with a TypeError when nothing answers
      if (error instanceof TypeError) {
        await lostOllama();
      } else {
        status.textContent = `Download failed: ${error instanceof Error ? error.message : String(error)}`;
      }
    }
  }

  /**
   * The context of a new turn: the personality of now (a new theme changes
   * the tone at once) and the summary of the project as it is now
   *
   * It stays the same for all the requests of the turn, while the model calls
   * tools: OpenAI and Anthropic only take the model's thinking back while
   * everything before it is unchanged. The tool results tell the model what
   * changed during the turn.
   */
  async function turnContext(provider: Provider): Promise<string> {
    // the summary should show the current columns: wait for the table reads,
    // but not for more than 5 seconds
    await Promise.race([
      readingQueue,
      new Promise((resolve) => setTimeout(resolve, 5000)),
    ]);
    return contextFor(provider);
  }

  /**
   * The instructions for a service: for the strong models, with the types of
   * the app state, which never change
   */
  function instructionsFor(provider: Provider): string {
    return provider.wholeApp
      ? `${instructions.replace(toolLimits, actionLimits)}\n\n${wholeAppInstructions}\n\n${skillInstructions}\n\n${codeInstructions}\n\n${pluginInstructions}\n\n${webInstructions}`
      : instructions;
  }

  /**
   * The tools for a service: for the strong models, also call_action, the
   * skills and run_code, and look_at_chart if they see images
   */
  function toolsFor(provider: Provider) {
    return [
      ...toolDefinitions,
      ...(provider.wholeApp
        ? [
            callActionDefinition,
            ...skillToolDefinitions,
            runCodeDefinition,
            saveComputedTableDefinition,
            proposePluginDefinition,
            readPluginDefinition,
            ...(provider.seesImages ? [lookAtChartDefinition] : []),
          ]
        : []),
    ].map((definition) => definition.function);
  }

  /**
   * Whether the model may search the web now: only with a service that can,
   * when the user allows it
   */
  function searchAllowed(provider: Provider): boolean {
    return webSearch && provider.searchPrice !== undefined;
  }

  /**
   * Whether the model gets the image of each chart with the result of the
   * code that drew it
   */
  function chartsSent(provider: Provider): boolean {
    return seeCharts && provider.seesImages;
  }

  /**
   * The context as it is now: for the strong models, with the skills that
   * are on and the app state as JSON after the summary
   */
  function contextFor(provider: Provider): string {
    const summary = buildContext(
      projectStore.getState(),
      dataStore,
      tables,
      persona,
    );
    if (!provider.wholeApp) {
      return summary;
    }
    const skills = describeSkills(skillList.activeSkills());
    const settings = [
      "# Settings of the chat",
      webSearchState(searchAllowed(provider)),
      !provider.seesImages
        ? "- Charts: you cannot see images, only the descriptions of the charts."
        : chartsSent(provider)
          ? "- Charts: you see them: the image of each chart that your code draws comes with its result."
          : "- Charts: you get their descriptions only; call look_at_chart to see one when the user asks.",
    ].join("\n");
    const state = stateJson(projectStore.getState(), appStore.getState());
    const plugins = describePlugins(services.plugins.list());
    return `${summary}\n\n${skills}\n\n${plugins}\n\n${settings}\n\n# The current app state, as JSON\n${state}`;
  }

  /**
   * Sends the instructions, the context of the turn, the conversation and the
   * tools to the model
   *
   * @param provider - The service
   * @param model - The model
   * @param key - The API key of the service, or ""
   * @param context - The context of the turn, see {@link turnContext}
   * @param search - Whether the model may search the web in this turn
   * @param answerNow - Whether the model must answer now, without tools
   * @param onText - Called with the text so far, each time a piece arrives
   * @returns The answer, or an error message
   */
  async function askModel(
    provider: Provider,
    model: string,
    key: string,
    context: string,
    search: boolean,
    answerNow: boolean,
    onText: (text: string) => void,
  ): Promise<ChatAnswer> {
    return provider.chat(
      {
        model,
        key,
        instructions: instructionsFor(provider),
        context,
        messages,
        tools: toolsFor(provider),
        webSearch: search,
        answerNow,
      },
      onText,
      { signal },
    );
  }

  /**
   * Shows under the last message of a turn, folded under "Usage", the tokens
   * of its requests, and for the services that bill them, about how much they
   * cost, also in total since the panel was opened
   */
  function showUsage(
    provider: Provider,
    model: string,
    usage: Usage,
    requests: number,
    extra: { searches: number; images: number },
  ): void {
    const message = history.lastElementChild;
    if (requests === 0 || message === null) {
      return;
    }
    const count = (tokens: number) => tokens.toLocaleString("en-US");
    const parts = [`${count(usage.input)} input`];
    if (provider.needsKey) {
      parts.push(
        `${count(usage.cacheRead)} from cache`,
        `${count(usage.cacheWrite)} written to cache`,
      );
    }
    parts.push(`${count(usage.output)} output`);
    if (extra.images > 0) {
      // the services count the images as input tokens
      parts.push(
        `${extra.images} chart image${extra.images === 1 ? "" : "s"} (in the input)`,
      );
    }
    const searchCost = extra.searches * (provider.searchPrice ?? 0);
    if (extra.searches > 0) {
      parts.push(
        `${extra.searches} web search${extra.searches === 1 ? "" : "es"} (${dollars(searchCost)})`,
      );
    }
    const prices = provider.models.find(({ name }) => name === model)?.prices;
    if (prices === undefined) {
      parts.push("free");
    } else {
      const cost =
        (usage.input * prices.input +
          usage.cacheRead * prices.cacheRead +
          usage.cacheWrite * prices.cacheWrite +
          usage.output * prices.output) /
          1_000_000 +
        searchCost;
      sessionCost += cost;
      parts.push(
        `≈ ${dollars(cost)}`,
        `this session ≈ ${dollars(sessionCost)}`,
      );
    }
    // folded, so that the chat is not crowded
    const usageBox = document.createElement("details");
    usageBox.style.cssText = "margin-top: 4px;";
    const toggle = document.createElement("summary");
    toggle.style.cssText = `${quietButtonStyle}width: fit-content;`;
    toggle.textContent = "Usage";
    toggle.title =
      "The tokens of this request, the web searches, and about how much they cost";
    const line = document.createElement("div");
    line.dataset.usage = "true";
    line.style.cssText = `${quietTextStyle}font-size: 11px; margin-top: 2px;`;
    line.textContent = `Tokens${requests > 1 ? ` (${requests} requests)` : ""}: ${parts.join(" · ")}`;
    usageBox.append(toggle, line);
    message.append(usageBox);
  }

  /**
   * Answers a question of a script in the sandbox, always with copies, so
   * that the script cannot change the data of the project
   *
   * @param question - What the script asks for, through tm
   * @param skill - The skill of the script, for tm.readFile; none for the
   * code of the model
   * @param options - Optional abort signal, and the changes of the request,
   * for the columns that the script saves
   * @returns The answer
   * @throws If the question cannot be answered, e.g. an unknown table
   */
  async function answerScript(
    question: ScriptQuestion,
    skill: Skill | undefined,
    options?: { signal?: AbortSignal; changes?: Change[] },
  ): Promise<unknown> {
    const { signal: stop, changes } = options ?? {};
    stop?.throwIfAborted();
    if (question.kind === "save_column") {
      return saveComputedColumn(question, changes, { signal: stop });
    }
    if (question.kind === "download") {
      return offerDownload(question.name, question.text);
    }
    if (question.kind === "file") {
      if (skill === undefined) {
        throw new Error(
          "only the scripts of skills have files to read, and this code is not part of a skill",
        );
      }
      const file = skill.files.get(question.path);
      if (file === undefined) {
        throw new Error(`the skill has no file "${question.path}"`);
      }
      return file.text();
    }
    return services.readData(question, { signal: stop });
  }

  /**
   * Saves a column that code computed, one value per row of a table, into the
   * table "(chat)" of that table (see services.ts), as a change that Undo
   * takes back, and shows it in the chat
   *
   * @param question - The table, the name of the column, the values, and the
   * item IDs of the values if they are not one per row
   * @param changes - The changes of the request
   * @param options - Optional abort signal
   * @returns What was saved, for the code
   * @throws If the column cannot be saved, with the reason for the code
   */
  async function saveComputedColumn(
    question: Extract<ScriptQuestion, { kind: "save_column" }>,
    changes: Change[] | undefined,
    options?: { signal?: AbortSignal },
  ): Promise<unknown> {
    if (changes === undefined) {
      throw new Error("columns can only be saved while the chat runs code");
    }
    const saved = await services.saveColumn(question, options);
    changes.push(saved.change);
    showComputedColumn(
      saved.tableId,
      saved.tableName,
      saved.column,
      saved.values,
    );
    return saved.result;
  }

  /**
   * Lets the labels and points that use a table use its table "(chat)" too,
   * and has the panel read it again
   *
   * @param originId - The original table
   * @param tableId - The table computed by the chat
   */
  function linkComputedTable(originId: string, tableId: string): void {
    const project = projectStore.getState();
    for (const object of [...project.labels, ...project.points]) {
      if (objectTables(object, tables).includes(originId)) {
        const remembered = tables.remembered.get(object.id) ?? new Set();
        remembered.add(tableId);
        tables.remembered.set(object.id, remembered);
      }
    }
    tables.info.delete(tableId);
    readTables();
  }

  /**
   * Links the tables computed by the chat that are new or have a new version
   * since the project was as before, e.g. a column that the code of the chat
   * or of a plugin made by the chat saved; without a project before, all of
   * them, e.g. when the panel opens
   *
   * @param before - The project before
   */
  function linkComputedTables(before?: ProjectStore): void {
    const sources = new Map(
      before?.tables.map(({ id, dataSource }) => [id, dataSource.source]),
    );
    for (const { id, dataSource } of projectStore.getState().tables) {
      const version = versionOf(dataSource.source);
      if (
        dataSource.type === computedTableType &&
        version !== undefined &&
        sources.get(id) !== dataSource.source
      ) {
        linkComputedTable(version.originId, id);
      }
    }
  }

  /** Forgets the tables computed by the chat that are not in the project any more */
  function forgetGoneComputedTables(): void {
    const ids = new Set(projectStore.getState().tables.map(({ id }) => id));
    for (const remembered of tables.remembered.values()) {
      for (const tableId of remembered) {
        if (tableId.startsWith(`${computedTableType}:`) && !ids.has(tableId)) {
          remembered.delete(tableId);
        }
      }
    }
  }

  /**
   * Shows in the chat that code saved a column, with buttons to keep its table
   * as a CSV file or to download it
   *
   * @param tableId - The table computed by the chat
   * @param tableName - Its name
   * @param column - The column
   * @param values - Its values
   */
  function showComputedColumn(
    tableId: string,
    tableName: string,
    column: string,
    values: ComputedValues,
  ): void {
    const body = addFrame("Table", "note");
    const what = describeValues(values)?.text ?? "the same value in every row";
    const line = document.createElement("div");
    line.textContent =
      `📊 New column ${column} in ${tableName} (${what}). It is temporary: ` +
      "it is lost when the page is reloaded, unless the table is kept as a CSV file.";
    const buttons = document.createElement("div");
    buttons.style.cssText = "display: flex; gap: 6px; margin-top: 4px;";
    const keep = makeButton(
      "Keep as CSV",
      () => {
        const target = keepTarget();
        const fileName = askCsvName(
          tableId,
          target === undefined
            ? `This project was not opened from a connected folder, so "${tableName}" is downloaded as a CSV file instead.`
            : `Save "${tableName}" as a CSV file in the folder "computed" next to the project file, in "${target.name}". Then the project reads the table from that file.`,
        );
        if (fileName !== undefined) {
          void keepComputedTable(tableId, { fileName, chosen: true }).then(
            reportKept,
          );
        }
      },
      smallButtonStyle,
    );
    keep.title =
      "Write the table as a CSV file next to the project file, in the connected folder";
    const download = makeButton(
      "Download CSV",
      () => {
        const fileName = askCsvName(
          tableId,
          `Download "${tableName}" as a CSV file.`,
        );
        if (fileName !== undefined) {
          reportKept(downloadComputedTable(tableId, fileName));
        }
      },
      smallButtonStyle,
    );
    download.title = "Download the table as a CSV file";
    buttons.append(keep, download);
    body.append(line, buttons);
  }

  /**
   * Shows what keeping or downloading a computed table did: only what went
   * wrong is an error, not e.g. a download instead, or the user's own choice
   */
  function reportKept(result: string): void {
    show(
      result.startsWith("Error") ? "Error" : "Note",
      result.replace(/^(Error|Done|Note): /, ""),
    );
  }

  /**
   * Asks the user for the name of the CSV file of a computed table, proposing
   * one with the table, its columns and the day, until the name is valid
   *
   * @param tableId - The table computed by the chat
   * @param action - What happens with the file
   * @returns The name, with ".csv", or undefined if the user cancelled
   */
  function askCsvName(tableId: string, action: string): string | undefined {
    const found = findComputedTable(tableId);
    if ("error" in found) {
      reportKept(found.error);
      return undefined;
    }
    let problem = "";
    let proposal = suggestCsvName(found.table.name, [
      ...found.version.columns.keys(),
    ]);
    for (;;) {
      const answer: unknown = window.prompt(
        `${problem}${action}\n\nFile name:`,
        proposal,
      );
      if (typeof answer !== "string") {
        return undefined;
      }
      const checked = csvFileName(answer);
      if ("name" in checked) {
        return checked.name;
      }
      problem = `${checked.error}.\n\n`;
      proposal = answer;
    }
  }

  /**
   * The connected folder that holds the project file, if the computed tables
   * can be written next to it
   *
   * @returns The folder, or undefined for a project opened from elsewhere
   */
  function keepTarget(): FileSystemDirectoryHandle | undefined {
    const workspace = appStore.getState().workspace ?? null;
    const projectSource = projectStore.getState().source;
    if (workspace === null || projectSource === null) {
      return undefined;
    }
    const where = SourceUtils.normalizeSource(
      "computed",
      workspace,
      projectSource,
    );
    return SourceUtils.isWorkspacePath(where) ? workspace : undefined;
  }

  /**
   * Finds a table computed by the chat, by its name or ID
   *
   * @param wanted - The name or ID
   * @returns The table and its values, or the error for the model
   */
  function findComputedTable(
    wanted: string,
  ):
    | { error: string }
    | { table: ProjectStore["tables"][number]; version: ComputedTable } {
    const table = projectStore
      .getState()
      .tables.find(({ id, name }) => id === wanted || name === wanted);
    const version = versionOf(table?.dataSource.source);
    if (table === undefined || table.dataSource.type !== computedTableType) {
      return {
        error: `Error: there is no temporary table computed by the chat called "${wanted}"${table === undefined ? "" : " (it was kept as a CSV file already)"}.`,
      };
    }
    if (version === undefined) {
      return {
        error: `Error: the values of "${table.name}" are gone: it was computed in an earlier session. Compute it again.`,
      };
    }
    return { table, version };
  }

  /**
   * Keeps a table computed by the chat as a CSV file next to the project file,
   * after asking the user; then the project reads it from that file, also
   * once saved and opened again, even without the chat
   *
   * @param wanted - The name or ID of the table
   * @param options - The name of the file, if given (else a useful one), and
   * whether the user chose it already, so that there is nothing more to ask
   * than whether to replace a file of that name
   * @returns The result for the model: "Done", "Note" when it was not kept
   * (e.g. by the user's choice), or "Error"
   */
  async function keepComputedTable(
    wanted: string,
    options?: { fileName?: string; chosen?: boolean },
  ): Promise<string> {
    const found = findComputedTable(wanted);
    if ("error" in found) {
      return found.error;
    }
    const { table, version } = found;
    const checked =
      options?.fileName === undefined
        ? { name: suggestCsvName(table.name, [...version.columns.keys()]) }
        : csvFileName(options.fileName);
    if ("error" in checked) {
      return `Error: ${checked.error}.`;
    }
    const fileName = checked.name;
    const workspace = keepTarget();
    if (workspace === undefined) {
      const downloaded = downloadComputedTable(table.id, fileName);
      return downloaded.startsWith("Error")
        ? downloaded
        : `Done: the project was not opened from a connected folder, so "${table.name}" cannot be written next to it; it was downloaded instead. ${downloaded.replace(/^Done: /, "")}`;
    }
    if (!idsSurviveCsv(version.ids)) {
      return `Note: "${table.name}" cannot be kept as a CSV file: its item IDs are text that looks like numbers, which a CSV file would turn into numbers, so the table would not match its items any more. It stays temporary; Download CSV still gives the file.`;
    }
    const allowed =
      options?.chosen === true ||
      window.confirm(
        `Save the table "${table.name}", computed by the chat, as the file computed/${fileName} next to the project file, in the folder "${workspace.name}"?\n\n` +
          "Then the project reads the table from that file, also after it is saved and opened again.",
      );
    if (!allowed) {
      return `Note: the user did not want "${table.name}" to be saved as a file. It stays temporary; do not ask again unless the user does.`;
    }
    const where = SourceUtils.normalizeSource(
      `computed/${fileName}`,
      workspace,
      projectStore.getState().source,
    );
    try {
      const folder = workspace as FileSystemDirectoryHandle & {
        requestPermission?: (options: {
          mode: "readwrite";
        }) => Promise<PermissionState>;
      };
      if (
        folder.requestPermission !== undefined &&
        (await folder.requestPermission({ mode: "readwrite" })) !== "granted"
      ) {
        return `Note: the browser was not allowed to write into the folder "${workspace.name}", so "${table.name}" stays temporary.`;
      }
      const segments = SourceUtils.getPathSegments(where);
      let directory: FileSystemDirectoryHandle = workspace;
      for (const [index, segment] of segments.slice(0, -1).entries()) {
        directory = await directory.getDirectoryHandle(segment, {
          // only the folder "computed" is made
          create: index === segments.length - 2,
        });
      }
      const exists = await directory.getFileHandle(fileName).then(
        () => true,
        () => false,
      );
      if (
        exists &&
        !window.confirm(
          `The folder "computed" next to the project file already has a file ${fileName}. Replace it with "${table.name}"?`,
        )
      ) {
        return `Note: "${table.name}" was not saved: the file computed/${fileName} already exists, and the user did not want to replace it. It stays temporary; another file name would do.`;
      }
      const file = await directory.getFileHandle(fileName, { create: true });
      const writable = await file.createWritable();
      await writable.write(computedTableCsv(version));
      await writable.close();
    } catch (error) {
      return `Error: "${table.name}" could not be written: ${error instanceof Error ? error.message : String(error)}. It stays temporary.`;
    }
    projectStore.getState().updateTable(table.id, {
      dataSource: {
        type: "csv",
        source: `computed/${fileName}`,
        idColumn: "id",
      } as TableDataSource,
    });
    tables.info.delete(table.id);
    readTables();
    return `Done: "${table.name}" was saved as computed/${fileName} next to the project file, and the project now reads it from there. Save the project to keep this; the table is not temporary any more, and Undo does not take it back.`;
  }

  /**
   * Downloads a table computed by the chat as a CSV file, with the item IDs
   * in the column "id"
   *
   * @param wanted - The name or ID of the table
   * @param given - The name of the file, else a useful one
   * @returns The result for the model
   */
  function downloadComputedTable(wanted: string, given?: string): string {
    const found = findComputedTable(wanted);
    if ("error" in found) {
      return found.error;
    }
    const { table, version } = found;
    const checked =
      given === undefined
        ? { name: suggestCsvName(table.name, [...version.columns.keys()]) }
        : csvFileName(given);
    if ("error" in checked) {
      return `Error: ${checked.error}.`;
    }
    const fileName = checked.name;
    const url = URL.createObjectURL(
      new Blob([computedTableCsv(version)], { type: "text/csv" }),
    );
    const link = document.createElement("a");
    link.download = fileName;
    link.href = url;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
    return `Done: "${table.name}" was downloaded as ${fileName}, with the item IDs in the column "id". In TissUUmaps the table stays temporary.`;
  }

  /**
   * Shows a script or the code of the model in the chat before it runs, as
   * one row like those of the tools, e.g. "📜 counter/scripts/count.js",
   * whose code opens with a click
   *
   * @param name - The name of the row, e.g. "📜 counter/scripts/count.js"
   * @param tooltip - What the row is, when the pointer is on it
   * @param code - The code
   * @returns Opens the code and keeps it open, e.g. while the user is asked
   * whether the script may run
   */
  function showScript(name: string, tooltip: string, code: string): () => void {
    const message = document.createElement("div");
    message.dataset.who = "Script";
    message.style.cssText =
      "padding: 4px 8px; border: 1px solid; border-radius: 6px;" +
      messageStyles.tool;
    const details = document.createElement("details");
    details.open = showToolDetails;
    const line = document.createElement("summary");
    line.style.cssText =
      "cursor: pointer; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;";
    line.textContent = name;
    line.title = tooltip;
    const pre = document.createElement("pre");
    pre.dataset.details = "true";
    pre.textContent = code;
    pre.style.cssText =
      "margin: 0; max-height: 240px; overflow: auto; white-space: pre;" +
      "font-family: monospace; font-size: 12px;";
    const copyButton = makeCopyButton(() => code, "the code");
    copyButton.style.alignSelf = "flex-end";
    const body = document.createElement("div");
    body.style.cssText =
      "display: flex; flex-direction: column; margin-top: 4px;";
    body.append(copyButton, pre);
    details.append(line, body);
    message.append(details);
    addToHistory(message);
    const row = { details, keepOpen: false };
    toolRows.push(row);
    return () => {
      row.keepOpen = true;
      details.open = true;
      history.scrollTop = history.scrollHeight;
    };
  }

  /**
   * Runs a script of a skill in the sandbox: shows its code, asks the user
   * the first time for a skill that came with the project, and shows how
   * long it has been running
   *
   * @param skill - The skill
   * @param path - The path of the script in the skill
   * @param args - What the script gets as tm.args
   * @param options - Optional abort signal, which stops the script, and
   * where to show the progress
   * @returns The result for the model
   * @throws If the script takes too long, or is stopped
   */
  async function runSkillScript(
    skill: Skill,
    path: string,
    args: unknown,
    options?: RunOptions,
  ): Promise<string> {
    const { signal: stop } = options ?? {};
    stop?.throwIfAborted();
    const code = (await skill.files.get(path)?.text()) ?? "";
    stop?.throwIfAborted();
    const python = path.endsWith(".py");
    const openCode = showScript(
      `📜 ${skill.name}/${path}`,
      `${path}, a ${python ? "Python" : "JavaScript"} script of the skill "${skill.name}": click to read its code`,
      code,
    );
    const from = skillList.projectSourceOf(skill);
    const key = `${projectStore.getState().instanceId}:${skill.name}`;
    if (from !== undefined && !allowedScripts.has(key)) {
      // the user must be able to read the code before saying yes
      openCode();
      // time to show the code before the question, which blocks the page
      await new Promise((resolve) => setTimeout(resolve, 100));
      const allowed = window.confirm(
        `The skill "${skill.name}" came with the open project (from ${from}). Run its script ${path}?\n\n` +
          "Its code is shown in the chat. Scripts run in a sandbox: they can read the data of the project, " +
          "but cannot change it, use the network or see your API key. You are asked once for each skill of the project.",
      );
      if (!allowed) {
        return `Error: the user did not allow the scripts of the skill "${skill.name}", which came with the project. Tell the user, and do not run them again unless they ask.`;
      }
      allowedScripts.add(key);
    }
    const names = {
      short: path,
      full: `${path} of the skill "${skill.name}"`,
      // e.g. "cluster-centers centers", for cluster-centers/scripts/centers.py
      chart: `${skill.name} ${path.replace(/^.*\//, "").replace(/\.\w+$/, "")}`,
    };
    return python
      ? runInPython(code, args, skill, names, kernelSource(skill), {
          ...options,
          path,
        })
      : runInSandbox(code, args, skill, names, options);
  }

  /**
   * Where the Python code of a skill comes from: code that comes from
   * elsewhere than the user (a skill of the project) gets a Python of its
   * own, so that it cannot change what the code of the chat uses, e.g. by
   * changing numpy, nor the other way round
   *
   * @param skill - The skill, or none for the code of the model
   * @returns e.g. "chat", or "project:<project>:/xenium/skills/counter"
   */
  function kernelSource(skill?: Skill): string {
    const origin = skill === undefined ? undefined : skillList.originOf(skill);
    if (origin?.kind === "project") {
      return `project:${projectStore.getState().instanceId}:${origin.where}/${skill!.name}`;
    }
    if (origin?.kind === "github") {
      const { repo, path, commit } = origin.github;
      return `github:${repo}/${path}@${commit}`;
    }
    return "chat";
  }

  /** The packages of Pyodide, read once from its list of packages */
  let pyodideList: Promise<PyodidePackages> | undefined;

  /**
   * The packages of Pyodide, for telling which ones the code of a skill from
   * GitHub can import; the list (about 120 KB) is downloaded and checked like
   * the other files of Python, and kept
   *
   * @returns The packages
   * @throws If the list cannot be downloaded or checked
   */
  function pyodidePackageList(): Promise<PyodidePackages> {
    pyodideList ??= pythonFiles
      .get(lockFile, { signal })
      .then((buffer) => pyodidePackages(new TextDecoder().decode(buffer)));
    // a list that could not be had is asked for again the next time
    pyodideList.catch(() => (pyodideList = undefined));
    return pyodideList;
  }

  /**
   * What the model is told about a skill from GitHub that the user added
   *
   * @param preview - What the skill holds
   * @returns The result for the model
   */
  function describeGitHubSkill(preview: GitHubPreview): string {
    const { origin, python } = preview;
    const names = preview.skills.map(({ name }) => `"${name}"`).join(", ");
    const missing = python.packages
      .filter((needed) => needed.pyodide === undefined)
      .map(({ name }) => name);
    return [
      `Done: the user added ${preview.skills.length === 1 ? "the skill" : "the skills"} ${names} from github.com/${origin.repo}${origin.path === "" ? "" : `/${origin.path}`} ` +
        `(commit ${origin.commit.slice(0, 7)}, ${origin.license === undefined ? "without a license: tell the user that its rights of use are not given" : `license ${origin.license}`}), ` +
        `with ${preview.files.length} files. It is on: load it with load_skill.`,
      ...(preview.madeSkillMd
        ? [
            "The repository has no SKILL.md: the chat wrote a minimal one. Read its README and its code with read_skill_file before you use it.",
          ]
        : []),
      ...(python.pyodideError !== undefined
        ? [
            `Which packages Pyodide has could not be checked: ${python.pyodideError}.`,
          ]
        : missing.length > 0
          ? [
              `Python packages that Pyodide does not have: ${missing.join(", ")}.` +
                (python.broken.length > 0
                  ? ` So these files cannot run: ${python.broken.map(({ path }) => path).join(", ")}.`
                  : ""),
            ]
          : []),
    ].join(" ");
  }

  /**
   * Reads a skill from GitHub, shows the user what it holds in the chat, and
   * adds it if the user says yes
   *
   * @param address - The address of the repository, or of a folder of it
   * @param options - Optional abort signal, which stops the reading and the
   * question, and where to show the progress
   * @returns The result for the model
   */
  async function addGitHubSkill(
    address: string,
    options?: { signal?: AbortSignal; progress?: (text: string) => void },
  ): Promise<string> {
    const { signal: stop, progress } = options ?? {};
    let preview: GitHubPreview;
    try {
      preview = await readGitHubSkill(address, {
        pyodide: pyodidePackageList,
        ...(progress === undefined ? {} : { onProgress: progress }),
        ...(stop === undefined ? {} : { signal: stop }),
      });
    } catch (error) {
      stop?.throwIfAborted();
      return `Error: the skill cannot be read from GitHub: ${error instanceof Error ? error.message : String(error)}.`;
    }
    progress?.("waiting for the user to decide");
    const question = confirmGitHubSkill(preview, {
      ...(stop === undefined ? {} : { signal: stop }),
    });
    addToHistory(question.element);
    const { add, remember } = await question.answer;
    stop?.throwIfAborted();
    if (!add) {
      return `Nothing changed: the user did not add the skill from github.com/${preview.origin.repo}. Do not propose it again unless they ask.`;
    }
    await skillList.addSkills(preview.skills, {
      github: preview.origin,
      remember,
    });
    return describeGitHubSkill(preview);
  }

  /**
   * "Add skill from GitHub…" in the settings: asks for the address, then
   * shows the same question as when the chat proposes a skill
   */
  async function addGitHubSkillFromSettings(): Promise<void> {
    const address = window.prompt(
      "The address of a GitHub repository, or of a folder of one, e.g. https://github.com/owner/repository",
    );
    if (address === null || address.trim() === "") {
      return;
    }
    show("Note", `Reading ${address.trim()} on GitHub…`);
    const result = await addGitHubSkill(address.trim(), { signal });
    if (result.startsWith("Error")) {
      show("Error", result.replace(/^Error: /, ""));
    }
  }

  /**
   * Shows the user new instructions for a skill, and uses them if the user
   * says yes
   *
   * @param skill - The skill
   * @param skillMd - Its new SKILL.md
   * @param options - Optional abort signal, which stops the question
   * @returns The result for the model
   */
  async function proposeSkillMd(
    skill: Skill,
    skillMd: string,
    options?: { signal?: AbortSignal },
  ): Promise<string> {
    const { signal: stop } = options ?? {};
    if (skillList.originOf(skill).kind === "project") {
      return `Error: the skill "${skill.name}" came with the project, and its instructions cannot be changed here.`;
    }
    const question = confirmSkillMd(skill.name, skillMd, {
      ...(stop === undefined ? {} : { signal: stop }),
    });
    addToHistory(question.element);
    const yes = await question.answer;
    stop?.throwIfAborted();
    if (!yes) {
      return `Nothing changed: the user kept the old instructions of "${skill.name}".`;
    }
    try {
      await skillList.replaceSkillMd(skill, skillMd);
    } catch (error) {
      return `Error: the new SKILL.md cannot be used: ${error instanceof Error ? error.message : String(error)}. Nothing changed.`;
    }
    return `Done: the skill "${skill.name}" has the new instructions; they are used from now on.`;
  }

  /** The actions of the stores, for reading the code of a plugin */
  function storeActions(): StoreActions {
    const stores = services.stores;
    return {
      project: actionsOf(projectStore.getState()),
      other: [
        ...actionsOf(appStore.getState()),
        ...actionsOf(settingsStore.getState()),
        ...actionsOf(stores.dataStore.getState()),
      ],
    };
  }

  /**
   * Shows a plugin made by the chat in the chat, with its code, what it uses
   * and its delicate parts; runs it and opens its panel if the user says yes
   *
   * @param entry - The plugin
   * @param kind - Why the user is asked
   * @param options - Optional abort signal, which answers no
   * @returns Whether it runs now and the browser remembers it, or why not
   */
  async function askToRunPlugin(
    entry: ChatPluginEntry,
    kind: PluginQuestionKind,
    options?: { signal?: AbortSignal },
  ): Promise<
    { ran: true; remembered: boolean } | { ran: false; error?: string }
  > {
    const existing = services.plugins.get(entry.id);
    const question = confirmPlugin(
      entry,
      readPluginCode(entry.code, storeActions()),
      {
        kind,
        ...(kind !== "remembered" && existing !== undefined
          ? { replaces: existing.entry }
          : {}),
        remembered: existing?.remembered ?? false,
        ...(options?.signal === undefined ? {} : { signal: options.signal }),
      },
    );
    addToHistory(question.element);
    const { run, remember } = await question.answer;
    if (!run) {
      return { ran: false };
    }
    try {
      await services.plugins.install(entry, { remember });
      await services.plugins.open(entry.id);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      question.status.textContent = `Did not run: ${message}`;
      return { ran: false, error: message };
    }
    question.status.textContent =
      "Running: its panel is open, and it is in the plugins menu.";
    return {
      ran: true,
      remembered: services.plugins.get(entry.id)?.remembered ?? false,
    };
  }

  /**
   * Shows the user a plugin that the model wrote, and runs it if the user
   * says yes
   *
   * @param plugin - Its id, name, description and code
   * @param options - The skill it is based on, if any; an optional abort
   * signal, which answers no
   * @returns The result for the model
   */
  async function proposePlugin(
    plugin: { id: string; name: string; description: string; code: string },
    options?: { skill?: Skill; signal?: AbortSignal },
  ): Promise<string> {
    const { skill, signal: stop } = options ?? {};
    let origin: PluginOrigin | undefined;
    if (skill !== undefined) {
      const from = skillList.originOf(skill);
      origin = {
        skill: skill.name,
        ...(from.kind === "github" ? { github: from.github } : {}),
        ...(from.kind === "project" ? { project: from.where } : {}),
      };
    }
    const entry: ChatPluginEntry = {
      ...plugin,
      made: new Date().toISOString(),
      ...(origin === undefined ? {} : { origin }),
    };
    const replaces = services.plugins.get(entry.id) !== undefined;
    const outcome = await askToRunPlugin(entry, "proposed", {
      ...(stop === undefined ? {} : { signal: stop }),
    });
    stop?.throwIfAborted();
    if (outcome.ran) {
      return (
        `Done: the user ran the plugin "${entry.name}" (id ${entry.id})${replaces ? ", which replaces its earlier version" : ""}; its panel is open in TissUUmaps. ` +
        (outcome.remembered
          ? "The browser remembers it for the next sessions. "
          : 'It is kept for this session only, unless the user turns on "Remember" for it in the settings of the chat. ') +
        "Tell the user where to find it, and that what it changes in the project is not taken back by Undo in the chat."
      );
    }
    if (outcome.error !== undefined) {
      return `Error: the user said yes, but the plugin "${entry.name}" could not run: ${outcome.error}. Fix the code and propose it again with the same id.`;
    }
    return `Nothing changed: the user did not run the plugin "${entry.name}". Do not propose it again unless they ask.`;
  }

  /**
   * Runs code that the model wrote in the sandbox, after showing it
   *
   * @param title - What the code computes, in the words of the model
   * @param code - The code
   * @param language - JavaScript, or Python
   * @param options - Optional abort signal, which stops the code, and where
   * to show the progress
   * @returns The result for the model
   * @throws If the code takes too long, or is stopped
   */
  async function runCode(
    title: string,
    code: string,
    language: "javascript" | "python",
    options?: RunOptions & { skill?: Skill },
  ): Promise<string> {
    options?.signal?.throwIfAborted();
    const python = language === "python";
    const skill = options?.skill;
    const details = [
      ...(python ? ["Python"] : []),
      ...(skill === undefined ? [] : [`with ${skill.name}`]),
    ];
    showScript(
      `📜 ${persona.name}'s code: ${title}${details.length > 0 ? ` (${details.join(", ")})` : ""}`,
      `${python ? "Python" : "JavaScript"} that ${persona.name} wrote${skill === undefined ? "" : `, with the files of the skill "${skill.name}"`}: click to read it`,
      code,
    );
    const names = {
      short: "the code",
      full: `the code "${title}"`,
      chart: title,
    };
    return python
      ? runInPython(code, {}, skill, names, kernelSource(skill), options)
      : runInSandbox(code, {}, skill, names, options);
  }

  /**
   * Shows a file of text that code gives to the user, with "Save" and
   * "Copy"; nothing is saved unless the user clicks
   *
   * @param given - Its name, e.g. "zscores.csv"
   * @param text - Its text
   * @returns What the code gets back
   * @throws If the name or the text cannot be given
   */
  function offerDownload(
    given: unknown,
    text: unknown,
  ): { name: string; bytes: number } {
    const checked = downloadFileName(given);
    if ("error" in checked) {
      throw new Error(checked.error);
    }
    if (typeof text !== "string") {
      throw new Error("give the text of the file, e.g. the CSV as a string");
    }
    const { name } = checked;
    const ending = /\.[^.]+$/.exec(name)![0].toLowerCase();
    const file = new Blob([text], { type: downloadTypes[ending]! });
    if (file.size > maxDownloadBytes) {
      throw new Error(
        `the file is too big (${(file.size / 1e6).toFixed(1)} MB, at most ${maxDownloadBytes / 1e6} MB)`,
      );
    }
    const body = addFrame("File", "tool");
    const line = document.createElement("div");
    line.textContent = `📄 ${name} (${file.size < 1000 ? `${file.size} B` : `${Math.round(file.size / 1000)} KB`}), from the code: click Save to keep it`;
    const saveButton = makeButton(
      "Save",
      () => {
        const url = URL.createObjectURL(file);
        const link = document.createElement("a");
        link.download = name;
        link.href = url;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 60_000);
      },
      quietButtonStyle,
    );
    saveButton.title = `Save ${name}`;
    const buttons = document.createElement("div");
    buttons.style.cssText =
      "display: flex; gap: 6px; justify-content: flex-end; white-space: normal;";
    buttons.append(
      makeCopyButton(() => text, "the text of the file"),
      saveButton,
    );
    body.append(line, buttons);
    return { name, bytes: file.size };
  }

  /** The addresses of the images of the charts, freed with the chat */
  const chartUrls: string[] = [];

  /** The charts of this chat, "Chart 1" first, which the model can look at */
  const charts: {
    png: ArrayBuffer;
    description: string;
    width: number;
    height: number;
  }[] = [];

  /**
   * The images that go with the result of the tool that runs now, for the
   * model, in base64; taken by the tool's message
   */
  const pendingImages: string[] = [];

  /** The service of the turn in progress, which decides what the model sees */
  let turnProvider: Provider | null = null;

  /**
   * Gives the model the image of a chart with the result of look_at_chart
   *
   * @param chart - Its number, as the chat shows it; without it, the last
   * @returns The result for the model
   */
  function lookAtChart(chart?: number): string {
    if (charts.length === 0) {
      return "Error: no chart was drawn in this chat yet: draw one with Python first.";
    }
    const number = chart ?? charts.length;
    const found = charts[number - 1];
    if (found === undefined) {
      return `Error: there is no Chart ${number}: the charts of this chat are Chart 1 to Chart ${charts.length}.`;
    }
    pendingImages.push(toBase64(found.png));
    return `Done: the image of Chart ${number} (${found.width} × ${found.height} pixels) comes with this result; you see it in this turn only. Its description: ${found.description || "(no plots)"}`;
  }

  /**
   * Shows the charts that Python code drew, each in a message with its
   * image and the buttons Copy and Save; only images that are PNG files
   *
   * @param drawn - The images, and what each shows
   * @param title - What the code computed, for the names of the files
   * @returns What the charts show, for the model, whose images come with the
   * result if the user lets the model see them
   */
  function showCharts(
    drawn: NonNullable<ScriptOutcome["charts"]>,
    title: string,
  ): string {
    if (drawn.length === 0) {
      return "";
    }
    const shown = drawn.slice(0, maxCharts);
    const sent = turnProvider !== null && chartsSent(turnProvider);
    const lines = shown.map(({ png, description }, index) => {
      const checked = checkPng(png);
      if ("error" in checked) {
        return `- Not shown: ${checked.error}.`;
      }
      charts.push({ png, description, ...checked });
      const chartName = `Chart ${charts.length}`;
      if (sent) {
        pendingImages.push(toBase64(png));
      }
      const fileName = chartFileName(title, index + 1, shown.length);
      const image = new Blob([png], { type: "image/png" });
      const url = URL.createObjectURL(image);
      chartUrls.push(url);
      const body = addFrame(chartName, "tool");
      // all the charts have the same kind of message
      (body.parentElement ?? body).dataset.who = "Chart";
      const picture = document.createElement("img");
      picture.src = url;
      picture.alt = description;
      picture.title = description;
      // the charts have a white background, also in the dark theme
      picture.style.cssText =
        "display: block; max-width: 100%; height: auto; margin-top: 4px; background: white;";
      const saveButton = makeButton(
        "Save",
        () => {
          const link = document.createElement("a");
          link.download = fileName;
          link.href = url;
          link.click();
        },
        quietButtonStyle,
      );
      saveButton.title = `Save the chart as ${fileName}`;
      const copyButton = makeCopyButton(() => "", "the chart", image);
      const buttons = document.createElement("div");
      buttons.style.cssText =
        "display: flex; gap: 6px; justify-content: flex-end; white-space: normal;";
      buttons.append(copyButton, saveButton);
      body.append(buttons, picture);
      return `- ${chartName}, ${checked.width} × ${checked.height} pixels: ${description || "(no plots)"}`;
    });
    const seen = sent
      ? "their images come with this result, for this turn only"
      : turnProvider?.seesImages === true
        ? "you do not see them; say what they show only from these words, or call look_at_chart if the user asks you to look at one"
        : "you do not see them; say what they show only from these words";
    return `\n\nCharts, shown to the user in the chat (${seen}):\n${lines.join("\n")}`;
  }

  /**
   * The result of a script or of the code of the model, for the model
   *
   * @param outcome - What the sandbox gave back
   * @param name - The code, for the model (e.g. 'the code "cells per cluster"')
   * @param started - When the code started to run, in milliseconds
   * @param charts - What the charts that it drew show, if any
   * @returns The text, cut if it is too long
   */
  function scriptResult(
    outcome: ScriptOutcome,
    name: string,
    started: number,
    charts = "",
  ): string {
    const seconds = ((Date.now() - started) / 1000).toFixed(1);
    const logs =
      outcome.logs.length > 0 ? `\n\nLog:\n${outcome.logs.join("\n")}` : "";
    const text =
      outcome.error !== undefined
        ? `Error: ${name} failed: ${outcome.error}${logs}`
        : `Done: ran ${name} in ${seconds} s.\n\nResult:\n${outcome.result || "(nothing)"}${charts}${logs}`;
    return text.length > maxScriptText
      ? `${text.slice(0, maxScriptText)}\n\n(Cut: only the first ${maxScriptText} of ${text.length} characters.)`
      : text;
  }

  /**
   * Runs Python code: starts Python first if needed, which may download its
   * files, and loads the packages that the code imports. Neither counts for
   * the time limit.
   *
   * @param code - The code, with a function main(tm)
   * @param args - What the code gets as tm.args
   * @param skill - The skill of a script, for tm.read_file
   * @param names - The code in the progress row, for the model, and in the
   * names of the files of its charts
   * @param source - Where the code comes from, e.g. "chat": code from
   * elsewhere gets a new Python, so that it cannot change what the code
   * from here uses
   * @param options - Optional abort signal, which stops the code, and where
   * to show the progress
   * @returns The result for the model
   * @throws If the code takes too long, or is stopped
   */
  async function runInPython(
    code: string,
    args: unknown,
    skill: Skill | undefined,
    names: { short: string; full: string; chart: string },
    source: string,
    options?: RunOptions & { path?: string },
  ): Promise<string> {
    const { signal: stop, progress, changes, path } = options ?? {};
    // the files of the skill, which its code can import
    const files =
      skill === undefined
        ? undefined
        : Object.fromEntries(
            await Promise.all(
              [...skill.files].map(
                async ([file, content]) =>
                  [file, await content.arrayBuffer()] as const,
              ),
            ),
          );
    stop?.throwIfAborted();
    let step = "starting Python";
    let started: number | undefined;
    let reported = "";
    // e.g. "0.4" or "12"
    const megabytes = (bytes: number) =>
      bytes < 10e6 ? (bytes / 1e6).toFixed(1) : Math.round(bytes / 1e6);
    const update = () => {
      const pythonDownload = services.pythonDownload();
      progress?.(
        pythonDownload !== undefined
          ? `Downloading Python, ${megabytes(pythonDownload.done)} of ${megabytes(pythonDownload.total)} MB`
          : started === undefined
            ? step
            : `running ${names.short}, ${Math.round((Date.now() - started) / 1000)} s${reported === "" ? "" : `: ${reported}`}`,
      );
    };
    showPythonDownload = update;
    update();
    const ticker = setInterval(update, 1000);
    try {
      const kernel = await pythonKernel.kernel(source, { signal: stop });
      step = "loading the Python packages";
      update();
      const outcome = await kernel.run(
        {
          code,
          args,
          ...(files === undefined ? {} : { files }),
          ...(path === undefined ? {} : { path }),
          ...(skill === undefined || skillPackageName(skill) === undefined
            ? {}
            : { packageName: skillPackageName(skill)! }),
          project: JSON.parse(
            stateJson(projectStore.getState(), appStore.getState()),
          ) as unknown,
        },
        (question) => answerScript(question, skill, { signal: stop, changes }),
        {
          timeLimit: scriptTimeLimit,
          signal: stop,
          onProgress: (text) => {
            reported = text;
            update();
          },
          onRunning: () => {
            started = Date.now();
            update();
          },
        },
      );
      const charts = showCharts(outcome.charts ?? [], names.chart);
      return scriptResult(outcome, names.full, started ?? Date.now(), charts);
    } finally {
      clearInterval(ticker);
      showPythonDownload = null;
      void updatePythonFiles();
    }
  }

  /**
   * Runs a script or the code of the model in the sandbox, and shows how
   * long it has been running
   *
   * @param code - The code: a module whose default export is a function of tm
   * @param args - What the code gets as tm.args
   * @param skill - The skill of a script, for tm.readFile
   * @param names - The code in the progress row (e.g. "scripts/count.js"),
   * and for the model (e.g. 'scripts/count.js of the skill "counter"')
   * @param options - Optional abort signal, which stops the code, and where
   * to show the progress
   * @returns The result for the model
   * @throws If the code takes too long, or is stopped
   */
  async function runInSandbox(
    code: string,
    args: unknown,
    skill: Skill | undefined,
    names: { short: string; full: string },
    options?: RunOptions,
  ): Promise<string> {
    const { signal: stop, progress, changes } = options ?? {};
    const started = Date.now();
    let reported = "";
    const update = () =>
      progress?.(
        `running ${names.short}, ${Math.round((Date.now() - started) / 1000)} s${reported === "" ? "" : `: ${reported}`}`,
      );
    update();
    const ticker = setInterval(update, 1000);
    try {
      const outcome = await runScript(
        {
          code,
          args,
          project: JSON.parse(
            stateJson(projectStore.getState(), appStore.getState()),
          ) as unknown,
        },
        (question) => answerScript(question, skill, { signal: stop, changes }),
        {
          timeLimit: scriptTimeLimit,
          signal: stop,
          onProgress: (text) => {
            reported = text;
            update();
          },
        },
      );
      return scriptResult(outcome, names.full, started);
    } finally {
      clearInterval(ticker);
    }
  }

  /**
   * Runs one tool call of the model
   *
   * A long tool shows its progress in a row of the chat, with a button that
   * stops it.
   *
   * @param shownCall - The call as the Tool row shows it
   * @returns The result for the model, and a warning for the user if another
   * visible object can cover the change
   */
  async function runTool(
    provider: Provider,
    name: string,
    args: ToolArguments,
    changes: Change[],
    shownCall: string,
  ): Promise<{ result: string; warning: string }> {
    const offered =
      (!wholeAppTools.includes(name) || provider.wholeApp) &&
      (!imageTools.includes(name) || provider.seesImages);
    const tool =
      Object.hasOwn(tools, name) && offered ? tools[name] : undefined;
    if (tool === undefined) {
      return {
        result: `Error: there is no tool called "${name}".`,
        warning: "",
      };
    }
    const changesBefore = changes.length;
    // stopped by its Stop button, or when the panel closes
    const stopper = new AbortController();
    const stopWithPanel = () => stopper.abort();
    signal.addEventListener("abort", stopWithPanel, { once: true });
    let progressRow: { frame: HTMLElement; line: HTMLElement } | null = null;
    const progress = (text: string) => {
      if (progressRow === null) {
        const body = addFrame("Tool", "tool");
        const line = document.createElement("div");
        const stopButton = makeButton(
          "Stop",
          () => stopper.abort(),
          `${smallButtonStyle}margin-top: 4px;`,
        );
        stopButton.title = "Stop this tool";
        body.append(line, stopButton);
        progressRow = { frame: body.parentElement ?? body, line };
      }
      progressRow.line.textContent = `${shownCall}\n… ${text}`;
    };
    let result: string;
    try {
      result = await tool(
        args,
        {
          projectStore,
          dataStore,
          appStore,
          tables,
          changes,
          skills: skillList.activeSkills(),
          progress,
          runSkillScript,
          runCode,
          addGitHubSkill,
          proposeSkillMd,
          keepComputedTable,
          lookAtChart,
          proposePlugin,
          chatPlugins: () => services.plugins.list(),
        },
        { signal: stopper.signal },
      );
    } catch (error) {
      signal.throwIfAborted();
      result = stopper.signal.aborted
        ? "Stopped by the user before the end. Tell the user, and do not run it again unless they ask."
        : `Error: ${error instanceof Error ? error.message : String(error)}`;
    } finally {
      signal.removeEventListener("abort", stopWithPanel);
      (progressRow as { frame: HTMLElement } | null)?.frame.remove();
    }
    const newChanges = changes.slice(changesBefore);
    rereadTables(newChanges);
    const warning = overlapWarning(projectStore.getState(), newChanges);
    if (warning !== "") {
      // the model is told right after "Done", which small models notice best
      result = result.replace(
        /^Done:? ?/,
        `Done. Tell the user: ${warning} What changed: `,
      );
    }
    return { result, warning };
  }

  /**
   * Sends a request to the model, and runs the tools it calls
   *
   * @param example - A suggested request to send; without it, the text in the
   * text box
   */
  async function send(example?: string): Promise<void> {
    const text = (example ?? input.value).trim();
    if (text === "" || sendButton.disabled) {
      return;
    }
    // the service, model and key of this request, even if the menus change
    // while the model works
    const provider = currentProvider();
    const model = modelSelect.value;
    const key = keys.get(provider.id) ?? "";
    if (provider.needsKey && key === "") {
      show(
        "Error",
        `Paste your ${provider.name} API key into the "API key" field first. No key yet? Click "Help".`,
      );
      return;
    }
    if (example === undefined) {
      input.value = "";
    }
    // for the up and down arrows
    if (sent.at(-1) !== text) {
      sent.push(text);
    }
    sentIndex = sent.length;
    show("You", text);
    // to forget this turn if it fails
    const start = messages.length;
    messages.push({ role: "user", text });
    updateExamples();
    sendButton.disabled = true;
    sendButton.textContent = "Waiting...";
    updateUndoButtons();
    const failedCalls: string[] = [];
    const changes: Change[] = [];
    const projectAtStart = projectStore.getState().instanceId;
    // the tokens of all the requests of this turn
    const usage: Usage = { input: 0, cacheRead: 0, cacheWrite: 0, output: 0 };
    let requests = 0;
    // the same for the whole turn, as the context and the tools must be
    const search = searchAllowed(provider);
    turnProvider = provider;
    let searches = 0;
    let images = 0;
    const sources: Source[] = [];
    // the last answer of the model, under which the sources go
    let finalAnswer: HTMLElement | null = null;
    // shown for the whole turn, below the messages, until the final answer
    startThinking();
    try {
      // the same for every step of this turn
      const context = await turnContext(provider);
      const limit = provider.wholeApp ? maxSteps : maxStepsLocal;
      // one more request after the last step of tools, for the final answer
      for (let step = 1; step <= limit + 1; step++) {
        const answerNow = step > limit;
        // the answer is shown while it arrives, in a message made when the
        // first piece of text comes
        let setText: ((text: string) => void) | null = null;
        let answerFrame: HTMLElement | null = null;
        const answer: ChatAnswer = await askModel(
          provider,
          model,
          key,
          context,
          search,
          answerNow,
          (textSoFar) => {
            if (setText === null) {
              setText = addMessage(persona.name, "assistant", true);
              answerFrame = lastFrame;
            }
            setText(textSoFar);
          },
        );
        if ("error" in answer) {
          show("Error", answer.error);
          showServiceProblem(provider, answer.error);
          messages.length = start;
          break;
        }
        messages.push(answer.message);
        requests++;
        for (const kind of [
          "input",
          "cacheRead",
          "cacheWrite",
          "output",
        ] as const) {
          usage[kind] += answer.message.usage?.[kind] ?? 0;
        }
        const { webSearches = [] } = answer.message;
        if (webSearches.length > 0) {
          searches += webSearches.length;
          // above the answer, since the searches came before it
          showTool(
            "web_search",
            `web_search, run by ${provider.name}`,
            describeSearches(webSearches),
            answerFrame,
          );
        }
        sources.push(...(answer.message.sources ?? []));
        const toolCalls = answer.message.toolCalls;
        // a long turn of web searches goes on when it is sent again
        if (
          toolCalls.length === 0 &&
          answer.message.paused === true &&
          !answerNow
        ) {
          continue;
        }
        if (toolCalls.length === 0) {
          // small models sometimes answer with an empty text
          if (setText === null) {
            show(persona.name, "(no text answer)", "assistant", true);
            answerFrame = lastFrame;
          }
          finalAnswer = answerFrame;
          break;
        }
        if (answerNow) {
          // calls made anyway are not run, but each gets a result, as the
          // services want
          for (const call of toolCalls) {
            messages.push({
              role: "tool",
              callId: call.id,
              name: call.name,
              result:
                "Error: not run: this request had reached its limit of steps.",
            });
          }
          finalAnswer = answerFrame;
          show(
            "Error",
            `Stopped after ${limit} steps: the model did not give a final answer.`,
          );
          break;
        }
        let repeated = false;
        for (const call of toolCalls) {
          const { name, arguments: args } = call;
          const callText = describeCall(name, args);
          // an action is shown as it reads, with names instead of IDs
          const shownCall =
            name === "call_action"
              ? describeAction(projectStore.getState(), args)
              : name === "run_code"
                ? // the code itself has its own row, just above
                  describeCall(name, {
                    title: args["title"],
                    ...(args["language"] === undefined
                      ? {}
                      : { language: args["language"] }),
                  })
                : callText;
          let outcome = {
            result: "Error: this exact call already failed.",
            warning: "",
          };
          if (failedCalls.includes(callText)) {
            repeated = true;
          } else {
            pendingImages.length = 0;
            outcome = await runTool(provider, name, args, changes, shownCall);
          }
          // e.g. the image of a chart, which only the services that see
          // images get
          const toolImages = provider.seesImages ? pendingImages.splice(0) : [];
          pendingImages.length = 0;
          images += toolImages.length;
          if (outcome.result.startsWith("Error")) {
            failedCalls.push(callText);
          }
          showTool(name, shownCall, outcome.result);
          if (outcome.warning !== "") {
            show("Note", outcome.warning);
          }
          messages.push({
            role: "tool",
            callId: call.id,
            name,
            result: outcome.result,
            ...(toolImages.length === 0 ? {} : { images: toolImages }),
          });
        }
        if (repeated) {
          show(
            "Error",
            "Stopped: the model repeated a tool call that had already failed.",
          );
          break;
        }
        if (step === limit) {
          // the next request asks for the answer, with what was found so far
          const last = messages.at(-1);
          if (last?.role === "tool") {
            last.result += `\n\n${lastStepNote}`;
          }
          show(
            "Note",
            `The model used ${limit} steps of tools, the most for one request: it now answers with what it found.`,
          );
        }
      }
    } catch (error) {
      if (signal.aborted) {
        return;
      }
      messages.length = start;
      if (usesOllama(provider)) {
        // fetch fails when nothing answers: Ollama is not running, or blocks the page
        show("Error", await lostOllama());
      } else {
        // fetch fails with a TypeError when nothing answers
        const message =
          error instanceof TypeError
            ? notReachable(provider)
            : `Something went wrong: ${hideKey(error instanceof Error ? error.message : String(error), key)}`;
        show("Error", message);
        showServiceProblem(provider, message);
      }
    }
    stopThinking();
    turnProvider = null;
    // under the last answer, with the pages the model cited first
    const lastMessage = finalAnswer ?? history.lastElementChild;
    const unique = sources.filter(
      (source, index) =>
        sources.findIndex(({ url }) => url === source.url) === index,
    );
    if (unique.length > 0 && lastMessage !== null) {
      lastMessage.append(makeSourcesRow(unique.slice(0, 12)));
    }
    showUsage(provider, model, usage, requests, { searches, images });
    // kept even if something failed later on, but not if another project
    // was opened in the meantime
    if (
      changes.length > 0 &&
      projectStore.getState().instanceId === projectAtStart
    ) {
      undoStack.push(changes);
    }
    sendButton.disabled = false;
    sendButton.textContent = "Send";
    updateUndoButtons();
  }

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void send();
    } else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      browseSent(event);
    }
  });
  providerSelect.addEventListener("change", () => {
    const provider = currentProvider();
    saveSetting("provider", provider.id);
    settingsOpen = false;
    showKey();
    fillModelMenu();
    updateStatus();
    if (usesOllama(provider)) {
      void checkOllama();
    } else if (!keyStatus.has(provider.id)) {
      void checkKey();
    }
  });
  modelSelect.addEventListener("change", () => {
    saveSetting(`model.${currentProvider().id}`, modelSelect.value);
    updateStatus();
  });
  keyInput.addEventListener("input", onKeyInput);
  // when the key is pasted or typed and the field is left
  keyInput.addEventListener("change", () => void checkKey(true));
  rememberBox.addEventListener("change", onRememberChange);
  // the selection changes on the whole page, and the answers can scroll
  document.addEventListener("selectionchange", updateQuoteButton);
  history.addEventListener("scroll", updateQuoteButton);
  // a new theme gives the assistant another personality
  const unsubscribeSettings = settingsStore.subscribe((state) => {
    persona = personaFor(state.dark);
  });

  container.style.cssText =
    "display: flex; flex-direction: column; gap: 8px; height: 100%;" +
    "position: relative; color: var(--foreground);";
  container.append(
    topRow,
    summary,
    settingsBox,
    history,
    examplesRow,
    bottomRow,
    footer,
    quoteButton,
  );

  // the container is attached to the panel only after mount, and its parent
  // has to fill the panel for "height: 100%" to work
  const observer = new ResizeObserver(() => {
    if (container.parentElement !== null) {
      container.parentElement.style.height = "100%";
    }
  });
  observer.observe(container);

  const unsubscribeData = dataStore.subscribe(readTables);
  const unsubscribeProject = projectStore.subscribe(onProjectChange);
  // e.g. the columns that plugins made by the chat saved while it was closed
  linkComputedTables();
  readTables();
  showKey();
  fillModelMenu();
  updateStatus();
  void check();
  void updatePythonFiles();
  updateUndoButtons();
  updateExamples();

  return () => {
    abortController.abort();
    stopPython();
    stopListeningToDownload();
    pluginList.dispose();
    if (services.hooks.skillsChanged === updateLayout) {
      services.hooks.skillsChanged = undefined;
      services.hooks.addSkillFromGitHub = undefined;
    }
    for (const url of chartUrls) {
      URL.revokeObjectURL(url);
    }
    stopThinking();
    observer.disconnect();
    unsubscribeData();
    unsubscribeProject();
    unsubscribeSettings();
    document.removeEventListener("selectionchange", updateQuoteButton);
  };
}
