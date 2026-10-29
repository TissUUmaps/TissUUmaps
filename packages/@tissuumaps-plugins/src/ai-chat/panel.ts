import type { PluginStores, ProjectStore } from "@tissuumaps/core";

import { callActionDefinition, describeAction } from "./call-action";
import { overlapWarning } from "./color-tools";
import { helpText } from "./help";
import {
  actionLimits,
  instructions,
  toolLimits,
  wholeAppInstructions,
} from "./instructions";
import { renderMarkdown } from "./markdown";
import { findById, isMapKind, same } from "./objects";
import { personaFor } from "./persona";
import {
  type ChatAnswer,
  type ChatMessage,
  type OllamaPullUpdate,
  type Provider,
  type ProviderId,
  type Usage,
  explainNotReachable,
  hideKey,
  notRunningMessage,
  ollamaUrl,
  providers,
  readJsonLines,
} from "./providers";
import { loadSetting, saveSetting } from "./settings";
import { buildContext, exampleRequests, stateJson } from "./summary";
import {
  describeMatrix,
  describeValues,
  findUsedColumns,
  linkedTables,
  listColumns,
  maxRowsToRead,
  objectTables,
} from "./tables";
import { toolDefinitions, tools } from "./tools";
import type {
  Change,
  ColumnInfo,
  ItemsObject,
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
  makeThinkingIndicator,
  messageStyles,
  notReachable,
  quietTextStyle,
  smallButtonStyle,
  toolIcons,
  usesOllama,
} from "./ui";
import { currentValue, describeValueChange, handNote, takeBack } from "./undo";

/** How many times in a row the model may call tools before it is stopped */
const maxSteps = 5;

/**
 * Mounts the chat panel
 *
 * @param container - The element to mount the panel into
 * @param stores - The stores of the application
 * @returns A callback that unmounts the panel
 */
export function mountAiChat(
  container: HTMLElement,
  { projectStore, dataStore, settingsStore, appStore }: PluginStores,
): () => void {
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

  const settingsBox = document.createElement("div");
  settingsBox.style.cssText =
    "display: flex; flex-direction: column; gap: 8px;";
  settingsBox.append(serviceRow, keyRow, statusRow, toolDetailsLabel);

  const history = document.createElement("div");
  history.style.cssText =
    "flex: 1; min-height: 100px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px;" +
    "padding: 8px; border: 1px solid var(--border); border-radius: 6px;";

  // one row above the text box, which scrolls sideways if it is too long
  const examplesRow = document.createElement("div");
  examplesRow.style.cssText =
    "display: flex; gap: 6px; overflow-x: auto; padding-bottom: 2px;";

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
  footer.textContent = credits;
  footer.style.cssText = `${quietTextStyle}font-size: 11px; text-align: right;`;

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
      for (const { name, matrix, hidden } of await listColumns(data, {
        signal,
      })) {
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
      saveTableInfo(tableId, projectId, {
        status: "ready",
        rows,
        tooBig,
        useful,
        other,
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
          `${smallButtonStyle}flex: none; white-space: nowrap;`,
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
  }

  /** The tool rows, which open and close with "Show tool details" */
  const toolRows: { details: HTMLDetailsElement; failed: boolean }[] = [];

  /**
   * Shows a tool call as one folded row: an icon and the first line of the
   * result, e.g. "🌫 Done: opacity of cell_labels 1 → 0.5"; opened, the call
   * and the whole result. Failed calls are always open.
   *
   * @param name - The tool
   * @param shownCall - The call, as it reads
   * @param result - The result for the model
   */
  function showTool(name: string, shownCall: string, result: string): void {
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
    addToHistory(message);
    toolRows.push({ details, failed });
  }

  /** Opens or folds the rows of the tools, as "Show tool details" says */
  function updateToolRows(): void {
    for (const { details, failed } of toolRows) {
      details.open = failed || showToolDetails;
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
    if (kind === "assistant") {
      // a part of the answers can be quoted
      body.dataset.quotable = "true";
    }
    return (text) => {
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
    toolRows.length = 0;
    updateExamples();
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
        lines.push(
          `${change.name} is not in the project any more, so it was skipped`,
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
      return;
    }
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
    summary.textContent = `${provider.name} · ${modelSelect.value} · key ✓`;
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
      ? `${instructions.replace(toolLimits, actionLimits)}\n\n${wholeAppInstructions}`
      : instructions;
  }

  /** The tools for a service: for the strong models, also call_action */
  function toolsFor(provider: Provider) {
    return [
      ...toolDefinitions,
      ...(provider.wholeApp ? [callActionDefinition] : []),
    ].map((definition) => definition.function);
  }

  /**
   * The context as it is now: for the strong models, with the app state as
   * JSON after the summary
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
    const state = stateJson(projectStore.getState(), appStore.getState());
    return `${summary}\n\n# The current app state, as JSON\n${state}`;
  }

  /**
   * Sends the instructions, the context of the turn, the conversation and the
   * tools to the model
   *
   * @param provider - The service
   * @param model - The model
   * @param key - The API key of the service, or ""
   * @param context - The context of the turn, see {@link turnContext}
   * @param onText - Called with the text so far, each time a piece arrives
   * @returns The answer, or an error message
   */
  async function askModel(
    provider: Provider,
    model: string,
    key: string,
    context: string,
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
      },
      onText,
      { signal },
    );
  }

  /**
   * Shows under the last message of a turn the tokens of its requests, and
   * for the services that bill them, about how much they cost, also in total
   * since the panel was opened
   */
  function showUsage(
    provider: Provider,
    model: string,
    usage: Usage,
    requests: number,
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
    const prices = provider.models.find(({ name }) => name === model)?.prices;
    if (prices === undefined) {
      parts.push("free");
    } else {
      const cost =
        (usage.input * prices.input +
          usage.cacheRead * prices.cacheRead +
          usage.cacheWrite * prices.cacheWrite +
          usage.output * prices.output) /
        1_000_000;
      sessionCost += cost;
      parts.push(
        `≈ ${dollars(cost)}`,
        `this session ≈ ${dollars(sessionCost)}`,
      );
    }
    const line = document.createElement("div");
    line.dataset.usage = "true";
    line.style.cssText = `${quietTextStyle}font-size: 11px; margin-top: 4px;`;
    line.textContent = `Tokens${requests > 1 ? ` (${requests} requests)` : ""}: ${parts.join(" · ")}`;
    message.append(line);
  }

  /**
   * Runs one tool call of the model
   *
   * @returns The result for the model, and a warning for the user if another
   * visible object can cover the change
   */
  async function runTool(
    provider: Provider,
    name: string,
    args: ToolArguments,
    changes: Change[],
  ): Promise<{ result: string; warning: string }> {
    const offered = name !== "call_action" || provider.wholeApp;
    const tool =
      Object.hasOwn(tools, name) && offered ? tools[name] : undefined;
    if (tool === undefined) {
      return {
        result: `Error: there is no tool called "${name}".`,
        warning: "",
      };
    }
    const changesBefore = changes.length;
    let result: string;
    try {
      result = await tool(
        args,
        { projectStore, dataStore, appStore, tables, changes },
        { signal },
      );
    } catch (error) {
      signal.throwIfAborted();
      result = `Error: ${error instanceof Error ? error.message : String(error)}`;
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
    // shown for the whole turn, below the messages, until the final answer
    startThinking();
    try {
      // the same for every step of this turn
      const context = await turnContext(provider);
      for (let step = 1; step <= maxSteps; step++) {
        // the answer is shown while it arrives, in a message made when the
        // first piece of text comes
        let setText: ((text: string) => void) | null = null;
        const answer: ChatAnswer = await askModel(
          provider,
          model,
          key,
          context,
          (textSoFar) => {
            setText ??= addMessage(persona.name, "assistant", true);
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
        const toolCalls = answer.message.toolCalls;
        if (toolCalls.length === 0) {
          // small models sometimes answer with an empty text
          if (setText === null) {
            show(persona.name, "(no text answer)", "assistant", true);
          }
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
              : callText;
          let outcome = {
            result: "Error: this exact call already failed.",
            warning: "",
          };
          if (failedCalls.includes(callText)) {
            repeated = true;
          } else {
            outcome = await runTool(provider, name, args, changes);
          }
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
          });
        }
        if (repeated) {
          show(
            "Error",
            "Stopped: the model repeated a tool call that had already failed.",
          );
          break;
        }
        if (step === maxSteps) {
          show(
            "Error",
            `Stopped after ${maxSteps} steps without a final answer.`,
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
    showUsage(provider, model, usage, requests);
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
  readTables();
  showKey();
  fillModelMenu();
  updateStatus();
  void check();
  updateUndoButtons();
  updateExamples();

  return () => {
    abortController.abort();
    stopThinking();
    observer.disconnect();
    unsubscribeData();
    unsubscribeProject();
    unsubscribeSettings();
    document.removeEventListener("selectionchange", updateQuoteButton);
  };
}
