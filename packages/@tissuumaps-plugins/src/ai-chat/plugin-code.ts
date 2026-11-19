// Before the user runs a plugin made by the chat, the chat reads its code and
// says what it uses, with a warning for each delicate thing: the network, the
// memory of the browser (where the API key may be), changes to the project,
// code from the internet. This is only a reading of the text of the code: it
// finds the usual ways of doing these things, not every way, which the
// confirmation says.

/** The names of the actions of the stores, which change what they hold */
export type StoreActions = {
  /** The actions of the project store, e.g. "updateLabels" */
  project: string[];
  /** The actions of the other stores, e.g. "setDark" */
  other: string[];
};

/** What the chat found in the code of a plugin */
export type PluginCodeReport = {
  /** What it uses, in a few words each, e.g. "a panel" */
  uses: string[];
  /** The delicate things it does, each with what was found */
  warnings: { kind: WarningKind; text: string; found: string[] }[];
};

/** The delicate things that a plugin can do */
export type WarningKind =
  | "network"
  | "storage"
  | "network-and-storage"
  | "project"
  | "app"
  | "remote-code"
  | "hidden"
  | "plugins";

/** The ways to use the network, by what they look like in the code */
const networkPatterns: [RegExp, string][] = [
  [/\bfetch\s*\(/, "fetch"],
  [/\bXMLHttpRequest\b/, "XMLHttpRequest"],
  [/\bWebSocket\b/, "WebSocket"],
  [/\bEventSource\b/, "EventSource"],
  [/\bsendBeacon\b/, "navigator.sendBeacon"],
  [/\bRTCPeerConnection\b/, "RTCPeerConnection"],
  [/\bwindow\.open\s*\(/, "window.open"],
];

/** The ways to read or write the memory of the browser */
const storagePatterns: [RegExp, string][] = [
  [/\blocalStorage\b/, "localStorage"],
  [/\bsessionStorage\b/, "sessionStorage"],
  [/\bindexedDB\b/, "indexedDB"],
  [/\bdocument\.cookie\b/, "document.cookie"],
  [/\bcaches\s*\.\s*(open|match|keys|delete|has)\b/, "caches"],
];

/** The ways to load and run code from the internet */
const remoteCodePatterns: [RegExp, string][] = [
  [/\bimport\s*\(\s*(?!["'`]\.)/, "import() of a module"],
  [/\bimportScripts\s*\(/, "importScripts"],
  [/createElement\s*\(\s*["'`]script["'`]/, "a <script> element"],
  [/<script\b/i, "a <script> element"],
  [/\bnew\s+Worker\s*\(/, "a Web Worker"],
];

/** The ways to run code made while the plugin runs, or to hide names */
const hiddenPatterns: [RegExp, string][] = [
  [/\beval\s*\(/, "eval"],
  [/\bnew\s+Function\s*\(/, "new Function"],
  [/\b(window|globalThis|self|document)\s*\[/, "names built at run time"],
  [/\batob\s*\(/, "atob"],
  [/\bString\.fromCharCode\s*\(/, "String.fromCharCode"],
];

/**
 * Escapes a name for a regular expression
 *
 * @param name - The name
 * @returns The escaped name
 */
function escapeName(name: string): string {
  return name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Finds which patterns the code matches
 *
 * @param code - The code
 * @param patterns - The patterns, each with its name
 * @returns The names of those found, each once
 */
function findPatterns(code: string, patterns: [RegExp, string][]): string[] {
  return [
    ...new Set(
      patterns
        .filter(([pattern]) => pattern.test(code))
        .map(([, name]) => name),
    ),
  ];
}

/**
 * Finds the actions of the stores that the code calls
 *
 * @param code - The code
 * @param names - The names of the actions
 * @returns Those called, each once
 */
function findCalls(code: string, names: string[]): string[] {
  return names.filter((name) =>
    new RegExp(`\\b${escapeName(name)}\\s*\\(`).test(code),
  );
}

/**
 * Lists the actions of a store, from its state: the functions whose names
 * do not only read (get…, has…, find…)
 *
 * @param state - The state of the store
 * @returns The names of the actions
 */
export function actionsOf(state: object): string[] {
  return Object.entries(state)
    .filter(
      ([name, value]) =>
        typeof value === "function" && !/^(get|has|find|is)[A-Z]/.test(name),
    )
    .map(([name]) => name);
}

/**
 * The addresses on the internet in the code, e.g. of a library
 *
 * @param code - The code
 * @returns The addresses, each once, at most 10
 */
function findAddresses(code: string): string[] {
  return [...new Set(code.match(/\bhttps?:\/\/[^\s"'`<>)\]]+/g) ?? [])].slice(
    0,
    10,
  );
}

/**
 * Reads the code of a plugin made by the chat: what it uses, and the
 * delicate things it does
 *
 * @param code - The code of the plugin
 * @param actions - The actions of the stores, see {@link actionsOf}
 * @returns What it uses, and the warnings
 */
export function readPluginCode(
  code: string,
  actions: StoreActions,
): PluginCodeReport {
  const uses: string[] = [];
  if (/\bmount\s*[(:]/.test(code)) {
    uses.push("a panel of its own, opened from the plugins menu");
  }
  if (/\bsetup\s*[(:]/.test(code)) {
    uses.push("a setup, which runs as soon as the plugin is added");
  }
  const python = /language\s*:\s*["'`]python["'`]/.test(code);
  const javascript =
    /\bchat\.run\s*\(/.test(code) &&
    (/language\s*:\s*["'`]javascript["'`]/.test(code) || !python);
  if (python || javascript) {
    uses.push(
      `the sandbox of the chat, for ${[
        ...(javascript ? ["JavaScript"] : []),
        ...(python ? ["Python"] : []),
      ].join(" and ")}, which has no network and cannot change the project`,
    );
  }
  // skill: "anhood", or skill: SKILL with const SKILL = "anhood"
  const constants = new Map(
    [
      ...code.matchAll(
        /\b(?:const|let|var)\s+(\w+)\s*=\s*["'`]([\w.-]+)["'`]/g,
      ),
    ].map(([, name, value]) => [name!, value!]),
  );
  const skills = [
    ...new Set(
      [...code.matchAll(/\bskill\s*:\s*(?:["'`]([\w.-]+)["'`]|(\w+))/g)]
        .map(([, text, name]) => text ?? constants.get(name!))
        .filter((name): name is string => name !== undefined),
    ),
  ];
  if (skills.length > 0) {
    uses.push(`the files of the skill ${skills.join(", ")}`);
  }
  if (/\bchat\.(read|columns|tables)\s*\(/.test(code)) {
    uses.push("the data of the tables, read through the chat");
  }
  if (/\b(projectStore|getState|subscribe)\b/.test(code)) {
    uses.push("the state of TissUUmaps (the stores), read or watched");
  }
  if (/\bdark\b|var\(--/.test(code)) {
    uses.push("the colors of the theme");
  }
  if (/\bchat\.showShapes\s*\(/.test(code)) {
    uses.push(
      "temporary shapes on the image (chat.showShapes), removed with the plugin",
    );
  }
  if (/\b(createObjectURL|\.download\s*=)/.test(code)) {
    uses.push("files that it gives you to save (downloads)");
  }
  if (/\bnavigator\.clipboard\b/.test(code)) {
    uses.push("the clipboard");
  }
  if (/getContext\s*\(\s*["'`]2d|<svg|createElementNS/.test(code)) {
    uses.push("drawings (a canvas or SVG)");
  }

  const warnings: PluginCodeReport["warnings"] = [];
  const network = findPatterns(code, networkPatterns);
  const storage = findPatterns(code, storagePatterns);
  const addresses = findAddresses(code);
  if (network.length > 0) {
    warnings.push({
      kind: "network",
      text: "It uses the network: it can send data (also the data of the project) to other computers, and receive data from them.",
      found: [...network, ...addresses],
    });
  }
  if (storage.length > 0) {
    warnings.push({
      kind: "storage",
      text: 'It uses the memory of the browser, where the chat keeps your API keys if you chose "Remember key on this computer".',
      found: storage,
    });
  }
  if (network.length > 0 && storage.length > 0) {
    warnings.push({
      kind: "network-and-storage",
      text: "It uses both the network and the memory of the browser: it could send your API keys to others. Run it only if you trust it fully.",
      found: [],
    });
  }
  const projectCalls = findCalls(code, actions.project);
  const saves = /\bchat\.saveColumn\s*\(/.test(code);
  if (projectCalls.length > 0 || saves || /\bsetState\s*\(/.test(code)) {
    warnings.push({
      kind: "project",
      text: 'It changes the project. "Undo" in the chat does not take back what a plugin changes: undo it by hand, or reload the project without saving it.',
      found: [
        ...projectCalls,
        ...(saves ? ["chat.saveColumn (adds columns to a table (chat))"] : []),
        ...(/\bsetState\s*\(/.test(code) ? ["setState"] : []),
      ],
    });
  }
  const appCalls = findCalls(code, actions.other).filter(
    (name) => !actions.project.includes(name),
  );
  if (appCalls.length > 0) {
    warnings.push({
      kind: "app",
      text: "It changes the view or the settings of TissUUmaps.",
      found: appCalls,
    });
  }
  const remote = findPatterns(code, remoteCodePatterns);
  if (remote.length > 0) {
    warnings.push({
      kind: "remote-code",
      text: "It can load and run more code, e.g. a library from the internet, which you do not see here.",
      found: [...remote, ...(network.length > 0 ? [] : addresses)],
    });
  }
  const hidden = findPatterns(code, hiddenPatterns);
  if (hidden.length > 0) {
    warnings.push({
      kind: "hidden",
      text: "It builds code or names while it runs, which hides what it does from this summary.",
      found: hidden,
    });
  }
  if (
    /\btissuumaps\s*[?!]?\.\s*(register|unregister|mount|unmount)Plugin/.test(
      code,
    )
  ) {
    warnings.push({
      kind: "plugins",
      text: "It adds, opens or closes other plugins.",
      found: ["window.tissuumaps"],
    });
  }
  return { uses, warnings };
}
