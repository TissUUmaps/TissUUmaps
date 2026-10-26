// the store types of @tissuumaps/core use Zustand's Immer middleware
import type {} from "zustand/middleware/immer";

import {
  type AppStore,
  type AppStoreApi,
  type Color,
  type ColorConfig,
  type Config,
  type DataStoreApi,
  type GroupValueMap,
  type Image,
  ImageChannelViewMode,
  type ImageData,
  ImageUtils,
  type Labels,
  type Layer,
  Marker,
  type Plugin,
  type PluginStores,
  type Points,
  type ProjectStore,
  type ProjectStoreApi,
  type Shapes,
  type SizeConfig,
  type TableColumnRef,
  type TableData,
  TableUtils,
  type TypedArrayOrArray,
  type VisibilityConfig,
  categoricalColorPalettes,
  continuousColorPalettes,
  getActiveConfigSource,
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
  isRandomConfig,
} from "@tissuumaps/core";

import { type Persona, personaFor, thinkingWords } from "./persona";
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
import {
  type Summary,
  benjaminiHochberg,
  formatP,
  log2FoldChange,
  mannWhitneySorted,
  spearman,
  summarize,
  summarizeSorted,
} from "./stats";
import { storeProperties, storeTypes } from "./store-types.generated";

// --- Settings and help ---

/** How many times in a row the model may call tools before it is stopped */
const maxSteps = 5;

/** The start of the localStorage keys under which the panel remembers settings */
const storagePrefix = "tissuumaps.ai-chat.";

/**
 * Reads a setting remembered in an earlier session, e.g. the chosen service
 *
 * @param name - The name of the setting, e.g. "provider" or "model.openai"
 * @returns The value, or `null` if none was remembered or the browser does
 * not allow storage (e.g. in a private window)
 */
function loadSetting(name: string): string | null {
  try {
    return window.localStorage.getItem(storagePrefix + name);
  } catch {
    return null;
  }
}

/**
 * Remembers a setting for the next session, or forgets it, if the browser
 * allows it
 *
 * @param name - The name of the setting
 * @param value - The value, or `null` to forget it
 */
function saveSetting(name: string, value: string | null): void {
  try {
    if (value === null) {
      window.localStorage.removeItem(storagePrefix + name);
    } else {
      window.localStorage.setItem(storagePrefix + name, value);
    }
  } catch {
    // not remembered: the default is used next time
  }
}

/** The text shown by the "Help" button */
const helpText = `How to use this chat

1. Choose a service
   - OpenAI or Anthropic: fast and good at following requests, on any computer,
     also when TissUUmaps is opened from a website. You need your own API key
     (see 2), and every request is billed to your account.
   - Ollama (local): runs on this computer. Free, private and works offline,
     but it can be slow on a computer without a good graphics card (see 4 and 5).
   - Ollama (cloud): runs on Ollama's servers, through the Ollama app on this
     computer. Free with an Ollama account (see 6).
   With OpenAI, Anthropic and Ollama (cloud), your messages and a summary of the
   open project are sent to that service.

2. OpenAI and Anthropic: your own API key
   An API key is a secret code that lets this chat use your account. TissUUmaps
   does not provide keys: everyone uses their own. Using the models costs money,
   so first add a payment method or buy credit in your account. With the first
   model in the menu, a request usually costs well under one cent; the other
   models cost more.
   - OpenAI: sign in at https://platform.openai.com, open
     https://platform.openai.com/api-keys and click "Create new secret key".
   - Anthropic: sign in at https://platform.claude.com, open
     https://platform.claude.com/settings/keys and click "Create key". Choose one
     workspace for the key (e.g. the Default Workspace) and when it expires.
   Copy the key, choose the service in the menu and paste the key into the
   "API key" field. The website shows the key only once: if you lose it, make a
   new one. When the key is accepted, the settings fold into one short line;
   ⚙ opens them again, to change, forget or check the key. They also open by
   themselves when something goes wrong.
   Each request is billed to your account, and your messages and the project
   summary are sent to the service you chose. Under each answer, a small line
   shows the tokens used (input, read from and written to the service's cache,
   output) and about how much they cost; the bill of the service counts.
   These models also get the types and the current state of the whole app,
   so they can answer detailed questions about the settings of each object,
   and change most of them: e.g. the color of an image channel, the shape of
   the points, a rotation, the order of the objects, a highlighted group.
   Every change can be undone. They never delete or add objects, and they
   rename something only when you ask.

3. Keeping your key safe
   - Treat the key like a password: do not share it, and do not put it in
     e-mails, files or screenshots.
   - Make a key just for TissUUmaps, and set a monthly spending limit in your
     account (under Limits), so that a stolen key cannot cost much.
   - If someone else may have your key, delete it on the page where you made it:
     it stops working at once. Then click "Forget key" here.
   - The key is only sent to the service you chose. Without "Remember key on this
     computer", it is forgotten when you close the page. With it, the key is
     saved in this browser, where anyone who uses this browser on this computer,
     and any browser extension or script that can read this page, could find it:
     do not use it on a shared computer. "Forget key" removes the key.

4. Install Ollama
   Download it from https://ollama.com/download and install it like any other program.
   It then runs in the background. If the status line says "not running", start Ollama
   from the Start menu (Windows) or the Applications folder (Mac), then click "Check again".

5. Choose an Ollama model
   - "(local)": runs on this computer.
   - "(not downloaded)": choose it and click "Download". This happens only once
     and takes a few GB of disk space.
   - "(online)", with "Ollama (cloud)": runs on Ollama's servers. Fast on any
     computer, but your messages and the project summary are sent to Ollama.
     gemma4:cloud is included in the free plan.

6. Turn on Ollama's online models
   Open the Ollama app, go to Settings and sign in. You can create a free account there.
   (If you use the terminal, "ollama signin" does the same.)

7. Undo
   "Undo" takes back the changes of your last request; click it again to go further back.
   "Undo all" takes back every change made by the chat. Changes you made by hand
   to other settings stay as they are.

8. Using Ollama when TissUUmaps is opened from a website (e.g. https://tissuumaps.github.io)
   OpenAI and Anthropic need nothing for this. For safety, Ollama only accepts
   pages opened from this computer (localhost). If the status line says that
   Ollama is running but blocks this website, give the website's address to
   Ollama with the setting OLLAMA_ORIGINS:
   - Windows: quit Ollama (right-click its icon next to the clock, then Quit).
     Open Settings, search for "environment variables" and choose "Edit environment
     variables for your account". Click "New", enter the name OLLAMA_ORIGINS and the
     value https://tissuumaps.github.io, then click OK. Start Ollama again from the
     Start menu.
   - Mac: in the Terminal app, run
       launchctl setenv OLLAMA_ORIGINS "https://tissuumaps.github.io"
     then quit the Ollama app and start it again. This setting is lost when the
     Mac restarts, so run the command again after a restart.
   Then click "Check again". If the browser asks whether the page may access devices
   on your local network, allow it: otherwise the browser itself blocks the page.

9. New chat, suggestions and shortcuts
   "New chat" starts a new conversation: the model forgets the earlier messages.
   Undo still works for the changes made before.
   Above the text box, a few suggested requests for the open project are shown:
   click one to send it. What you are typing stays as it is.
   In the text box, the up and down arrows go through the messages you sent
   before, like in a terminal.
   Select a part of an answer and click "Quote" to quote it in your next message.
   While the model thinks, a word in Italian changes every few seconds: point
   at it to see what it means in English.

10. Dia and Aid
   With the light theme of TissUUmaps, the assistant is Dia: gentle and kind.
   With the dark theme, it is Aid: serious and a little mysterious. Only the
   tone changes, at once when you switch the theme: what the assistant can do,
   and its rules (such as telling the truth about what it changed), stay the same.

What the assistant can do: answer questions about the open project, show or hide
images, labels, points and shapes, make them more or less transparent, make
points bigger or smaller, color labels and points by a column of their table
(e.g. "color the cell labels by leiden", "cluster 3 in red, the others grey"),
give them one color (e.g. "color the cell circles green"),
and show only some groups (e.g. "show only clusters 1 and 3", then "show all the
cells again"). It cannot change or delete anything else.`;

// --- Tools ---

/** The arguments of a tool call, as the model sends them */
type ToolArguments = Record<string, unknown>;

/** The sections of the project that the tools work on */
type ObjectKind = "images" | "labels" | "points" | "shapes";

/** The sections whose items can be colored and filtered by a table column */
type ItemsKind = "labels" | "points";

/**
 * The lists of maps of the project: the color and filter tools write color
 * and visibility maps, call_action all of them
 */
type MapKind =
  "markerMaps" | "sizeMaps" | "colorMaps" | "visibilityMaps" | "opacityMaps";

/** The lists of the project whose entries can be changed */
type ListKind = ObjectKind | "layers" | MapKind;

/** An object of the project that the tools can change */
type ProjectObject = Image | Labels | Points | Shapes;

/** An object whose items can be colored and filtered by a table column */
type ItemsObject = Labels | Points;

/** An object that a tool works on, with its section */
type Target = { kind: ObjectKind; object: ProjectObject };

/** An object whose items a color or filter tool works on, with its section */
type ItemsTarget = { kind: ItemsKind; object: ItemsObject };

/** One change made by a tool, remembered so that "Undo" can take it back */
type Change = {
  /**
   * Without it, one property changed; "added": a map was added, which Undo
   * deletes; "moved": an entry was moved to another index
   */
  type?: "added" | "moved";
  /**
   * The list of the changed entry; "project" for a setting of the project
   * itself, "app" for a setting of the view in the app store
   */
  kind: ListKind | "project" | "app";
  /** The ID of the entry, or "" for the project and the app */
  id: string;
  name: string;
  /** The property, or "index" for a move, or "" for an added map */
  property: string;
  before: unknown;
  after: unknown;
};

/** What a tool works with, besides the arguments of the model */
type ToolContext = {
  projectStore: ProjectStoreApi;
  dataStore: DataStoreApi;
  /** For call_action, which sets the view */
  appStore: AppStoreApi;
  tables: TableKnowledge;
  /** The changes of the current request, which the tool adds to */
  changes: Change[];
  /**
   * Shows how far a long tool is, e.g. "testing the genes: 120 of 377", with
   * a button that stops it
   */
  progress?: (text: string) => void;
};

/**
 * A tool the model can call
 *
 * It returns a short text for the model, starting with "Done", "Nothing
 * changed" or "Error". If something is wrong, it changes nothing.
 */
type Tool = (
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
) => string | Promise<string>;

/** The sections of the project, in summary order */
const objectKinds: ObjectKind[] = ["images", "labels", "points", "shapes"];

/** The sections whose items can be colored and filtered */
const itemsKinds: ItemsKind[] = ["labels", "points"];

/** The ids parameter that most tools share */
const idsParameter = {
  type: "array",
  items: { type: "string" },
  description:
    "The ids of the objects, exactly as in the project summary. Put several objects in one list.",
};

/** The category parameter, for all the objects of a section of the summary */
const categoryParameter = {
  type: "string",
  enum: objectKinds,
  description:
    'Instead of ids: all the objects of this section of the summary, e.g. "points" for all points.',
};

/** The type parameter, for ids that are in more than one section */
const typeParameter = {
  type: "string",
  enum: objectKinds,
  description:
    "Optional. Only needed when the same id is in more than one section of the summary.",
};

/** The category parameter of the color and filter tools */
const itemsCategoryParameter = {
  type: "string",
  enum: itemsKinds,
  description: "Instead of ids: all the labels or all the points.",
};

/** The column parameter of the tools that work on the groups of a column */
const groupsColumnParameter = {
  type: "string",
  description: "Optional. Without it: the column the object is colored by now.",
};

/** The color names the model may use */
const colorNamesDescription =
  "red, green, blue, yellow, orange, purple, pink, brown, cyan, magenta, white, black, grey, light grey, dark grey";

/** The tools, as described to the model */
const toolDefinitions = [
  {
    type: "function",
    function: {
      name: "set_visibility",
      description:
        "Show or hide objects: the ones in ids, or all the objects of a category.",
      parameters: {
        type: "object",
        required: ["visible"],
        properties: {
          ids: idsParameter,
          category: categoryParameter,
          visible: {
            type: "boolean",
            description: "true to show, false to hide",
          },
          type: typeParameter,
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "show_only",
      description:
        "Show only the given objects (ids or a whole category) and hide all the other images, labels, points and shapes. Without ids and category it hides everything.",
      parameters: {
        type: "object",
        required: [],
        properties: {
          ids: idsParameter,
          category: categoryParameter,
          type: typeParameter,
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "set_opacity",
      description:
        "Set the opacity of objects (ids or a whole category), from 0 (invisible) to 1 (fully opaque).",
      parameters: {
        type: "object",
        required: ["opacity"],
        properties: {
          ids: idsParameter,
          category: categoryParameter,
          opacity: {
            type: "number",
            description:
              'From 0 to 1. "50% transparent" is 0.5, "30% transparent" is 0.7, "80% opaque" is 0.8.',
          },
          type: typeParameter,
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "set_point_size",
      description:
        'Make the points of points objects (ids, or category "points" for all) bigger or smaller than they are now, by a percentage.',
      parameters: {
        type: "object",
        required: ["change_percent"],
        properties: {
          ids: idsParameter,
          category: {
            type: "string",
            enum: ["points"],
            description: 'Use "points" for all the points objects.',
          },
          change_percent: {
            type: "number",
            description:
              'The percentage the user said, positive for bigger, negative for smaller. Do no math: "10% smaller" is -10, "50% bigger" is 50, "twice as big" is 100, "half the size" is -50.',
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "color_by_column",
      description:
        "Color labels or points by a column of their own table: each group gets its own color, numbers get a color scale.",
      parameters: {
        type: "object",
        required: ["column"],
        properties: {
          ids: idsParameter,
          category: itemsCategoryParameter,
          column: {
            type: "string",
            description:
              'A column from "columns you can use" of the object, e.g. "leiden", or a gene name, e.g. "EPCAM".',
          },
          palette: {
            type: "string",
            description:
              'Only if the user names one, e.g. "viridis" or "tableau10".',
          },
          range: {
            type: "array",
            items: { type: "number" },
            description:
              "Only for numbers, and only if the user gives one: [lowest, highest], e.g. [0, 500].",
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "set_group_colors",
      description:
        'Give chosen colors to some groups of labels or points, e.g. "cluster 3 in red, the others grey".',
      parameters: {
        type: "object",
        required: ["colors"],
        properties: {
          ids: idsParameter,
          category: itemsCategoryParameter,
          colors: {
            type: "object",
            additionalProperties: { type: "string" },
            description: `Group → color, e.g. {"3": "red", "T cell": "blue"}. Colors in English (${colorNamesDescription}) or "#rrggbb".`,
          },
          others: {
            type: "string",
            description:
              "The color of all the other groups. Leave it out if the user does not say it: they become light grey.",
          },
          column: groupsColumnParameter,
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "set_color",
      description:
        'Give one color to all the items of labels or points, e.g. "color the cell circles green".',
      parameters: {
        type: "object",
        required: ["color"],
        properties: {
          ids: idsParameter,
          category: itemsCategoryParameter,
          color: {
            type: "string",
            description: `A color in English (${colorNamesDescription}) or "#rrggbb".`,
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "show_only_groups",
      description:
        'Show only some groups of labels or points, e.g. "only clusters 1 and 3", and hide their other items.',
      parameters: {
        type: "object",
        required: ["groups"],
        properties: {
          ids: idsParameter,
          category: itemsCategoryParameter,
          groups: {
            type: "array",
            items: { type: "string" },
            description:
              'The groups to show, e.g. ["1", "3"] or ["EPCAM", "KRT7"].',
          },
          column: groupsColumnParameter,
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "show_all_groups",
      description:
        "Remove the group filter of labels or points: all their items are shown again.",
      parameters: {
        type: "object",
        required: [],
        properties: {
          ids: idsParameter,
          category: itemsCategoryParameter,
        },
      },
    },
  },
];

/** The function that runs each tool, by tool name */
const tools: Record<string, Tool> = {
  set_visibility: setVisibility,
  show_only: showOnly,
  set_opacity: setOpacity,
  set_point_size: setPointSize,
  color_by_column: colorByColumn,
  set_group_colors: setGroupColors,
  set_color: setColor,
  show_only_groups: showOnlyGroups,
  show_all_groups: showAllGroups,
  // only offered to OpenAI and Anthropic, see runTool
  call_action: callAction,
  column_stats: columnStats,
  marker_genes: markerGenes,
  compare_groups: compareGroups,
  correlation,
};

/**
 * Whether two values are the same, also for settings like `{ groupBy: ... }`,
 * which can be equal without being the same object
 *
 * @param a - The first value
 * @param b - The second value
 * @returns Whether their JSON is the same, whatever the order of the keys
 */
function same(a: unknown, b: unknown): boolean {
  return JSON.stringify(sortedKeys(a)) === JSON.stringify(sortedKeys(b));
}

/** A value with the keys of its objects in order, to compare it as JSON */
function sortedKeys(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortedKeys);
  }
  if (isRecord(value)) {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, sortedKeys(value[key])]),
    );
  }
  return value;
}

/**
 * Writes a list of names for a tool result, e.g. "he_image, transcripts"
 *
 * @param names - The names
 * @returns The names separated by commas, or "none"
 */
function list(names: string[]): string {
  return names.join(", ") || "none";
}

/**
 * Rounds a number to 3 decimals, to show it in a readable way
 *
 * @param number - The number
 * @returns The rounded number
 */
function round(number: number): number {
  return Math.round(number * 1000) / 1000;
}

/**
 * Accepts "image" for "images", as small models often write
 *
 * @param text - The section name given by the model
 * @returns The section, or `undefined` if there is none with that name
 */
function toObjectKind(text: unknown): ObjectKind | undefined {
  return objectKinds.find(
    (kind) => kind === text || kind === `${String(text)}s`,
  );
}

/**
 * Finds an object or map of the project by its ID
 *
 * @param project - The project
 * @param kind - The section or map list the ID is in
 * @param id - The ID
 * @returns The object or map, or `undefined` if there is none
 */
function findById(
  project: ProjectStore,
  kind: ListKind,
  id: string,
): { id: string; name: string } | undefined {
  const objects: readonly { id: string; name: string }[] = project[kind];
  return objects.find((object) => object.id === id);
}

/**
 * Finds the section of an ID, which can be in more than one; its prefix (e.g.
 * "shapes/") does not always match the section, so all sections are searched
 *
 * @param project - The project
 * @param id - The ID
 * @param type - The section the model named, if any
 * @returns The section, or an error message for the model
 */
function findKind(
  project: ProjectStore,
  id: string,
  type: unknown,
): { kind: ObjectKind } | { error: string } {
  const kinds = objectKinds.filter((kind) => findById(project, kind, id));
  const [onlyKind] = kinds;
  if (onlyKind === undefined) {
    if (project.tables.some((table) => table.id === id)) {
      return {
        error: `Error: "${id}" is a table. Tables cannot be shown, hidden or changed.`,
      };
    }
    return {
      error: `Error: there is no object with id "${id}". Use an id from the project summary.`,
    };
  }
  const kind = toObjectKind(type);
  if (kind === undefined) {
    if (kinds.length > 1) {
      return {
        error: `Error: the id "${id}" is in more than one section: ${kinds.join(", ")}. Call the tool again with type set to one of them.`,
      };
    }
    return { kind: onlyKind };
  }
  if (!kinds.includes(kind)) {
    return {
      error: `Error: the id "${id}" is not in ${kind}, it is in: ${kinds.join(", ")}.`,
    };
  }
  return { kind };
}

/**
 * Updates an object or map with its project store action
 *
 * @param project - The project
 * @param kind - The section or map list
 * @param id - The ID of the object or map
 * @param updates - The properties to change
 */
function applyUpdate(
  project: ProjectStore,
  kind: ListKind,
  id: string,
  updates: Record<string, unknown>,
): void {
  switch (kind) {
    case "images":
      project.updateImage(id, updates);
      break;
    case "labels":
      project.updateLabels(id, updates);
      break;
    case "points":
      project.updatePoints(id, updates);
      break;
    case "shapes":
      project.updateShapes(id, updates);
      break;
    case "layers":
      project.updateLayer(id, updates);
      break;
    case "markerMaps":
      project.updateMarkerMap(id, updates);
      break;
    case "sizeMaps":
      project.updateSizeMap(id, updates);
      break;
    case "colorMaps":
      project.updateColorMap(id, updates);
      break;
    case "visibilityMaps":
      project.updateVisibilityMap(id, updates);
      break;
    case "opacityMaps":
      project.updateOpacityMap(id, updates);
      break;
  }
}

/** Whether a list is one of the lists of maps */
function isMapKind(kind: Change["kind"]): kind is MapKind {
  return kind.endsWith("Maps");
}

/**
 * Changes one property of an object or map, and remembers the value from
 * before, so that "Undo" can put it back; all tools change the project only
 * through here
 *
 * @param project - The project
 * @param changes - The changes of the current request, which this one is added to
 * @param kind - The section or map list
 * @param id - The ID of the object or map
 * @param property - The property to change
 * @param value - The new value
 * @throws Error if there is no such object or map
 */
function change(
  project: ProjectStore,
  changes: Change[],
  kind: ListKind,
  id: string,
  property: string,
  value: unknown,
): void {
  const object = findById(project, kind, id);
  if (object === undefined) {
    throw new Error(`There is no ${kind} with id "${id}"`);
  }
  changes.push({
    kind,
    id,
    name: object.name,
    property,
    before: (object as Record<string, unknown>)[property],
    after: value,
  });
  applyUpdate(project, kind, id, { [property]: value });
}

/**
 * Shows a hidden object whose opacity, size, colors or filter changed, since
 * the change could not be seen otherwise; "Undo" hides it again
 *
 * @param project - The project
 * @param changes - The changes of the current request
 * @param kind - The section of the object
 * @param object - The object
 * @returns Whether the object was hidden and is now shown
 */
function showIfHidden(
  project: ProjectStore,
  changes: Change[],
  kind: ObjectKind,
  object: ProjectObject,
): boolean {
  if (object.visibility) {
    return false;
  }
  change(project, changes, kind, object.id, "visibility", true);
  return true;
}

/**
 * Finds the objects a tool should change: those in the "ids" argument, or all
 * the objects of the "category" argument; ids win over a category, since they
 * are more precise. If one ID is wrong, nothing is changed.
 *
 * @param project - The project
 * @param args - The arguments of the tool call
 * @param kinds - The sections the tool works on
 * @returns The objects, or an error message for the model
 */
function findTargets(
  project: ProjectStore,
  args: ToolArguments,
  kinds: readonly ObjectKind[] = objectKinds,
): { targets: Target[] } | { error: string } {
  let ids = args["ids"];
  // small models send one id as text, or "id" instead of "ids"
  if (typeof ids === "string") {
    ids = [ids];
  }
  if (ids === undefined && typeof args["id"] === "string") {
    ids = [args["id"]];
  }
  const idList = Array.isArray(ids) ? ids.map(String) : [];
  if (args["category"] !== undefined && idList.length === 0) {
    const category = toObjectKind(args["category"]);
    if (category === undefined || !kinds.includes(category)) {
      return { error: `Error: category must be ${kinds.join(" or ")}.` };
    }
    const objects: readonly ProjectObject[] = project[category];
    if (objects.length === 0) {
      return { error: `Error: there are no ${category} in this project.` };
    }
    // the objects that really are in this section, whatever their IDs look
    // like (e.g. "shapes/cell_circles@global" is in points)
    return {
      targets: objects.map((object) => ({ kind: category, object })),
    };
  }
  if (idList.length === 0) {
    return { error: "Error: give ids (a list of ids) or a category." };
  }
  const targets: Target[] = [];
  for (const id of idList) {
    const found = findKind(
      project,
      id,
      kinds.length === 1 ? kinds[0] : args["type"],
    );
    if ("error" in found) {
      return { error: `${found.error} Nothing was changed.` };
    }
    if (!kinds.includes(found.kind)) {
      return {
        error: `Error: "${id}" is in ${found.kind}, but this tool only works on ${kinds.join(" and ")}. Nothing was changed.`,
      };
    }
    const object = findById(project, found.kind, id) as ProjectObject;
    if (!targets.some((t) => t.kind === found.kind && t.object.id === id)) {
      targets.push({ kind: found.kind, object });
    }
  }
  return { targets };
}

/**
 * Like {@link findTargets}, for the color and filter tools, which work on
 * labels and points only
 *
 * @param project - The project
 * @param args - The arguments of the tool call
 * @returns The objects, or an error message for the model
 */
function findItemsTargets(
  project: ProjectStore,
  args: ToolArguments,
): { targets: ItemsTarget[] } | { error: string } {
  const found = findTargets(project, args, itemsKinds);
  return "error" in found ? found : { targets: found.targets as ItemsTarget[] };
}

/**
 * Runs the set_visibility tool
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @returns The result for the model
 */
function setVisibility(args: ToolArguments, context: ToolContext): string {
  const project = context.projectStore.getState();
  const visible = args["visible"];
  if (typeof visible !== "boolean") {
    return "Error: visible must be true or false.";
  }
  const found = findTargets(project, args);
  if ("error" in found) {
    return found.error;
  }
  const state = visible ? "visible" : "hidden";
  const changed: string[] = [];
  const already: string[] = [];
  for (const { kind, object } of found.targets) {
    if (object.visibility === visible) {
      already.push(object.name);
    } else {
      change(project, context.changes, kind, object.id, "visibility", visible);
      changed.push(object.name);
    }
  }
  if (changed.length === 0) {
    // small models follow the last message best, so it says how to answer
    return (
      `Nothing changed: ${list(already)} already ${state}. ` +
      `Tell the user it was already ${state}; do not say that you ${visible ? "showed" : "hid"} it.`
    );
  }
  let text = `Done: now ${state}: ${list(changed)}.`;
  if (already.length > 0) {
    text += ` Already ${state} before: ${list(already)}.`;
  }
  return text;
}

/**
 * Runs the show_only tool: shows the given objects and hides all the others;
 * without ids and category, it hides everything
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @returns The result for the model
 */
function showOnly(args: ToolArguments, context: ToolContext): string {
  const project = context.projectStore.getState();
  const ids = args["ids"];
  const noIds = ids === undefined || (Array.isArray(ids) && ids.length === 0);
  const hideEverything =
    args["category"] === undefined && noIds && args["id"] === undefined;
  // as "section id", since an ID is only unique within its section
  let keep: string[] = [];
  if (!hideEverything) {
    const found = findTargets(project, args);
    if ("error" in found) {
      return found.error;
    }
    keep = found.targets.map(({ kind, object }) => `${kind} ${object.id}`);
  }
  const shown: string[] = [];
  const hidden: string[] = [];
  const alreadyVisible: string[] = [];
  const alreadyHidden: string[] = [];
  for (const kind of objectKinds) {
    const objects: readonly ProjectObject[] = project[kind];
    for (const object of objects) {
      const visible = keep.includes(`${kind} ${object.id}`);
      if (object.visibility === visible) {
        (visible ? alreadyVisible : alreadyHidden).push(object.name);
      } else {
        change(
          project,
          context.changes,
          kind,
          object.id,
          "visibility",
          visible,
        );
        (visible ? shown : hidden).push(object.name);
      }
    }
  }
  const lists =
    ` Shown now: ${list(shown)}. Hidden now: ${list(hidden)}.` +
    ` Already visible: ${list(alreadyVisible)}. Already hidden: ${list(alreadyHidden)}.`;
  if (shown.length + hidden.length > 0) {
    return `Done.${lists}`;
  }
  return (
    `Nothing changed: everything was already as requested.${lists}` +
    " Tell the user that it was already like this; do not say that you changed anything."
  );
}

/**
 * Runs the set_opacity tool, which changes the object's own opacity (not the
 * one of its layer)
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @returns The result for the model
 */
function setOpacity(args: ToolArguments, context: ToolContext): string {
  const project = context.projectStore.getState();
  const opacity = args["opacity"];
  if (typeof opacity !== "number" || opacity < 0 || opacity > 1) {
    return 'Error: opacity must be a number from 0 to 1 ("50% transparent" is 0.5, "30% transparent" is 0.7).';
  }
  const found = findTargets(project, args);
  if ("error" in found) {
    return found.error;
  }
  const changed: string[] = [];
  const already: string[] = [];
  const shownToo: string[] = [];
  for (const { kind, object } of found.targets) {
    if (object.opacity === opacity) {
      already.push(object.name);
      continue;
    }
    change(project, context.changes, kind, object.id, "opacity", opacity);
    changed.push(`${object.name} (${round(object.opacity)} → ${opacity})`);
    if (showIfHidden(project, context.changes, kind, object)) {
      shownToo.push(object.name);
    }
  }
  if (changed.length === 0) {
    return (
      `Nothing changed: ${list(already)} already had opacity ${opacity}. ` +
      "Tell the user it already had this opacity; do not say that you changed it."
    );
  }
  let text = `Done: opacity is now ${opacity} for ${list(changed)}.`;
  if (already.length > 0) {
    text += ` Already at opacity ${opacity}: ${list(already)}.`;
  }
  if (shownToo.length > 0) {
    text += ` Hidden before, so now also visible: ${list(shownToo)}.`;
  }
  return text;
}

/**
 * Runs the set_point_size tool: makes the points bigger or smaller by a
 * percentage of their current size
 *
 * The model only copies the percentage the user said and the plugin does the
 * math, which small models get wrong. The tool changes the point size factor,
 * which scales every point, so it also works when the size of each point comes
 * from a table column.
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @returns The result for the model
 */
function setPointSize(args: ToolArguments, context: ToolContext): string {
  const project = context.projectStore.getState();
  const percent = args["change_percent"];
  if (typeof percent !== "number" || percent <= -100) {
    return 'Error: change_percent must be a number above -100 ("10% smaller" is -10, "50% bigger" is 50).';
  }
  const found = findTargets(project, args, ["points"]);
  if ("error" in found) {
    return found.error;
  }
  const points = found.targets.map(({ object }) => object as Points);
  if (percent === 0) {
    return (
      `Nothing changed: a change of 0% keeps the size of ${list(points.map((p) => p.name))}. ` +
      "Tell the user the size did not change; do not say that you changed it."
    );
  }
  const changed: string[] = [];
  const shownToo: string[] = [];
  for (const object of points) {
    const newFactor = object.pointSizeFactor * (1 + percent / 100);
    change(
      project,
      context.changes,
      "points",
      object.id,
      "pointSizeFactor",
      newFactor,
    );
    changed.push(
      `${object.name} (point size factor ${round(object.pointSizeFactor)} → ${round(newFactor)})`,
    );
    if (showIfHidden(project, context.changes, "points", object)) {
      shownToo.push(object.name);
    }
  }
  const direction = percent > 0 ? "BIGGER" : "SMALLER";
  let text = `Done: now ${Math.abs(percent)}% ${direction} than before: ${list(changed)}.`;
  if (shownToo.length > 0) {
    text += ` Hidden before, so now also visible: ${list(shownToo)}.`;
  }
  return text;
}

// --- Colors and filters ---

/** The colors the model may name */
const colorNames: Record<string, Color> = {
  red: { r: 255, g: 0, b: 0 },
  green: { r: 0, g: 200, b: 0 },
  blue: { r: 0, g: 90, b: 255 },
  yellow: { r: 255, g: 220, b: 0 },
  orange: { r: 255, g: 140, b: 0 },
  purple: { r: 140, g: 60, b: 200 },
  pink: { r: 255, g: 105, b: 180 },
  brown: { r: 140, g: 80, b: 30 },
  cyan: { r: 0, g: 220, b: 220 },
  magenta: { r: 255, g: 0, b: 255 },
  white: { r: 255, g: 255, b: 255 },
  black: { r: 0, g: 0, b: 0 },
  grey: { r: 128, g: 128, b: 128 },
  "light grey": { r: 211, g: 211, b: 211 },
  "dark grey": { r: 80, g: 80, b: 80 },
};

/** The color of the other groups when the user does not name one */
const othersColor = "light grey";

/** Words a small model may put into the colors for "all the other groups" */
const othersWords = ["others", "other", "the others", "rest", "all others"];

/** The IDs of the palettes for groups, from TissUUmaps */
const groupPalettes = categoricalColorPalettes.map((palette) => palette.id);

/** The IDs of the palettes for numbers, from TissUUmaps */
const numberPalettes = continuousColorPalettes.map((palette) => palette.id);

/** Up to this many different values, all of them are listed (e.g. clusters) */
const maxGroups = 20;

/**
 * More different texts than this are probably names of single items (e.g.
 * cell IDs), which are not useful for colors or filters
 */
const maxCategories = 1000;

/**
 * In tables with more rows, only the columns used now (e.g. for colors) are
 * read: TissUUmaps has loaded those already, the others would take time and
 * memory
 */
const maxRowsToRead = 5_000_000;

/**
 * AnnData groups that do not have one row per item: var/ describes the genes,
 * uns/ holds other data
 */
const notPerRowGroups = ["var/", "varm/", "varp/", "uns/"];

/**
 * Groups with one row per item whose coordinates and neighbor graphs are not
 * useful for colors and filters
 */
const hiddenGroups = ["obsm/", "obsp/"];

/**
 * Columns that are not useful for colors and filters: internal indexes, IDs
 * of single items, and control or deprecated counts (e.g. in Xenium data)
 */
const hiddenColumn =
  /(^|\/)_?_?index(_level_\d+__)?$|(^|[/_])id$|control|deprecated|unassigned/i;

/** The AnnData matrices with one column per gene: X and layers/... */
const geneMatrix = /(^|\/)X$|^layers\//;

/** Names of columns that usually hold clusters */
const clusterColumn = /clust|leiden|louvain|kmeans|k_means|phenograph/i;

/**
 * Turns "Red", "light gray" or "#ff0000" into a color
 *
 * @param text - The color as the model wrote it
 * @returns The color, or `undefined` if it is not known
 */
function toColor(text: unknown): Color | undefined {
  const name = String(text)
    .trim()
    .toLowerCase()
    .replaceAll("gray", "grey")
    .replace(/^(light|dark)-?grey$/, "$1 grey");
  const named = colorNames[name];
  if (named !== undefined) {
    return named;
  }
  const hex = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/.exec(name);
  if (hex?.[1] === undefined || hex[2] === undefined || hex[3] === undefined) {
    return undefined;
  }
  return {
    r: parseInt(hex[1], 16),
    g: parseInt(hex[2], 16),
    b: parseInt(hex[3], 16),
  };
}

/**
 * Names a color for the summary, e.g. "red" or "rgb(10, 20, 30)"
 *
 * @param color - The color
 * @returns Its name
 */
function colorName(color: Color): string {
  return (
    Object.keys(colorNames).find((name) => same(colorNames[name], color)) ??
    `rgb(${color.r}, ${color.g}, ${color.b})`
  );
}

/**
 * The color setting of labels or points
 *
 * @param object - The labels or points
 * @returns The color setting
 */
function colorConfigOf(object: ItemsObject): ColorConfig {
  return "labelColor" in object ? object.labelColor : object.pointColor;
}

/**
 * The visibility setting of labels or points, which holds a group filter
 *
 * @param object - The labels or points
 * @returns The visibility setting
 */
function visibilityConfigOf(object: ItemsObject): VisibilityConfig {
  return "labelVisibility" in object
    ? object.labelVisibility
    : object.pointVisibility;
}

/**
 * The name of the color setting of a section
 *
 * @param kind - The section
 * @returns The property name
 */
function colorSetting(kind: ItemsKind): "labelColor" | "pointColor" {
  return kind === "labels" ? "labelColor" : "pointColor";
}

/**
 * The name of the visibility setting of a section
 *
 * @param kind - The section
 * @returns The property name
 */
function visibilitySetting(
  kind: ItemsKind,
): "labelVisibility" | "pointVisibility" {
  return kind === "labels" ? "labelVisibility" : "pointVisibility";
}

/**
 * The table column a setting takes its values from, if any
 *
 * @param config - The setting
 * @returns The column, or `undefined` for a constant or random setting
 */
function columnRefOf(config: Config<string>): TableColumnRef | undefined {
  const source = getActiveConfigSource(config);
  if (source === "from" && isFromConfig(config)) {
    return config.from;
  }
  if (source === "groupBy" && isGroupByConfig<false>(config)) {
    return config.groupBy;
  }
  return undefined;
}

/**
 * The column whose groups give the colors of labels or points now, if any
 *
 * @param object - The labels or points
 * @returns The column, or `undefined` if the colors do not come from groups
 */
function groupByColumnOf(object: ItemsObject): TableColumnRef | undefined {
  const config = colorConfigOf(object);
  return getActiveConfigSource(config) === "groupBy" &&
    isGroupByConfig<false>(config)
    ? config.groupBy
    : undefined;
}

/**
 * The settings of labels or points that can take their values from a column
 *
 * @param object - The labels or points
 * @returns The settings
 */
function columnSettingsOf(object: ItemsObject): Config<string>[] {
  return "labelColor" in object
    ? [object.labelColor, object.labelVisibility, object.labelOpacity]
    : [
        object.pointColor,
        object.pointVisibility,
        object.pointOpacity,
        object.pointSize,
        object.pointMarker,
      ];
}

/**
 * The table whose rows annotate the items of an object (e.g. one row per
 * cell), if any
 *
 * @param object - The object
 * @returns The table ID, or `undefined` if none is linked
 */
function annotatingTable(object: object): string | undefined {
  if (!("dataSource" in object)) {
    return undefined;
  }
  const dataSource = object.dataSource;
  if (
    typeof dataSource !== "object" ||
    dataSource === null ||
    !("table" in dataSource)
  ) {
    return undefined;
  }
  return typeof dataSource.table === "string" ? dataSource.table : undefined;
}

/**
 * The x and y columns of points loaded from a table, which only tell where
 * each point is drawn
 *
 * @param object - The labels or points
 * @returns The column names, empty for other objects
 */
function positionColumnsOf(object: ItemsObject): string[] {
  const dataSource: object & { type: string } = object.dataSource;
  if (dataSource.type !== "table") {
    return [];
  }
  const x =
    "x" in dataSource && typeof dataSource.x === "string" ? dataSource.x : "x";
  const y =
    "y" in dataSource && typeof dataSource.y === "string" ? dataSource.y : "y";
  return [x, y];
}

/** What was found out about one table */
type TableInfo =
  | { status: "reading" }
  | { status: "error"; error: string }
  | {
      status: "ready";
      rows: number;
      tooBig: boolean;
      useful: ColumnInfo[];
      other: number;
    };

/** A column useful for colors and filters */
type ColumnInfo = {
  name: string;
  /** e.g. "12 groups: 0, 1, ..., 11" or "numbers, 5 to 830" */
  text: string;
  /** All the groups, or 5 examples when there are {@link many} */
  groups?: string[];
  many?: boolean;
  /** Example gene names of a gene matrix, empty when the names are not available */
  genes?: string[];
};

/** What the panel knows about the tables of the open project */
type TableKnowledge = {
  /** What was found out about each table, by table ID */
  info: Map<string, TableInfo>;
  /**
   * The other tables each object's settings have used, by object ID, so that
   * after e.g. coloring the cell circles by their own table, the columns of
   * the AnnData table can still be used
   */
  remembered: Map<string, Set<string>>;
};

/**
 * The tables whose columns labels or points can use: their own table, and the
 * tables their settings use or have used (e.g. cell circles colored by a
 * column of the AnnData table), whose rows TissUUmaps matches to the items by
 * their IDs
 *
 * @param object - The labels or points
 * @param tables - What the panel knows about the tables, updated here
 * @returns The table IDs, the object's own table first
 */
function objectTables(object: ItemsObject, tables: TableKnowledge): string[] {
  const remembered = tables.remembered.get(object.id) ?? new Set<string>();
  for (const config of columnSettingsOf(object)) {
    const table = columnRefOf(config)?.table;
    if (table !== undefined) {
      remembered.add(table);
    }
  }
  tables.remembered.set(object.id, remembered);
  const own = annotatingTable(object);
  return [...new Set([...(own === undefined ? [] : [own]), ...remembered])];
}

/**
 * The tables of all labels and points
 *
 * @param project - The project
 * @param tables - What the panel knows about the tables
 * @returns The table IDs, each once
 */
function linkedTables(project: ProjectStore, tables: TableKnowledge): string[] {
  const objects: ItemsObject[] = [...project.labels, ...project.points];
  return [
    ...new Set(objects.flatMap((object) => objectTables(object, tables))),
  ];
}

/**
 * The columns of a table that labels and points use now, e.g. for colors;
 * TissUUmaps keeps them in memory, so reading them again costs little, even in
 * very big tables
 *
 * @param project - The project
 * @param tableId - The table
 * @returns The column names
 */
function findUsedColumns(project: ProjectStore, tableId: string): Set<string> {
  const used = new Set<string>();
  const objects: ItemsObject[] = [...project.labels, ...project.points];
  for (const object of objects) {
    for (const config of columnSettingsOf(object)) {
      const ref = columnRefOf(config);
      if (
        ref !== undefined &&
        (ref.table ?? annotatingTable(object)) === tableId
      ) {
        used.add(ref.column);
      }
    }
  }
  return used;
}

/**
 * Lists the columns of a loaded table, also one level into groups like
 * "obs/"; a matrix (e.g. X, one column per gene) is a group without "/"
 *
 * @param data - The table data
 * @param options - Optional abort signal
 * @returns The columns
 */
async function listColumns(
  data: TableData,
  options?: { signal?: AbortSignal },
): Promise<{ name: string; matrix: boolean; hidden: boolean }[]> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const columns: { name: string; matrix: boolean; hidden: boolean }[] = [];
  for (const suggestion of await data.suggestColumnQueries("", { signal })) {
    if (notPerRowGroups.includes(suggestion.query)) {
      continue;
    }
    const isGroup = suggestion.group === true && suggestion.query.endsWith("/");
    const inner = isGroup
      ? await data.suggestColumnQueries(suggestion.query, { signal })
      : [suggestion];
    const hiddenGroup = hiddenGroups.includes(suggestion.query);
    for (const { query, group } of inner) {
      // a group within a group is not looked into
      if (group === true && query.endsWith("/")) {
        continue;
      }
      columns.push({
        name: query,
        matrix: group === true,
        hidden: hiddenGroup || hiddenColumn.test(query),
      });
    }
  }
  return columns;
}

/**
 * Lists the columns of a matrix, e.g. X[EPCAM], X[KRT7], ...
 *
 * @param data - The table data
 * @param name - The matrix name, e.g. "X"
 * @param options - Optional abort signal
 * @returns The column names
 */
async function matrixColumns(
  data: TableData,
  name: string,
  options?: { signal?: AbortSignal },
): Promise<string[]> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  return (await data.suggestColumnQueries(name, { signal }))
    .map((suggestion) => suggestion.query)
    .filter((query) => query.startsWith(`${name}[`));
}

/**
 * Whether the columns of a matrix are named after genes, rather than only
 * numbered (X[0], X[1], ...)
 *
 * @param columns - The column names
 * @param name - The matrix name
 * @returns Whether there are gene names
 */
function hasGeneNames(columns: string[], name: string): boolean {
  return columns.some(
    (column) => !/^\d+$/.test(column.slice(name.length + 1, -1)),
  );
}

/**
 * Finds out why TissUUmaps could not name the genes of a matrix, if possible
 *
 * TissUUmaps names the columns of X after the index of var/. A plugin cannot
 * read var/ itself, since `loadValues` only gives columns with one value per
 * row (per cell) and var/ has one per gene, but the error tells why: a row
 * count error means that the values could be read.
 *
 * @param data - The table data
 * @param options - Optional abort signal
 * @returns The reason in brackets, e.g. " (var/_index: Failed to fetch)", or ""
 */
async function varIndexProblem(
  data: TableData,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  let index: string | undefined = "var/";
  try {
    const varColumns = (await data.suggestColumnQueries("var/", { signal }))
      .filter((suggestion) => suggestion.group !== true)
      .map((suggestion) => suggestion.query);
    index =
      varColumns.find((column) => column.endsWith("/_index")) ?? varColumns[0];
    if (index === undefined) {
      return " (var/ has no columns)";
    }
    await data.loadValues(index, { signal });
    return "";
  } catch (error) {
    signal?.throwIfAborted();
    const message = error instanceof Error ? error.message : String(error);
    return /has \d+ rows, but the table has/.test(message)
      ? ""
      : ` (${index}: ${message})`;
  }
}

/**
 * Describes a matrix of genes, such as X of AnnData
 *
 * @param data - The table data
 * @param name - The matrix name
 * @param options - Optional abort signal
 * @returns The description, e.g. "377 genes, e.g. X[EPCAM], X[KRT7], X[CD3E]",
 * with example gene names, or `undefined` for other matrices
 */
async function describeMatrix(
  data: TableData,
  name: string,
  options?: { signal?: AbortSignal },
): Promise<Omit<ColumnInfo, "name"> | undefined> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  if (!geneMatrix.test(name)) {
    return undefined;
  }
  const columns = await matrixColumns(data, name, { signal });
  if (columns.length === 0) {
    return undefined;
  }
  if (!hasGeneNames(columns, name)) {
    const problem = await varIndexProblem(data, { signal });
    return {
      text:
        `${columns.length} genes, gene names not available${problem}: ` +
        `only by number, ${name}[0] to ${name}[${columns.length - 1}]`,
      genes: [],
    };
  }
  const examples = columns.slice(0, 3);
  return {
    text: `${columns.length} genes, e.g. ${examples.join(", ")}`,
    genes: examples.map((column) => column.slice(name.length + 1, -1)),
  };
}

/**
 * Sorts groups like people expect: "2" before "10", "B cell" before "T cell"
 *
 * @param groups - The groups
 * @returns The sorted groups
 */
function sortGroups(groups: Iterable<string>): string[] {
  return [...groups].sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true }),
  );
}

/**
 * Describes the values of a column; it stops counting early, so it stays
 * quick even with millions of rows
 *
 * @param values - The values, numbers as typed arrays
 * @returns The description, e.g. "12 groups: 0, 1, ..., 11" or "numbers, 5 to
 * 830", or `undefined` if the column is not useful: one value for all rows, or
 * a different text in almost every row (e.g. cell IDs)
 */
function describeValues(
  values: TypedArrayOrArray<unknown>,
): Omit<ColumnInfo, "name"> | undefined {
  const numbers = ArrayBuffer.isView(values);
  const seen = new Set<string>();
  for (const value of values) {
    seen.add(String(value));
    if (seen.size > (numbers ? maxGroups : maxCategories)) {
      break;
    }
  }
  if (seen.size <= 1) {
    return undefined;
  }
  const groups = sortGroups(seen);
  if (groups.length <= maxGroups) {
    return { text: `${groups.length} groups: ${groups.join(", ")}`, groups };
  }
  if (numbers) {
    let min = Infinity;
    let max = -Infinity;
    for (const value of values) {
      // Number() also converts 64-bit integers
      const number = Number(value);
      min = Math.min(min, number);
      max = Math.max(max, number);
    }
    return {
      text: min <= max ? `numbers, ${round(min)} to ${round(max)}` : "numbers",
    };
  }
  if (groups.length > maxCategories) {
    return undefined;
  }
  const examples = groups.slice(0, 5);
  return {
    text: `${groups.length} groups, e.g. ${examples.join(", ")}`,
    groups: examples,
    many: true,
  };
}

/** A column of a table, found for a tool */
type FoundColumn = {
  tableId: string;
  data: TableData;
  column: string;
  /** The column as TissUUmaps names it in a setting, with the table only if it is another one */
  ref: TableColumnRef;
};

/**
 * Finds a column by the name the model gave: exactly, ignoring upper and lower
 * case, without a group like "obs/", or as a gene of X ("EPCAM" is X[EPCAM])
 *
 * It looks in the object's own table first, then in the tables its settings
 * use or have used. Without a name, it is the column the object is colored by
 * now.
 *
 * @param dataStore - The data store
 * @param tables - What the panel knows about the tables
 * @param object - The labels or points
 * @param wanted - The column name the model gave, if any
 * @param options - Optional abort signal
 * @returns The column, or an error message for the model
 */
async function findColumn(
  dataStore: DataStoreApi,
  tables: TableKnowledge,
  object: ItemsObject,
  wanted: unknown,
  options?: { signal?: AbortSignal },
): Promise<FoundColumn | { error: string }> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const own = annotatingTable(object);
  let tableIds = objectTables(object, tables);
  let name = typeof wanted === "string" ? wanted : "";
  if (name === "") {
    const groupBy = groupByColumnOf(object);
    if (groupBy === undefined) {
      // e.g. "clusters" with several cluster columns: the model must not guess
      const clusters: string[] = [];
      for (const tableId of tableIds) {
        const data = await dataStore.getState().tableDataRefs.get(tableId)
          ?.promise;
        for (const column of data === undefined
          ? []
          : await listColumns(data, { signal })) {
          if (
            !column.matrix &&
            !column.hidden &&
            clusterColumn.test(column.name)
          ) {
            clusters.push(column.name);
          }
        }
      }
      let text = `Error: ${object.name} is not colored by groups now, so give column.`;
      if (clusters.length > 1) {
        text += ` Do not guess: ask the user which column they mean: ${list(clusters)}.`;
      }
      return { error: text };
    }
    name = groupBy.column;
    const table = groupBy.table ?? own;
    tableIds = table === undefined ? [] : [table];
  }
  if (tableIds.length === 0) {
    return {
      error: `Error: ${object.name} has no table, so it has no columns.`,
    };
  }
  const lower = name.toLowerCase();
  let loaded = false;
  const unnamed: string[] = [];
  for (const tableId of tableIds) {
    const tableRef = dataStore.getState().tableDataRefs.get(tableId);
    if (tableRef === undefined) {
      continue;
    }
    loaded = true;
    const data = await tableRef.promise;
    const names: string[] = [];
    for (const column of await listColumns(data, { signal })) {
      if (!column.matrix) {
        names.push(column.name);
        continue;
      }
      if (!geneMatrix.test(column.name)) {
        continue;
      }
      const columns = await matrixColumns(data, column.name, { signal });
      const matrix = column.name;
      // "X" alone is the whole matrix: explain how to name one of its columns
      if (matrix.toLowerCase() === lower) {
        const how = hasGeneNames(columns, matrix)
          ? `one gene, e.g. ${columns[0]}`
          : `a gene by number, ${matrix}[0] to ${matrix}[${columns.length - 1}]: the gene names are not available here`;
        return {
          error: `Error: "${matrix}" has one column per gene: give ${matrix}[gene name], so ${how}.`,
        };
      }
      if (!hasGeneNames(columns, matrix)) {
        unnamed.push(`${matrix}[0] to ${matrix}[${columns.length - 1}]`);
      }
      names.push(...columns);
    }
    const column =
      names.find((n) => n.toLowerCase() === lower) ??
      names.find((n) => n.toLowerCase().endsWith(`/${lower}`)) ??
      names.find((n) => n.toLowerCase() === `x[${lower}]`);
    if (column !== undefined) {
      const ref = tableId === own ? { column } : { table: tableId, column };
      return { tableId, data, column, ref };
    }
  }
  if (!loaded) {
    return {
      error: `Error: the table of ${object.name} is not loaded yet. Try again in a moment.`,
    };
  }
  if (unnamed.length > 0) {
    return {
      error:
        `Error: there is no column "${name}", and the gene names are not available in this project ` +
        `(genes only by number, ${list(unnamed)}), so genes cannot be found by name. Tell the user this.`,
    };
  }
  return {
    error: `Error: ${object.name} has no column "${name}". Use one from "columns you can use" in the summary.`,
  };
}

/**
 * Reads the values of a column, as groups
 *
 * @param data - The table data
 * @param column - The column name
 * @param options - Optional abort signal
 * @returns Whether the column has numbers, and its different values as text,
 * or `null` when there are too many different numbers to be groups
 */
async function readColumn(
  data: TableData,
  column: string,
  options?: { signal?: AbortSignal },
): Promise<{ numbers: boolean; groups: string[] | null }> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const values = await data.loadValues(column, { signal });
  const numbers = ArrayBuffer.isView(values);
  const seen = new Set<string>();
  for (const value of values) {
    seen.add(String(value));
    if (numbers && seen.size > maxGroups) {
      return { numbers, groups: null };
    }
  }
  return { numbers, groups: sortGroups(seen) };
}

/**
 * Finds a group as it is written in the table: "epcam" is "EPCAM", and
 * "cluster 3" is "3"
 *
 * @param groups - The groups of the column
 * @param wanted - The group the model gave
 * @returns The group, or `undefined` if there is none
 */
function findGroup(groups: string[], wanted: unknown): string | undefined {
  const lower = String(wanted).trim().toLowerCase();
  const lastWord = lower.split(/\s+/).at(-1);
  return (
    groups.find((group) => group.toLowerCase() === lower) ??
    groups.find((group) => group.toLowerCase() === lastWord)
  );
}

/**
 * Finds a palette by name, among those for groups or those for numbers
 *
 * @param name - The palette name the model gave
 * @param forGroups - Whether a palette for groups is wanted
 * @returns The palette ID, or `undefined` if there is none
 */
function findPalette(name: unknown, forGroups: boolean): string | undefined {
  const lower = String(name).toLowerCase();
  // the batlow made for groups
  if (forGroups && lower === "batlow") {
    return "batlowS";
  }
  return (forGroups ? groupPalettes : numberPalettes).find(
    (palette) => palette.toLowerCase() === lower,
  );
}

/**
 * Shows a few groups, e.g. for an error: "0, 1, 2, ... (16 groups)"
 *
 * @param groups - The groups
 * @returns The text
 */
function someGroups(groups: string[]): string {
  return groups.length <= maxGroups
    ? groups.join(", ")
    : `${groups.slice(0, 10).join(", ")}, ... (${groups.length} groups)`;
}

/**
 * Sets the values of the color map or visibility map that the chat uses for
 * one object, and adds the map to the project the first time
 *
 * There is one map per object, reused every time, so that the project does
 * not fill up with maps. Changes to an existing map go through
 * {@link change}, so that "Undo" puts them back; a new map is remembered as
 * added, so that "Undo" deletes it.
 *
 * @typeParam TValue - The type of the map values
 * @param project - The project
 * @param changes - The changes of the current request
 * @param mapKind - The map list
 * @param map - The map with its new values
 */
function setChatMap<TValue extends Color | boolean>(
  project: ProjectStore,
  changes: Change[],
  mapKind: MapKind,
  map: GroupValueMap<TValue>,
): void {
  if (findById(project, mapKind, map.id) === undefined) {
    if (mapKind === "colorMaps") {
      project.addColorMap(map as GroupValueMap<Color>);
    } else {
      project.addVisibilityMap(map as GroupValueMap<boolean>);
    }
    // so that Undo deletes it again
    changes.push({
      type: "added",
      kind: mapKind,
      id: map.id,
      name: map.name,
      property: "",
      before: undefined,
      after: map,
    });
    return;
  }
  change(project, changes, mapKind, map.id, "values", map.values);
  change(project, changes, mapKind, map.id, "default", map.default);
}

/**
 * Makes a plan for each object; objects where it does not work are skipped,
 * so that e.g. category "labels" still colors the labels that have the column
 *
 * @typeParam TPlan - The type of a plan
 * @param targets - The objects
 * @param makePlan - Makes the plan for one object, or gives an error message
 * @param options - Optional abort signal
 * @returns The plans and the skipped objects, or an error message if it works
 * for none of them
 */
async function planEach<TPlan>(
  targets: ItemsTarget[],
  makePlan: (target: ItemsTarget) => Promise<TPlan | { error: string }>,
  options?: { signal?: AbortSignal },
): Promise<
  | { plans: TPlan[]; skipped: { name: string; error: string }[] }
  | { error: string }
> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const plans: TPlan[] = [];
  const skipped: { name: string; error: string }[] = [];
  for (const target of targets) {
    const plan = await makePlan(target);
    if (typeof plan === "object" && plan !== null && "error" in plan) {
      skipped.push({ name: target.object.name, error: plan.error });
    } else {
      plans.push(plan);
    }
  }
  if (plans.length === 0) {
    return {
      error: `${skipped.map((s) => s.error).join(" ")} Nothing was changed.`,
    };
  }
  return { plans, skipped };
}

/**
 * The end of a tool result about the skipped objects
 *
 * @param skipped - The skipped objects, with the reason
 * @returns The text, e.g. ' Not changed: nucleus_labels (it has no column "leiden").', or ""
 */
function skippedText(skipped: { name: string; error: string }[]): string {
  if (skipped.length === 0) {
    return "";
  }
  const reasons = skipped.map(
    (s) => `${s.name} (${s.error.replace(/^Error: /, "")})`,
  );
  return ` Not changed: ${reasons.join("; ")} Tell the user which objects were not changed and why.`;
}

/**
 * Runs the color_by_column tool: groups get a palette for groups, numbers a
 * color scale; without a palette or range, TissUUmaps' usual choice
 * ("distinct" or "viridis", and a scale from the lowest to the highest value)
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @param options - Optional abort signal
 * @returns The result for the model
 */
async function colorByColumn(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const project = context.projectStore.getState();
  const palette = args["palette"];
  const range = args["range"];
  const validRange =
    Array.isArray(range) &&
    range.length === 2 &&
    range.every((value) => Number.isFinite(value)) &&
    Number(range[0]) < Number(range[1])
      ? ([Number(range[0]), Number(range[1])] as [number, number])
      : undefined;
  if (range !== undefined && validRange === undefined) {
    return "Error: range must be [lowest, highest], e.g. [0, 500].";
  }
  const found = findItemsTargets(project, args);
  if ("error" in found) {
    return found.error;
  }
  const planned = await planEach(
    found.targets,
    async ({ kind, object }) => {
      const target = await findColumn(
        context.dataStore,
        context.tables,
        object,
        args["column"],
        { signal },
      );
      if ("error" in target) {
        return target;
      }
      const { numbers, groups } = await readColumn(target.data, target.column, {
        signal,
      });
      // a palette for numbers or a range also make few different numbers a scale
      const scale =
        groups === null ||
        (numbers &&
          (validRange !== undefined ||
            findPalette(palette, false) !== undefined));
      if (validRange !== undefined && !scale) {
        return {
          error: `Error: "${target.column}" has groups, not numbers, so it takes no range.`,
        };
      }
      const chosen =
        palette === undefined
          ? scale
            ? "viridis"
            : "distinct"
          : findPalette(palette, !scale);
      if (chosen === undefined) {
        const palettes = scale ? numberPalettes : groupPalettes;
        return {
          error: `Error: "${String(palette)}" is not a palette for ${scale ? "numbers" : "groups"}. Use one of: ${palettes.join(", ")}.`,
        };
      }
      const config: ColorConfig = scale
        ? {
            from: {
              ...target.ref,
              palette: chosen,
              ...(validRange && { range: validRange }),
            },
          }
        : { groupBy: { ...target.ref, map: undefined, palette: chosen } };
      const how = scale
        ? `the numbers in ${target.column} (palette ${chosen}, ${validRange ? `range ${validRange[0]} to ${validRange[1]}` : "from the lowest to the highest value"})`
        : `the ${groups?.length ?? 0} groups of ${target.column} (palette ${chosen})`;
      return { kind, object, config, how };
    },
    { signal },
  );
  if ("error" in planned) {
    return planned.error;
  }
  const done: string[] = [];
  const already: string[] = [];
  const shownToo: string[] = [];
  for (const { kind, object, config, how } of planned.plans) {
    if (same(colorConfigOf(object), config)) {
      already.push(`${object.name} by ${how}`);
      continue;
    }
    change(
      project,
      context.changes,
      kind,
      object.id,
      colorSetting(kind),
      config,
    );
    done.push(`${object.name} by ${how}`);
    if (showIfHidden(project, context.changes, kind, object)) {
      shownToo.push(object.name);
    }
  }
  if (done.length === 0) {
    return (
      `Nothing changed: already colored like this: ${list(already)}. Tell the user it was already like this; do not say that you changed it.` +
      skippedText(planned.skipped)
    );
  }
  let text = `Done: now colored: ${list(done)}.`;
  if (already.length > 0) {
    text += ` Already colored like this before: ${list(already)}.`;
  }
  if (shownToo.length > 0) {
    text += ` Hidden before, so now also visible: ${list(shownToo)}.`;
  }
  return text + skippedText(planned.skipped);
}

/**
 * Runs the set_group_colors tool: gives chosen colors to some groups, and one
 * color to all the others (light grey if the user does not say)
 *
 * The colors go into the chat's color map of the object. Colors chosen
 * earlier for the same column are kept, so that "and cluster 5 in blue" adds
 * to "cluster 3 in red".
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @param options - Optional abort signal
 * @returns The result for the model
 */
async function setGroupColors(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const project = context.projectStore.getState();
  const argsColors = args["colors"];
  const colors: Record<string, unknown> =
    typeof argsColors === "object" &&
    argsColors !== null &&
    !Array.isArray(argsColors)
      ? { ...argsColors }
      : {};
  let othersText = args["others"];
  // {"3": "red", "others": "grey"} means others: "grey"
  for (const word of othersWords) {
    if (word in colors) {
      othersText ??= colors[word];
      delete colors[word];
    }
  }
  if (Object.keys(colors).length === 0) {
    return 'Error: colors must give a color to each group, e.g. {"3": "red"}. To give one color to all the items, use set_color.';
  }
  const unknown = [...Object.values(colors), othersText ?? othersColor].filter(
    (color) => toColor(color) === undefined,
  );
  if (unknown.length > 0) {
    return `Error: unknown color "${String(unknown[0])}". Use one of: ${Object.keys(colorNames).join(", ")}, or "#rrggbb".`;
  }
  const others = toColor(othersText ?? othersColor) as Color;
  const found = findItemsTargets(project, args);
  if ("error" in found) {
    return found.error;
  }
  const planned = await planEach(
    found.targets,
    async ({ kind, object }) => {
      const target = await findColumn(
        context.dataStore,
        context.tables,
        object,
        args["column"],
        { signal },
      );
      if ("error" in target) {
        return target;
      }
      const { groups } = await readColumn(target.data, target.column, {
        signal,
      });
      if (groups === null) {
        return {
          error: `Error: "${target.column}" has numbers, not groups. Use color_by_column for numbers.`,
        };
      }
      const mapId = `chat-colors/${kind}/${object.id}`;
      const map: GroupValueMap<Color> = {
        id: mapId,
        name: `Chat colors – ${object.name}`,
        values: {},
        default: others,
      };
      // the colors chosen earlier for the same column stay, unless the user
      // names new "others"
      const earlier = project.colorMaps.find((m) => m.id === mapId);
      const ref = { ...target.ref, map: mapId };
      if (earlier !== undefined && same(groupByColumnOf(object), ref)) {
        map.values = { ...earlier.values };
        if (othersText === undefined) {
          map.default = earlier.default;
        }
      }
      const named: string[] = [];
      for (const [wanted, colorText] of Object.entries(colors)) {
        const group = findGroup(groups, wanted);
        const color = toColor(colorText) as Color;
        if (group === undefined) {
          return {
            error: `Error: ${target.column} has no group "${wanted}". Its groups: ${someGroups(groups)}.`,
          };
        }
        map.values[group] = color;
        named.push(`${group} ${colorName(color)}`);
      }
      const config: ColorConfig = { groupBy: ref };
      const text = `${object.name} by ${target.column}: ${named.join(", ")}, all other groups ${colorName(map.default ?? others)}`;
      return { kind, object, map, config, text };
    },
    { signal },
  );
  if ("error" in planned) {
    return planned.error;
  }
  const shownToo: string[] = [];
  for (const { kind, object, map, config } of planned.plans) {
    setChatMap(project, context.changes, "colorMaps", map);
    if (!same(colorConfigOf(object), config)) {
      change(
        project,
        context.changes,
        kind,
        object.id,
        colorSetting(kind),
        config,
      );
    }
    if (showIfHidden(project, context.changes, kind, object)) {
      shownToo.push(object.name);
    }
  }
  let text = `Done: now colored ${list(planned.plans.map((plan) => plan.text))}.`;
  if (shownToo.length > 0) {
    text += ` Hidden before, so now also visible: ${list(shownToo)}.`;
  }
  return text + skippedText(planned.skipped);
}

/**
 * Runs the set_color tool: gives one color to all the items of labels or points
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @returns The result for the model
 */
function setColor(args: ToolArguments, context: ToolContext): string {
  const project = context.projectStore.getState();
  const color = toColor(args["color"] ?? "");
  if (color === undefined) {
    return `Error: unknown color "${String(args["color"])}". Use one of: ${Object.keys(colorNames).join(", ")}, or "#rrggbb".`;
  }
  const found = findItemsTargets(project, args);
  if ("error" in found) {
    return found.error;
  }
  const config: ColorConfig = { constant: { value: color } };
  const done: string[] = [];
  const already: string[] = [];
  const shownToo: string[] = [];
  for (const { kind, object } of found.targets) {
    if (same(colorConfigOf(object), config)) {
      already.push(object.name);
      continue;
    }
    change(
      project,
      context.changes,
      kind,
      object.id,
      colorSetting(kind),
      config,
    );
    done.push(object.name);
    if (showIfHidden(project, context.changes, kind, object)) {
      shownToo.push(object.name);
    }
  }
  const name = colorName(color);
  if (done.length === 0) {
    return `Nothing changed: ${list(already)} already all ${name}. Tell the user it was already ${name}; do not say that you changed it.`;
  }
  let text = `Done: now all ${name}: ${list(done)}.`;
  if (already.length > 0) {
    text += ` Already ${name} before: ${list(already)}.`;
  }
  if (shownToo.length > 0) {
    text += ` Hidden before, so now also visible: ${list(shownToo)}.`;
  }
  return text;
}

/**
 * Runs the show_only_groups tool: shows only some groups of labels or points
 * and hides their other items, with the chat's visibility map of the object
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @param options - Optional abort signal
 * @returns The result for the model
 */
async function showOnlyGroups(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const project = context.projectStore.getState();
  const argsGroups = args["groups"];
  const wantedGroups =
    typeof argsGroups === "string" ? [argsGroups] : argsGroups;
  if (!Array.isArray(wantedGroups) || wantedGroups.length === 0) {
    return 'Error: groups must list the groups to show, e.g. ["1", "3"].';
  }
  const found = findItemsTargets(project, args);
  if ("error" in found) {
    return found.error;
  }
  const planned = await planEach(
    found.targets,
    async ({ kind, object }) => {
      const target = await findColumn(
        context.dataStore,
        context.tables,
        object,
        args["column"],
        { signal },
      );
      if ("error" in target) {
        return target;
      }
      const { groups } = await readColumn(target.data, target.column, {
        signal,
      });
      if (groups === null) {
        return {
          error: `Error: "${target.column}" has numbers, not groups: filters on numbers (e.g. "area > 50") are not possible.`,
        };
      }
      const map: GroupValueMap<boolean> = {
        id: `chat-filter/${kind}/${object.id}`,
        name: `Chat filter – ${object.name}`,
        values: {},
        default: false,
      };
      for (const wanted of wantedGroups) {
        const group = findGroup(groups, wanted);
        if (group === undefined) {
          return {
            error: `Error: ${target.column} has no group "${String(wanted)}". Its groups: ${someGroups(groups)}.`,
          };
        }
        map.values[group] = true;
      }
      const config: VisibilityConfig = {
        groupBy: { ...target.ref, map: map.id },
      };
      const text = `${object.name}: only ${Object.keys(map.values).join(", ")} of ${target.column}`;
      return { kind, object, map, config, text };
    },
    { signal },
  );
  if ("error" in planned) {
    return planned.error;
  }
  const shownToo: string[] = [];
  for (const { kind, object, map, config } of planned.plans) {
    setChatMap(project, context.changes, "visibilityMaps", map);
    if (!same(visibilityConfigOf(object), config)) {
      change(
        project,
        context.changes,
        kind,
        object.id,
        visibilitySetting(kind),
        config,
      );
    }
    if (showIfHidden(project, context.changes, kind, object)) {
      shownToo.push(object.name);
    }
  }
  let text = `Done: now shown ${list(planned.plans.map((plan) => plan.text))}; their other items are hidden.`;
  if (shownToo.length > 0) {
    text += ` Hidden before, so now also visible: ${list(shownToo)}.`;
  }
  return text + skippedText(planned.skipped);
}

/**
 * Runs the show_all_groups tool: removes the group filter; objects without a
 * filter are left as they are, so that e.g. category "labels" does not show
 * other labels
 *
 * @param args - The arguments of the tool call
 * @param context - The stores and the changes of the current request
 * @returns The result for the model
 */
function showAllGroups(args: ToolArguments, context: ToolContext): string {
  const project = context.projectStore.getState();
  const found = findItemsTargets(project, args);
  if ("error" in found) {
    return found.error;
  }
  const all: VisibilityConfig = { constant: { value: true } };
  const done: string[] = [];
  const without: string[] = [];
  const shownToo: string[] = [];
  for (const { kind, object } of found.targets) {
    if (same(visibilityConfigOf(object), all)) {
      without.push(object.name);
      continue;
    }
    change(
      project,
      context.changes,
      kind,
      object.id,
      visibilitySetting(kind),
      all,
    );
    done.push(object.name);
    if (showIfHidden(project, context.changes, kind, object)) {
      shownToo.push(object.name);
    }
  }
  if (done.length === 0) {
    return `Nothing changed: ${list(without)} had no group filter, so all their items were already shown. Tell the user; do not say that you changed it.`;
  }
  let text = `Done: filter removed, all items shown again: ${list(done)}.`;
  if (shownToo.length > 0) {
    text += ` Hidden before, so now also visible: ${list(shownToo)}.`;
  }
  if (without.length > 0) {
    text += ` No filter, left as they were: ${list(without)}.`;
  }
  return text;
}

/**
 * Other labels or points that show the same items in the same place: the same
 * file (e.g. the same labels added twice, like "cells" and "cells-epcam"), or
 * items annotated by the same table
 *
 * @param project - The project
 * @param kind - The section of the object
 * @param object - The labels or points
 * @returns The other objects
 */
function sameItemsAs(
  project: ProjectStore,
  kind: ItemsKind,
  object: ItemsObject,
): ItemsObject[] {
  const objects: readonly ItemsObject[] = project[kind];
  return objects.filter(
    (other) => other.id !== object.id && showSameItems(object, other),
  );
}

/**
 * Whether two labels or points show the same items in the same place
 *
 * @param a - The first object
 * @param b - The second object
 * @returns Whether they show the same items
 */
function showSameItems(a: ItemsObject, b: ItemsObject): boolean {
  if (!same(a.layer, b.layer)) {
    return false;
  }
  const { table: tableA, ...fileA } = a.dataSource as {
    table?: string;
    source?: string;
  };
  const { table: tableB, ...fileB } = b.dataSource as {
    table?: string;
    source?: string;
  };
  if (fileA.source !== undefined && same(fileA, fileB)) {
    return true;
  }
  return tableA !== undefined && tableA === tableB;
}

/**
 * Warns when another visible object shows the same items as an object whose
 * colors or filter changed, since it can cover the change (e.g. a filter on
 * "cells" while "cells-epcam" is visible)
 *
 * @param project - The project
 * @param someChanges - The changes of one tool call
 * @returns The warning for the user, or "" when there is nothing to warn about
 */
function overlapWarning(project: ProjectStore, someChanges: Change[]): string {
  const coverable = [
    "labelColor",
    "pointColor",
    "labelVisibility",
    "pointVisibility",
  ];
  const changed = someChanges.filter((c) => coverable.includes(c.property));
  const changedKeys = changed.map((c) => `${c.kind} ${c.id}`);
  const warnings = new Set<string>();
  for (const { kind, id } of changed) {
    if (kind !== "labels" && kind !== "points") {
      continue;
    }
    const object = findById(project, kind, id) as ItemsObject | undefined;
    if (object === undefined) {
      continue;
    }
    for (const other of sameItemsAs(project, kind, object)) {
      if (other.visibility && !changedKeys.includes(`${kind} ${other.id}`)) {
        warnings.add(
          `${other.name} is also visible and shows the same items as ${object.name}, in the same place: ` +
            `it can cover this change. Hide ${other.name} to see it.`,
        );
      }
    }
  }
  return [...warnings].join(" ");
}

// --- Actions of the app state, for OpenAI and Anthropic ---
// call_action lets the strong models call the actions of the stores, which
// they know from the types (store-types.generated.ts). Only actions that can
// be taken back are allowed, and every argument is checked first: if anything
// is wrong, nothing changes and the model is told why.

/** The actions that update an entry of a list, with the entry's type */
const updateActions: Record<
  string,
  { kind: ObjectKind | "layers"; type: string }
> = {
  updateLayer: { kind: "layers", type: "Layer" },
  updateImage: { kind: "images", type: "Image" },
  updateLabels: { kind: "labels", type: "Labels" },
  updatePoints: { kind: "points", type: "Points" },
  updateShapes: { kind: "shapes", type: "Shapes" },
};

/** The kinds of values that settings and maps hold */
type ValueKind = "color" | "boolean" | "opacity" | "size" | "marker";

/** The lists of maps, with their name in the actions and their values */
const mapLists: Record<MapKind, { name: string; value: ValueKind }> = {
  markerMaps: { name: "MarkerMap", value: "marker" },
  sizeMaps: { name: "SizeMap", value: "size" },
  colorMaps: { name: "ColorMap", value: "color" },
  visibilityMaps: { name: "VisibilityMap", value: "boolean" },
  opacityMaps: { name: "OpacityMap", value: "opacity" },
};

/** The actions that move an entry to another index, by list */
const moveActions: Record<string, ObjectKind | "layers"> = {
  moveLayer: "layers",
  moveImage: "images",
  moveLabels: "labels",
  movePoints: "points",
  moveShapes: "shapes",
};

/** The settings of the view in the app store that call_action can set */
type ViewSetting =
  | "highlightedItemGroup"
  | "expandedImageIds"
  | "expandedLabelsIds"
  | "expandedPointsIds"
  | "expandedShapesIds"
  | "expandedTableIds";

/**
 * The actions of the app store that only change the view, with the setting
 * they set and, for the expanded entries, the list of the IDs they take
 */
const viewActions: Record<
  string,
  { setting: ViewSetting; list?: ObjectKind | "tables" }
> = {
  setHighlightedItemGroup: { setting: "highlightedItemGroup" },
  setExpandedImageIds: { setting: "expandedImageIds", list: "images" },
  setExpandedLabelsIds: { setting: "expandedLabelsIds", list: "labels" },
  setExpandedPointsIds: { setting: "expandedPointsIds", list: "points" },
  setExpandedShapesIds: { setting: "expandedShapesIds", list: "shapes" },
  setExpandedTableIds: { setting: "expandedTableIds", list: "tables" },
};

/** The settings of the project itself that call_action can set */
const projectActions: Record<string, "name" | "viewerBackgroundColor"> = {
  setName: "name",
  setViewerBackgroundColor: "viewerBackgroundColor",
};

/** The actions the model may not call, with the reason it is told */
const refusedActions: [RegExp, string][] = [
  [/^(delete|clear)/, "deleting cannot be taken back safely"],
  [
    /^add(Layer|Image|Labels|Points|Shapes|Table)$/,
    "new objects would need their data from a file",
  ],
  [
    /^updateTable$/,
    "a table only has a name and the source of its data, which cannot be changed",
  ],
  [/^set(OS|GL)Options$/, "the viewer options are not described in the types"],
  [
    /^setInteractionMode$/,
    "the drawing modes are not finished, and the viewer could not be panned any more",
  ],
  [/^setActivePanelId$/, "another panel could hide the chat"],
  [
    /^setImageChannelPreview$/,
    "to show one channel, call updateImage with channelViewMode and activeChannel",
  ],
  [
    /^(setWorkspace|setProjectOpen|register\w*DataProvider|setDark)$/,
    "it changes how TissUUmaps opens and loads data, or its settings",
  ],
];

/** The names of the actions the model may call, for its instructions */
const allowedActionNames = [
  ...Object.keys(updateActions),
  ...Object.values(mapLists).flatMap(({ name }) => [
    `add${name}`,
    `update${name}`,
  ]),
  ...Object.keys(moveActions),
  ...Object.keys(projectActions),
  ...Object.keys(viewActions),
];

/** What a setting that is a configuration (e.g. labelColor) can hold */
type ConfigRule = {
  value: ValueKind;
  maps: MapKind;
  /** Whether groupBy needs a map, which colors and markers do not */
  mapRequired: boolean;
  /** Colors take palettes, a range, and random colors */
  color?: boolean;
  /** Sizes take a unit */
  unit?: boolean;
};

/** The rule of the color settings */
const colorRule: ConfigRule = {
  value: "color",
  maps: "colorMaps",
  mapRequired: false,
  color: true,
};

/** The rule of the visibility settings */
const visibilityRule: ConfigRule = {
  value: "boolean",
  maps: "visibilityMaps",
  mapRequired: true,
};

/** The rule of the opacity settings */
const opacityRule: ConfigRule = {
  value: "opacity",
  maps: "opacityMaps",
  mapRequired: true,
};

/** The settings of objects that are configurations, with their rules */
const configSettings: Record<string, ConfigRule> = {
  labelColor: colorRule,
  pointColor: colorRule,
  shapeFillColor: colorRule,
  shapeStrokeColor: colorRule,
  labelVisibility: visibilityRule,
  pointVisibility: visibilityRule,
  shapeVisibility: visibilityRule,
  shapeFillVisibility: visibilityRule,
  shapeStrokeVisibility: visibilityRule,
  labelOpacity: opacityRule,
  pointOpacity: opacityRule,
  shapeOpacity: opacityRule,
  shapeFillOpacity: opacityRule,
  shapeStrokeOpacity: opacityRule,
  pointSize: {
    value: "size",
    maps: "sizeMaps",
    mapRequired: true,
    unit: true,
  },
  pointMarker: { value: "marker", maps: "markerMaps", mapRequired: false },
};

/** The marker shapes, as their numbers */
const markerValues: number[] = Object.values(Marker);

/** The ways the channels of an image can be shown */
const channelViewModes: string[] = Object.values(ImageChannelViewMode);

/** The units of sizes */
const coordinateSpaces = ["layer", "data", "world"];

/** The tool that calls an action, only offered to OpenAI and Anthropic */
const callActionDefinition = {
  type: "function",
  function: {
    name: "call_action",
    description:
      "Call an action of the app state (see its types) for a change that the other tools cannot make, " +
      "e.g. the color of an image channel, the marker of points, a transform, the order of objects. " +
      `Allowed actions: ${allowedActionNames.join(", ")}.`,
    parameters: {
      type: "object",
      required: ["store", "action", "args"],
      properties: {
        store: {
          type: "string",
          enum: ["projectStore", "appStore"],
          description:
            "The store of the action: appStore for setHighlightedItemGroup and setExpanded*.",
        },
        action: {
          type: "string",
          description: 'The name of the action, e.g. "updateImage".',
        },
        args: {
          type: "array",
          items: {},
          description:
            'The arguments of the action, in order, e.g. ["images/he_image@global", {"transform": {"rotation": 90}}]. ' +
            "For transform and channels, give only what changes; channels is a list with one entry per channel, " +
            '{} for the channels that stay. Colors are {"r":0-255,"g":0-255,"b":0-255}. Use ids from the app state.',
        },
      },
    },
  },
};

/** Whether a value is one of some texts */
function isOneOf(value: unknown, texts: string[]): boolean {
  return typeof value === "string" && texts.includes(value);
}

/** Whether a value is a plain object, e.g. the changes of an update */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Whether a value is a color, e.g. {"r": 255, "g": 0, "b": 0} */
function isRgb(value: unknown): value is Color {
  return (
    isRecord(value) &&
    Object.keys(value).length === 3 &&
    ["r", "g", "b"].every((key) => {
      const component = value[key];
      return (
        typeof component === "number" &&
        Number.isInteger(component) &&
        component >= 0 &&
        component <= 255
      );
    })
  );
}

/**
 * Checks a value of a setting or a map
 *
 * @param kind - The kind of value it should be
 * @param value - The value
 * @param what - Where the value is, for the message
 * @returns What is wrong, or "" if nothing is
 */
function checkValue(kind: ValueKind, value: unknown, what: string): string {
  switch (kind) {
    case "color":
      return isRgb(value)
        ? ""
        : `${what} must be a color {"r":0-255,"g":0-255,"b":0-255}`;
    case "boolean":
      return typeof value === "boolean" ? "" : `${what} must be true or false`;
    case "opacity":
      return typeof value === "number" && value >= 0 && value <= 1
        ? ""
        : `${what} must be a number from 0 to 1`;
    case "size":
      return typeof value === "number" && value > 0
        ? ""
        : `${what} must be a number above 0`;
    case "marker":
      return typeof value === "number" && markerValues.includes(value)
        ? ""
        : `${what} must be a marker number from 0 to 14 (e.g. 2 Square, 6 Disc)`;
  }
}

/**
 * Checks that a column of a setting exists, in the object's own table or in
 * the table the setting names
 *
 * @param context - What the tool works with
 * @param object - The object
 * @param ref - The column, as {table?, column}
 * @param what - Where the column is, for the message
 * @param options - Optional abort signal
 * @returns What is wrong, or "" if nothing is
 */
async function checkColumn(
  context: ToolContext,
  object: ProjectObject,
  ref: unknown,
  what: string,
  options?: { signal?: AbortSignal },
): Promise<string> {
  if (
    !isRecord(ref) ||
    typeof ref.column !== "string" ||
    (ref.table !== undefined && typeof ref.table !== "string")
  ) {
    return `${what} needs "column", and "table" if it is in another table`;
  }
  const column = ref.column;
  const tableId =
    typeof ref.table === "string" ? ref.table : annotatingTable(object);
  if (tableId === undefined) {
    return `${object.name} has no table of its own: give "table" in ${what}`;
  }
  const tableRef = context.dataStore.getState().tableDataRefs.get(tableId);
  if (tableRef === undefined) {
    return `the table "${tableId}" is not loaded`;
  }
  const data = await tableRef.promise;
  for (const { name, matrix } of await listColumns(data, options)) {
    if (name === column) {
      return "";
    }
    if (
      matrix &&
      column.startsWith(`${name}[`) &&
      (await matrixColumns(data, name, options)).includes(column)
    ) {
      return "";
    }
  }
  return `the table "${tableId}" has no column "${column}": give its exact name, e.g. "obs/leiden"`;
}

/**
 * Checks a setting that is a configuration, e.g. labelColor
 *
 * @param setting - The name of the setting
 * @param rule - What it can hold
 * @param config - The new configuration
 * @param object - The object
 * @param context - What the tool works with
 * @param options - Optional abort signal
 * @returns What is wrong, or "" if nothing is
 */
async function checkConfig(
  setting: string,
  rule: ConfigRule,
  config: unknown,
  object: ProjectObject,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  if (!isRecord(config)) {
    return `${setting} must be an object, e.g. {"constant":{"value":...}}`;
  }
  const sources = [
    "constant",
    "from",
    "groupBy",
    ...(rule.color === true ? ["random"] : []),
  ];
  const parts = Object.keys(config).filter((key) => key !== "source");
  const unknown = parts.filter((key) => !sources.includes(key));
  if (unknown.length > 0 || parts.length === 0) {
    return `${setting} takes one of ${list(sources)}${unknown.length > 0 ? `, not ${list(unknown)}` : ""}`;
  }
  if (config.source !== undefined && !isOneOf(config.source, parts)) {
    return `${setting}.source must name one of its parts: ${list(parts)}`;
  }
  const project = context.projectStore.getState();
  const problems: string[] = [];
  const checkUnit = (part: Record<string, unknown>, where: string) => {
    if (part.unit !== undefined && !isOneOf(part.unit, coordinateSpaces)) {
      problems.push(`${where}.unit must be one of ${list(coordinateSpaces)}`);
    }
  };
  if (isConstantConfig(config)) {
    const constant: unknown = config.constant;
    if (isRecord(constant)) {
      const problem = checkValue(
        rule.value,
        constant.value,
        `${setting}.constant.value`,
      );
      if (problem !== "") {
        problems.push(problem);
      }
      checkUnit(constant, `${setting}.constant`);
    } else {
      problems.push(`${setting}.constant must be {"value":...}`);
    }
  }
  if (isFromConfig(config)) {
    const from: unknown = config.from;
    const problem = await checkColumn(
      context,
      object,
      from,
      `${setting}.from`,
      options,
    );
    if (problem !== "") {
      problems.push(problem);
    }
    if (isRecord(from)) {
      if (rule.color === true && !isOneOf(from.palette, numberPalettes)) {
        problems.push(
          `${setting}.from.palette must be one of ${list(numberPalettes)}`,
        );
      }
      const range = from.range;
      if (
        range !== undefined &&
        !(
          Array.isArray(range) &&
          range.length === 2 &&
          typeof range[0] === "number" &&
          typeof range[1] === "number" &&
          range[0] < range[1]
        )
      ) {
        problems.push(`${setting}.from.range must be [lowest, highest]`);
      }
      checkUnit(from, `${setting}.from`);
    }
  }
  if (isGroupByConfig(config)) {
    const groupBy: unknown = config.groupBy;
    const problem = await checkColumn(
      context,
      object,
      groupBy,
      `${setting}.groupBy`,
      options,
    );
    if (problem !== "") {
      problems.push(problem);
    }
    if (isRecord(groupBy)) {
      if (groupBy.map === undefined) {
        if (rule.mapRequired) {
          problems.push(
            `${setting}.groupBy needs "map": the id of a map of ${rule.maps}`,
          );
        }
      } else if (
        typeof groupBy.map !== "string" ||
        findById(project, rule.maps, groupBy.map) === undefined
      ) {
        problems.push(
          `${setting}.groupBy.map must be the id of a map of ${rule.maps}`,
        );
      }
      if (
        groupBy.palette !== undefined &&
        !(rule.color === true && isOneOf(groupBy.palette, groupPalettes))
      ) {
        problems.push(
          `${setting}.groupBy.palette must be one of ${list(groupPalettes)}`,
        );
      }
      checkUnit(groupBy, `${setting}.groupBy`);
    }
  }
  if (isRandomConfig(config)) {
    const random: unknown = config.random;
    if (!isRecord(random) || !isOneOf(random.palette, groupPalettes)) {
      problems.push(
        `${setting}.random needs "palette", one of ${list(groupPalettes)}`,
      );
    }
  }
  return problems.join("; ");
}

/**
 * The data of an image, if it is loaded
 *
 * @param dataStore - The data store
 * @param image - The image
 * @returns The data, or `undefined` while it is not loaded
 */
function loadedImageData(
  dataStore: DataStoreApi,
  image: Image,
): ImageData | undefined {
  const ref = dataStore.getState().imageDataRefs.get(image.id);
  return ref?.status === "loaded" ? ref.data : undefined;
}

/** A channel of an image, as the Images panel shows it */
type ShownChannel = {
  name: string;
  visibility: boolean;
  opacity: number;
  color: Color;
  contrastLimits?: [number, number];
};

/**
 * The channels of an image as the Images panel shows them: as many as its
 * data has, even when the image's channels list is shorter, each with the
 * settings of the image, or else those of its data
 *
 * @param image - The image
 * @param data - The data of the image, if it is loaded
 * @returns The channels; without the data, those the image lists
 */
function shownChannels(
  image: Image,
  data: ImageData | undefined,
): ShownChannel[] {
  const sizeC = data?.getSizeC();
  if (data === undefined || sizeC === undefined) {
    return (image.channels ?? []).map((channel, c) => ({
      name: channel.name ?? `Channel ${c}`,
      visibility: channel.visibility ?? true,
      opacity: channel.opacity ?? 1,
      color: channel.color ?? ImageUtils.getDefaultChannelColor(c),
      contrastLimits: channel.contrastLimits,
    }));
  }
  return Array.from({ length: sizeC }, (_, c) => ({
    name:
      image.channels?.[c]?.name ?? data.getChannelName?.(c) ?? `Channel ${c}`,
    visibility: ImageUtils.getChannelVisibility(image, data, c),
    opacity: ImageUtils.getChannelOpacity(image, data, c),
    color: ImageUtils.getChannelColor(image, data, c),
    contrastLimits: ImageUtils.getChannelContrastLimits(image, data, c),
  }));
}

/**
 * Checks the channels of an image, and fills each entry in with the channel
 * as it is now
 *
 * Like the Images panel, it counts the channels of the image's data, and
 * lengthens the image's channels list when it is shorter, with {} for the
 * channels it does not list yet.
 *
 * @param image - The image
 * @param value - The channels the model gave, one entry per channel
 * @param data - The data of the image, if it is loaded
 * @returns What is wrong, or the channels as they will be
 */
function checkChannels(
  image: Image,
  value: unknown,
  data: ImageData | undefined,
): { problem: string } | { channels: unknown[] } {
  const current = image.channels ?? [];
  if (!Array.isArray(value)) {
    return {
      problem:
        "channels must be a list with one entry per channel, {} for the channels that stay",
    };
  }
  const shown = shownChannels(image, data);
  if (shown.length > 0 && value.length !== shown.length) {
    const names = shown.map(({ name }, c) => `${c} ${name}`);
    return {
      problem: `${image.name} has ${shown.length} channels (${names.join(", ")}): give ${shown.length} entries, {} for the channels that stay`,
    };
  }
  const known = storeProperties.ImageChannel ?? [];
  const problems: string[] = [];
  const length = Math.max(value.length, current.length);
  const channels = Array.from({ length }, (_, index) => {
    const entry: unknown = value[index] ?? {};
    const what = `channels[${index}]`;
    if (!isRecord(entry)) {
      problems.push(`${what} must be an object`);
      return entry;
    }
    for (const [key, setting] of Object.entries(entry)) {
      let problem = "";
      if (!known.includes(key)) {
        problem = `a channel has no "${key}"; it has ${list(known)}`;
      } else if (key === "name" && typeof setting !== "string") {
        problem = `${what}.name must be a text`;
      } else if (key === "visibility") {
        problem = checkValue("boolean", setting, `${what}.visibility`);
      } else if (key === "opacity") {
        problem = checkValue("opacity", setting, `${what}.opacity`);
      } else if (key === "color") {
        problem = checkValue("color", setting, `${what}.color`);
      } else if (
        key === "contrastLimits" &&
        !(
          Array.isArray(setting) &&
          setting.length === 2 &&
          typeof setting[0] === "number" &&
          typeof setting[1] === "number" &&
          setting[0] < setting[1]
        )
      ) {
        problem = `${what}.contrastLimits must be [lowest, highest]`;
      }
      if (problem !== "") {
        problems.push(problem);
      }
    }
    return { ...current[index], ...entry };
  });
  return problems.length > 0 ? { problem: problems.join("; ") } : { channels };
}

/**
 * Checks the changes of an update action, and fills in what the model may
 * give in part (the transform and the channels)
 *
 * @param type - The type of the entry, e.g. "Image"
 * @param object - The entry
 * @param updates - The changes the model gave
 * @param context - What the tool works with
 * @param options - Optional abort signal
 * @returns What is wrong, and the changes as they will be made
 */
async function checkUpdates(
  type: string,
  object: ProjectObject | Layer,
  updates: Record<string, unknown>,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<{ problems: string[]; values: Record<string, unknown> }> {
  const project = context.projectStore.getState();
  const known = storeProperties[type] ?? [];
  const problems: string[] = [];
  const values: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(updates)) {
    values[key] = value;
    if (key === "id" || key === "dataSource") {
      problems.push(`${key} cannot be changed`);
      continue;
    }
    if (!known.includes(key)) {
      const changeable = known.filter(
        (name) => name !== "id" && name !== "dataSource",
      );
      problems.push(`${type} has no "${key}"; it has ${list(changeable)}`);
      continue;
    }
    const rule = configSettings[key];
    let problem = "";
    if (rule !== undefined && "dataSource" in object) {
      problem = await checkConfig(key, rule, value, object, context, options);
    } else if (key === "name") {
      problem =
        typeof value === "string" && value.trim() !== ""
          ? ""
          : "name must be a text that is not empty";
    } else if (key === "visibility") {
      problem = checkValue("boolean", value, key);
    } else if (key === "opacity") {
      problem = checkValue("opacity", value, key);
    } else if (key === "pointSizeFactor") {
      problem = checkValue("size", value, key);
    } else if (key === "layer") {
      problem =
        typeof value === "string" &&
        findById(project, "layers", value) !== undefined
          ? ""
          : `layer must be the id of a layer: ${list(project.layers.map(({ id }) => id))}`;
    } else if (key === "transform") {
      const transform = object.transform;
      if (isRecord(value)) {
        const merged = {
          ...transform,
          ...value,
          translation: {
            ...transform.translation,
            ...(isRecord(value.translation) ? value.translation : {}),
          },
        };
        values[key] = merged;
        const valid =
          typeof merged.flip === "boolean" &&
          typeof merged.scale === "number" &&
          merged.scale > 0 &&
          typeof merged.rotation === "number" &&
          Number.isFinite(merged.rotation) &&
          Number.isFinite(merged.translation.x) &&
          Number.isFinite(merged.translation.y);
        problem = valid
          ? ""
          : "transform takes flip (true or false), scale (above 0), rotation (in degrees) and translation {x, y}";
      } else {
        problem = 'transform must be an object, e.g. {"rotation": 90}';
      }
    } else if (key === "channelViewMode") {
      problem = isOneOf(value, channelViewModes)
        ? ""
        : `channelViewMode must be one of ${list(channelViewModes)}`;
    } else if (key === "activeChannel") {
      const image = object as Image;
      const count = shownChannels(
        image,
        loadedImageData(context.dataStore, image),
      ).length;
      problem =
        typeof value === "number" &&
        Number.isInteger(value) &&
        value >= 0 &&
        (count === 0 || value < count)
          ? ""
          : `activeChannel must be the number of a channel, from 0${count === 0 ? "" : ` to ${count - 1}`}`;
    } else if (key === "channels") {
      const image = object as Image;
      const checked = checkChannels(
        image,
        value,
        loadedImageData(context.dataStore, image),
      );
      if ("problem" in checked) {
        problem = checked.problem;
      } else {
        values[key] = checked.channels;
      }
    } else {
      problem = `${key} cannot be changed with call_action`;
    }
    if (problem !== "") {
      problems.push(problem);
    }
  }
  return { problems, values };
}

/**
 * Checks the values of a map
 *
 * @param kind - The list of the map
 * @param map - The map, or the changes of a map
 * @returns What is wrong, or "" if nothing is
 */
function checkMap(kind: MapKind, map: Record<string, unknown>): string {
  const { value } = mapLists[kind];
  const problems: string[] = [];
  for (const [key, entry] of Object.entries(map)) {
    if (key === "name") {
      if (typeof entry !== "string" || entry.trim() === "") {
        problems.push("name must be a text that is not empty");
      }
    } else if (key === "values") {
      if (isRecord(entry)) {
        for (const [group, setting] of Object.entries(entry)) {
          const problem = checkValue(value, setting, `values["${group}"]`);
          if (problem !== "") {
            problems.push(problem);
          }
        }
      } else {
        problems.push(
          'values must be an object of group → value, e.g. {"3": ...}',
        );
      }
    } else if (key === "default") {
      const problem =
        entry === undefined ? "" : checkValue(value, entry, "default");
      if (problem !== "") {
        problems.push(problem);
      }
    } else if (key !== "id") {
      problems.push(
        `a map has no "${key}"; it has id, name, values and default`,
      );
    }
  }
  return problems.join("; ");
}

/** Moves an entry of a list to another index, with its project store action */
function applyMove(
  project: ProjectStore,
  kind: ObjectKind | "layers",
  id: string,
  index: number,
): void {
  switch (kind) {
    case "layers":
      project.moveLayer(id, index);
      break;
    case "images":
      project.moveImage(id, index);
      break;
    case "labels":
      project.moveLabels(id, index);
      break;
    case "points":
      project.movePoints(id, index);
      break;
    case "shapes":
      project.moveShapes(id, index);
      break;
  }
}

/** Adds a map, with its project store action */
function applyAddMap(
  project: ProjectStore,
  kind: MapKind,
  map: GroupValueMap<unknown>,
): void {
  switch (kind) {
    case "markerMaps":
      project.addMarkerMap(map as GroupValueMap<Marker>);
      break;
    case "sizeMaps":
      project.addSizeMap(map as GroupValueMap<number>);
      break;
    case "colorMaps":
      project.addColorMap(map as GroupValueMap<Color>);
      break;
    case "visibilityMaps":
      project.addVisibilityMap(map as GroupValueMap<boolean>);
      break;
    case "opacityMaps":
      project.addOpacityMap(map as GroupValueMap<number>);
      break;
  }
}

/** Deletes a map, with its project store action, for "Undo" */
function deleteMap(project: ProjectStore, kind: MapKind, id: string): void {
  switch (kind) {
    case "markerMaps":
      project.deleteMarkerMap(id);
      break;
    case "sizeMaps":
      project.deleteSizeMap(id);
      break;
    case "colorMaps":
      project.deleteColorMap(id);
      break;
    case "visibilityMaps":
      project.deleteVisibilityMap(id);
      break;
    case "opacityMaps":
      project.deleteOpacityMap(id);
      break;
  }
}

/** Sets a setting of the project itself, with its project store action */
function setProjectSetting(
  project: ProjectStore,
  property: string,
  value: unknown,
): void {
  if (property === "name") {
    project.setName(value as string);
  } else if (property === "viewerBackgroundColor") {
    project.setViewerBackgroundColor(value as Color);
  }
}

/** Sets a setting of the view, with its app store action */
function setView(app: AppStore, setting: string, value: unknown): void {
  switch (setting) {
    case "highlightedItemGroup":
      app.setHighlightedItemGroup(
        value as Parameters<AppStore["setHighlightedItemGroup"]>[0],
      );
      break;
    case "expandedImageIds":
      app.setExpandedImageIds(value as string[]);
      break;
    case "expandedLabelsIds":
      app.setExpandedLabelsIds(value as string[]);
      break;
    case "expandedPointsIds":
      app.setExpandedPointsIds(value as string[]);
      break;
    case "expandedShapesIds":
      app.setExpandedShapesIds(value as string[]);
      break;
    case "expandedTableIds":
      app.setExpandedTableIds(value as string[]);
      break;
  }
}

/**
 * The name of an entry of the project by its ID, to show actions readably
 *
 * @param project - The project
 * @param id - The ID
 * @returns The name, or `undefined` if no entry has that ID
 */
function nameById(project: ProjectStore, id: string): string | undefined {
  const lists: ListKind[] = [
    ...objectKinds,
    "layers",
    ...(Object.keys(mapLists) as MapKind[]),
  ];
  for (const kind of lists) {
    const entry = findById(project, kind, id);
    if (entry !== undefined) {
      return entry.name;
    }
  }
  return project.tables.find((table) => table.id === id)?.name;
}

/** The parts of a configuration, e.g. the "groupBy" of {"groupBy": ...} */
const configParts = ["source", "constant", "from", "groupBy", "random"];

/**
 * Whether a value is shown whole rather than part by part: a color, or a
 * configuration like {"groupBy": ...}, but not an object that only has a
 * part with such a name, like the group of setHighlightedItemGroup
 *
 * @param value - The value
 * @returns Whether it is shown whole
 */
function isWhole(value: unknown): boolean {
  return (
    isRgb(value) ||
    (isRecord(value) &&
      Object.keys(value).length > 0 &&
      Object.keys(value).every((key) => configParts.includes(key)))
  );
}

/**
 * The parts of a value that are set, one per line of the Tool row: e.g.
 * "channels[1].color: {...}"; empty entries ({} for a channel that stays)
 * are left out, colors and configurations are shown whole, and the IDs of
 * entries of the project by their names
 *
 * @param value - The value
 * @param path - Where it is
 * @param name - The name of an entry by its ID, if it has one
 * @returns The parts
 */
function setParts(
  value: unknown,
  path: string,
  name: (id: string) => string | undefined,
): string[] {
  if (Array.isArray(value) && value.some(isRecord)) {
    return value.flatMap((entry, index) =>
      setParts(entry, `${path}[${index}]`, name),
    );
  }
  if (isRecord(value) && !isWhole(value)) {
    return Object.entries(value).flatMap(([key, entry]) =>
      setParts(entry, path === "" ? key : `${path}.${key}`, name),
    );
  }
  const show = (entry: unknown) =>
    (typeof entry === "string" ? name(entry) : undefined) ??
    JSON.stringify(entry);
  const shown =
    Array.isArray(value) && value.every((entry) => typeof entry === "string")
      ? `[${value.map(show).join(", ")}]`
      : show(value);
  return [path === "" ? shown : `${path}: ${shown}`];
}

/**
 * Shows an action readably, with names instead of IDs, e.g.
 * 'updateImage(morphology_focus, {channels[1].color: {"r":255,"g":0,"b":0}})'
 * or 'setHighlightedItemGroup({annotatedObject.labelsId: cell_labels,
 * groupBy.column: "obs/leiden", group: "3"})'
 *
 * @param project - The project
 * @param args - The arguments of call_action
 * @returns The text
 */
function describeAction(project: ProjectStore, args: ToolArguments): string {
  const actionArgs = Array.isArray(args.args) ? (args.args as unknown[]) : [];
  const name = (id: string) => nameById(project, id);
  const parts = actionArgs.map((arg) => {
    const shown = setParts(arg, "", name);
    return isRecord(arg) && !isWhole(arg)
      ? `{${shown.join(", ")}}`
      : shown.join(", ");
  });
  return `${String(args.action)}(${parts.join(", ")})`;
}

/** What an action changed, and what it left as it already was */
type Outcome = { changed: string[]; already: string[] };

/**
 * Shows a value in a tool result, e.g. red, 0.5, "DAPI" or true
 *
 * @param value - The value
 * @returns The text, cut after 150 characters
 */
function showValue(value: unknown): string {
  if (isRgb(value)) {
    return colorName(value);
  }
  if (typeof value === "number") {
    return String(round(value));
  }
  if (value === undefined) {
    return "not set";
  }
  const text = JSON.stringify(value);
  return text.length > 150 ? `${text.slice(0, 150)}...` : text;
}

/**
 * Compares what an action sets with the values from before, part by part like
 * {@link setParts}, e.g. "transform.rotation 0 → 90", or
 * "transform.rotation = 90" when it already was 90
 *
 * @param before - The value from before
 * @param after - What the action sets
 * @param path - Where it is
 * @param outcome - What changed and what did not, which this adds to
 */
function compareParts(
  before: unknown,
  after: unknown,
  path: string,
  outcome: Outcome,
): void {
  if (Array.isArray(after) && after.some(isRecord)) {
    const old: unknown[] = Array.isArray(before) ? before : [];
    after.forEach((entry: unknown, index) => {
      compareParts(old[index], entry, `${path}[${index}]`, outcome);
    });
    return;
  }
  if (isRecord(after) && !isWhole(after)) {
    for (const [key, entry] of Object.entries(after)) {
      const old = isRecord(before) ? before[key] : undefined;
      compareParts(old, entry, path === "" ? key : `${path}.${key}`, outcome);
    }
    return;
  }
  if (same(before, after)) {
    outcome.already.push(`${path} = ${showValue(after)}`);
  } else {
    outcome.changed.push(`${path} ${showValue(before)} → ${showValue(after)}`);
  }
}

/**
 * The result of an action for the model, which lists exactly what changed and
 * what already was so, so that the model cannot claim more, e.g. "Done.
 * Changed in he_image: opacity 1 → 0.5. Already so: visibility = true."
 *
 * @param name - What the action changed, e.g. the name of an object
 * @param outcome - What changed and what did not
 * @returns The text
 */
function outcomeText(name: string, outcome: Outcome): string {
  const already = outcome.already.join("; ");
  if (outcome.changed.length === 0) {
    return already === ""
      ? `Nothing changed: no change for ${name} was given.`
      : `Nothing changed: in ${name} already ${already}.`;
  }
  return (
    `Done. Changed in ${name}: ${outcome.changed.join("; ")}.` +
    (already === "" ? "" : ` Already so: ${already}.`)
  );
}

/**
 * Describes a highlighted group, e.g. 'group "3" of column "obs/leiden" of
 * cell_labels', or "none"
 *
 * @param project - The project
 * @param value - The group, as setHighlightedItemGroup takes it
 * @returns The text
 */
function describeHighlight(project: ProjectStore, value: unknown): string {
  if (!isRecord(value)) {
    return "none";
  }
  const id = isRecord(value.annotatedObject)
    ? Object.values(value.annotatedObject)[0]
    : undefined;
  const object =
    typeof id === "string" ? (nameById(project, id) ?? id) : "an object";
  const column = isRecord(value.groupBy) ? value.groupBy.column : undefined;
  return `group ${showValue(value.group)} of column ${showValue(column)} of ${object}`;
}

/**
 * The tool call_action: calls one allowed action of the app state, after
 * checking all its arguments, through {@link change} and the like, so that
 * "Undo" takes it back
 */
async function callAction(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const { action, store } = args;
  const actionArgs = Array.isArray(args.args) ? (args.args as unknown[]) : [];
  if (typeof action !== "string" || action === "") {
    return 'Error: give action, e.g. "updateImage".';
  }
  const refusal = refusedActions.find(([pattern]) => pattern.test(action));
  if (refusal !== undefined) {
    return `Error: ${action} is not allowed: ${refusal[1]}. Nothing changed.`;
  }
  const mapAction = /^(add|update)(\w+Map)$/.exec(action);
  const mapKind = (Object.keys(mapLists) as MapKind[]).find(
    (kind) => mapLists[kind].name === mapAction?.[2],
  );
  if (
    !allowedActionNames.includes(action) ||
    (mapAction !== null && mapKind === undefined)
  ) {
    return `Error: there is no action "${action}" that you can call. You can call: ${allowedActionNames.join(", ")}.`;
  }
  const storeName = action in viewActions ? "appStore" : "projectStore";
  if (store !== storeName) {
    return `Error: ${action} is an action of ${storeName}: call it with store "${storeName}".`;
  }
  const project = context.projectStore.getState();
  const { changes } = context;
  const [first, second] = actionArgs;

  const update = updateActions[action];
  if (update !== undefined) {
    const object =
      typeof first === "string"
        ? (findById(project, update.kind, first) as
            ProjectObject | Layer | undefined)
        : undefined;
    if (object === undefined) {
      return `Error: there is no entry of ${update.kind} with id ${JSON.stringify(first)}. Use an id from the app state.`;
    }
    if (!isRecord(second) || Object.keys(second).length === 0) {
      return 'Error: give the changes as the second argument, e.g. {"opacity": 0.5}.';
    }
    const { problems, values } = await checkUpdates(
      update.type,
      object,
      second,
      context,
      options,
    );
    if (problems.length > 0) {
      return `Error: nothing changed: ${problems.join("; ")}.`;
    }
    const current = object as unknown as Record<string, unknown>;
    const image = update.kind === "images" ? (object as Image) : undefined;
    const channels =
      image === undefined
        ? []
        : shownChannels(image, loadedImageData(context.dataStore, image));
    const outcome: Outcome = { changed: [], already: [] };
    for (const [key, value] of Object.entries(values)) {
      const part: Outcome = { changed: [], already: [] };
      if (key === "channels" && Array.isArray(second.channels)) {
        // each channel as the panel shows it, with its name
        second.channels.forEach((entry: unknown, c) => {
          const name = channels[c]?.name ?? `Channel ${c}`;
          compareParts(channels[c], entry, `channels[${c}] (${name})`, part);
        });
      } else {
        // the transform as the model gave it: only what changes
        const after = key === "transform" ? second.transform : value;
        compareParts(current[key], after, key, part);
      }
      // only what changes is changed, and taken back by "Undo"
      if (part.changed.length > 0) {
        change(project, changes, update.kind, object.id, key, value);
      }
      outcome.changed.push(...part.changed);
      outcome.already.push(...part.already);
    }
    let notes = "";
    const aboutChannels = ["channels", "channelViewMode", "activeChannel"].some(
      (key) => key in values,
    );
    if (outcome.changed.length > 0 && image !== undefined && aboutChannels) {
      // checked above: a view mode and a channel number
      const mode = (values.channelViewMode ?? image.channelViewMode) as
        string | undefined;
      const active = Number(values.activeChannel ?? image.activeChannel ?? 0);
      if (mode !== undefined && mode !== ImageChannelViewMode.composite) {
        const name = channels[active]?.name ?? `Channel ${active}`;
        notes += ` In the ${mode} view mode, only channel ${active} (${name}) is shown, whatever the visibility of the channels.`;
      }
    }
    if (
      outcome.changed.length > 0 &&
      !object.visibility &&
      values.visibility !== true
    ) {
      notes += ` ${object.name} is hidden: show it to see the change.`;
    }
    return `${outcomeText(object.name, outcome)}${notes}`;
  }

  if (mapKind !== undefined && mapAction?.[1] === "add") {
    if (!isRecord(first) || typeof first.id !== "string" || first.id === "") {
      return 'Error: give the map as the argument: {"id": ..., "name": ..., "values": {...}}.';
    }
    if (findById(project, mapKind, first.id) !== undefined) {
      return `Error: a map of ${mapKind} with id "${first.id}" already exists: call update${mapLists[mapKind].name}.`;
    }
    // name and values are checked also when they are missing
    const problem = checkMap(mapKind, {
      ...first,
      name: first.name,
      values: first.values,
    });
    if (problem !== "") {
      return `Error: nothing changed: ${problem}.`;
    }
    const map = first as GroupValueMap<unknown>;
    applyAddMap(project, mapKind, map);
    changes.push({
      type: "added",
      kind: mapKind,
      id: map.id,
      name: map.name,
      property: "",
      before: undefined,
      after: map,
    });
    const groups = Object.keys(map.values);
    return `Done. Added to ${mapKind} the map "${map.name}" (id ${map.id}), with ${groups.length} group${groups.length === 1 ? "" : "s"}: ${list(groups)}.`;
  }

  if (mapKind !== undefined) {
    const map =
      typeof first === "string" ? findById(project, mapKind, first) : undefined;
    if (map === undefined) {
      return `Error: there is no map of ${mapKind} with id ${JSON.stringify(first)}.`;
    }
    if (!isRecord(second) || Object.keys(second).length === 0) {
      return 'Error: give the changes as the second argument, e.g. {"values": {...}}.';
    }
    const problem = checkMap(mapKind, second);
    if (problem !== "") {
      return `Error: nothing changed: ${problem}.`;
    }
    const current = map as Record<string, unknown>;
    const outcome: Outcome = { changed: [], already: [] };
    for (const [key, value] of Object.entries(second)) {
      const part: Outcome = { changed: [], already: [] };
      compareParts(current[key], value, key, part);
      // new values replace the old ones: the groups left out are removed
      if (key === "values" && isRecord(value) && isRecord(current.values)) {
        for (const [group, old] of Object.entries(current.values)) {
          if (!(group in value)) {
            part.changed.push(`values.${group} ${showValue(old)} → not set`);
          }
        }
      }
      if (part.changed.length > 0) {
        change(project, changes, mapKind, map.id, key, value);
      }
      outcome.changed.push(...part.changed);
      outcome.already.push(...part.already);
    }
    return outcomeText(`the map "${map.name}"`, outcome);
  }

  const move = moveActions[action];
  if (move !== undefined) {
    const list: readonly { id: string; name: string }[] = project[move];
    const from = list.findIndex((entry) => entry.id === first);
    const entry = list[from];
    if (entry === undefined) {
      return `Error: there is no entry of ${move} with id ${JSON.stringify(first)}.`;
    }
    if (
      typeof second !== "number" ||
      !Number.isInteger(second) ||
      second < 0 ||
      second >= list.length
    ) {
      return `Error: the new index must be a number from 0 to ${list.length - 1}.`;
    }
    if (second === from) {
      return `Nothing changed: ${entry.name} is already at index ${from}.`;
    }
    applyMove(project, move, entry.id, second);
    changes.push({
      type: "moved",
      kind: move,
      id: entry.id,
      name: entry.name,
      property: "index",
      before: from,
      after: second,
    });
    return `Done. Moved ${entry.name} from index ${from} to ${second}.`;
  }

  const projectSetting = projectActions[action];
  if (projectSetting !== undefined) {
    const problem =
      projectSetting === "name"
        ? typeof first === "string" && first.trim() !== ""
          ? ""
          : "the name must be a text that is not empty"
        : checkValue("color", first, "the color");
    if (problem !== "") {
      return `Error: nothing changed: ${problem}.`;
    }
    const before = project[projectSetting];
    if (same(before, first)) {
      return `Nothing changed: the project ${projectSetting} already is ${showValue(first)}.`;
    }
    changes.push({
      kind: "project",
      id: "",
      name: "the project",
      property: projectSetting,
      before,
      after: first,
    });
    setProjectSetting(project, projectSetting, first);
    return `Done. Changed in the project: ${projectSetting} ${showValue(before)} → ${showValue(first)}.`;
  }

  const view = viewActions[action];
  if (view !== undefined) {
    const app = context.appStore.getState();
    let problem = "";
    if (view.list !== undefined) {
      const ids: readonly { id: string }[] = project[view.list];
      problem =
        Array.isArray(first) &&
        first.every((id) => ids.some((entry) => entry.id === id))
          ? ""
          : `give a list of ids of ${view.list}`;
    } else if (first !== null) {
      problem = await checkHighlight(project, context, first, options);
    }
    if (problem !== "") {
      return `Error: nothing changed: ${problem}.`;
    }
    const before = app[view.setting];
    const show = (value: unknown): string =>
      view.list === undefined
        ? describeHighlight(project, value)
        : `[${list((value as string[]).map((id) => nameById(project, id) ?? id))}]`;
    if (same(before, first)) {
      return `Nothing changed: ${view.setting} already is ${show(first)}.`;
    }
    changes.push({
      kind: "app",
      id: "",
      name: "the view",
      property: view.setting,
      before,
      after: first,
    });
    setView(app, view.setting, first);
    return `Done. Changed in the view: ${view.setting} ${show(before)} → ${show(first)}.`;
  }
  return `Error: there is no action "${action}" that you can call.`;
}

/**
 * Checks a group to highlight: its object, column and group
 *
 * @param project - The project
 * @param context - What the tool works with
 * @param value - The group, as setHighlightedItemGroup takes it
 * @param options - Optional abort signal
 * @returns What is wrong, or "" if nothing is
 */
async function checkHighlight(
  project: ProjectStore,
  context: ToolContext,
  value: unknown,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const shape =
    'give {"annotatedObject": {"labelsId": ...} (or pointsId, shapesId), "groupBy": {"column": ...}, "group": "..."}, or null for none';
  if (
    !isRecord(value) ||
    !isRecord(value.annotatedObject) ||
    typeof value.group !== "string"
  ) {
    return shape;
  }
  const kinds: Record<string, ObjectKind> = {
    labelsId: "labels",
    pointsId: "points",
    shapesId: "shapes",
  };
  const [key, id] = Object.entries(value.annotatedObject)[0] ?? [];
  const kind = key === undefined ? undefined : kinds[key];
  const object =
    kind === undefined || typeof id !== "string"
      ? undefined
      : (findById(project, kind, id) as ProjectObject | undefined);
  if (object === undefined) {
    return `there is no object ${JSON.stringify(value.annotatedObject)}: ${shape}`;
  }
  return checkColumn(context, object, value.groupBy, "groupBy", options);
}

// --- Data analysis, for OpenAI and Anthropic ---
// The tools read columns of the tables and compute statistics with stats.ts;
// they change nothing in the project.

/** By default, and at most, this many groups or genes are listed in a result */
const defaultListed = 10;
const maxListed = 20;

/** The labels or points whose items a data tool looks at */
const analysisIdParameter = {
  type: "string",
  description:
    "The id of the labels or points whose items (e.g. cells) to look at, from the summary.",
};

/** A column of groups for a data tool */
const analysisByParameter = {
  type: "string",
  description:
    'A column of groups, e.g. "obs/leiden", from "columns you can use". Without it, the column the object is colored by now.',
};

/** A group of the column "by" */
const analysisGroupParameter = {
  type: "string",
  description: 'A group of the column "by", e.g. "3".',
};

/** Another group to compare with */
const analysisReferenceParameter = {
  type: "string",
  description:
    'Another group of "by" to compare with. Without it, the group is compared with all the other items.',
};

/** How many groups or genes to list */
const analysisLimitParameter = {
  type: "integer",
  description: `How many groups or genes to list, ${defaultListed} if not given, at most ${maxListed}.`,
};

/** The tools that compute statistics, only offered to OpenAI and Anthropic */
const analysisDefinitions = [
  {
    type: "function",
    function: {
      name: "column_stats",
      description:
        "Statistics of a column of labels or points, from its table. A column of groups (e.g. obs/leiden): " +
        "the number of items in each group. Numbers or a gene (e.g. obs/total_counts, EPCAM): count, min, " +
        "max, mean, median and % above 0. With by: the same for each group of by, sorted (e.g. the mean " +
        "EPCAM in each cluster), or for a column of groups, its main groups within each group of by " +
        "(e.g. the cell types in each cluster). Changes nothing.",
      parameters: {
        type: "object",
        required: ["id", "column"],
        properties: {
          id: analysisIdParameter,
          column: {
            type: "string",
            description:
              'The column, e.g. "obs/leiden", "obs/total_counts", or a gene such as "EPCAM".',
          },
          by: {
            ...analysisByParameter,
            description:
              'A column of groups, e.g. "obs/leiden", to give the statistics of each group.',
          },
          sort: {
            type: "string",
            enum: ["count", "mean", "median", "percent_above_0"],
            description:
              "What to sort the groups by: count for groups, mean for numbers if not given.",
          },
          order: {
            type: "string",
            enum: ["highest", "lowest"],
            description: "highest first (if not given) or lowest first.",
          },
          limit: analysisLimitParameter,
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "marker_genes",
      description:
        "The marker genes of a group, like scanpy's rank_genes_groups with the Wilcoxon test: for each " +
        "gene, the log2 fold change, the % of items expressing it in the group and outside, and the " +
        "Wilcoxon rank-sum p-value, adjusted with Benjamini-Hochberg over all the genes. Reads every " +
        "gene, so it can take a while; the result is kept for later questions. Changes nothing.",
      parameters: {
        type: "object",
        required: ["id", "group"],
        properties: {
          id: analysisIdParameter,
          by: analysisByParameter,
          group: analysisGroupParameter,
          reference: analysisReferenceParameter,
          direction: {
            type: "string",
            enum: ["up", "down"],
            description:
              "up (if not given): the genes higher in the group; down: the genes lower in the group.",
          },
          limit: analysisLimitParameter,
          matrix: {
            type: "string",
            description:
              'The matrix of genes, "X" if not given, or e.g. "layers/counts".',
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "compare_groups",
      description:
        "Compares a column of numbers or a gene between a group and another group (or all the other " +
        "items), e.g. the area of cluster 3 and cluster 5: count, mean and median of each, their " +
        "differences, and the Wilcoxon rank-sum test. Changes nothing.",
      parameters: {
        type: "object",
        required: ["id", "column", "group"],
        properties: {
          id: analysisIdParameter,
          column: {
            type: "string",
            description:
              'The column of numbers or the gene, e.g. "obs/cell_area" or "EPCAM".',
          },
          by: analysisByParameter,
          group: analysisGroupParameter,
          reference: analysisReferenceParameter,
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "correlation",
      description:
        "The Spearman correlation between two columns of numbers or genes, over all the items or " +
        "within one group, with its p-value. Changes nothing.",
      parameters: {
        type: "object",
        required: ["id", "column_a", "column_b"],
        properties: {
          id: analysisIdParameter,
          column_a: {
            type: "string",
            description: 'The first column or gene, e.g. "EPCAM".',
          },
          column_b: {
            type: "string",
            description: 'The second column or gene, e.g. "KRT7".',
          },
          by: {
            ...analysisByParameter,
            description:
              'A column of groups, e.g. "obs/leiden", with group: only the items of that group.',
          },
          group: {
            type: "string",
            description:
              'Only the items of this group of by, e.g. "3". Without it, all the items.',
          },
        },
      },
    },
  },
];

/** The names of the tools only offered to OpenAI and Anthropic */
const wholeAppToolNames = [
  "call_action",
  ...analysisDefinitions.map((definition) => definition.function.name),
];

/**
 * Writes a number of a result readably, with 4 significant digits
 *
 * @param value - The number
 * @returns The text, e.g. "45.23" or "0.01234"
 */
function short(value: number): string {
  return Number.isFinite(value) ? String(Number(value.toPrecision(4))) : "-";
}

/**
 * Writes a count readably, e.g. "162,254"
 *
 * @param value - The count
 * @returns The text
 */
function count(value: number): string {
  return value.toLocaleString("en-US");
}

/**
 * Writes a fraction as a percentage, e.g. "12.4%"
 *
 * @param fraction - The fraction, from 0 to 1
 * @returns The text
 */
function percent(fraction: number): string {
  return Number.isFinite(fraction)
    ? `${Math.round(fraction * 1000) / 10}%`
    : "-";
}

/**
 * Describes a summary of numbers in a few words
 *
 * @param summary - The summary
 * @returns The text, e.g. "mean 45.2, median 41, min 3, max 410, above 0 in 100%"
 */
function describeSummary(summary: Summary): string {
  return (
    `mean ${short(summary.mean)}, median ${short(summary.median)}, ` +
    `min ${short(summary.min)}, max ${short(summary.max)}, above 0 in ${percent(summary.positive)}`
  );
}

/**
 * The labels or points a data tool looks at
 *
 * @param project - The project
 * @param id - The id the model gave
 * @returns The object, or an error message for the model
 */
function analysisObject(
  project: ProjectStore,
  id: unknown,
): ItemsObject | { error: string } {
  const objects: ItemsObject[] = [...project.labels, ...project.points];
  return (
    objects.find((object) => object.id === id) ?? {
      error: `Error: there are no labels or points with id ${JSON.stringify(id)}. Give the id of labels or points from the summary.`,
    }
  );
}

/**
 * The result of a data tool for a table too big to read
 *
 * @param tableId - The table
 * @param rows - Its number of rows
 * @returns The error message for the model
 */
function tooBigError(tableId: string, rows: number): string {
  return (
    `Error: the table ${tableId} has ${count(rows)} rows, too many to compute statistics in the browser. ` +
    "Tell the user; the table of the cells (one row per cell) can answer many such questions."
  );
}

/**
 * Finds the columns a data tool reads, see {@link findColumn}
 *
 * @param context - What the tool works with
 * @param object - The labels or points
 * @param names - The column names the model gave; one that is not given is
 * the column the object is colored by now
 * @param options - Optional abort signal
 * @returns The columns, or an error message for the model
 */
async function findColumns(
  context: ToolContext,
  object: ItemsObject,
  names: unknown[],
  options?: { signal?: AbortSignal },
): Promise<FoundColumn[] | { error: string }> {
  const found: FoundColumn[] = [];
  for (const name of names) {
    const column = await findColumn(
      context.dataStore,
      context.tables,
      object,
      name,
      options,
    );
    if ("error" in column) {
      return column;
    }
    found.push(column);
  }
  return found;
}

/**
 * Puts the values of a column of another table in the rows of a table, by the
 * IDs of their items, as TissUUmaps matches the rows of two tables
 *
 * @param base - The table whose rows to use
 * @param other - The table of the values
 * @param values - The values, in the rows of the other table
 * @param options - Optional abort signal
 * @returns The values in the rows of the base table: NaN, or undefined for
 * texts, for the items the other table does not have
 */
async function alignRows(
  base: TableData,
  other: TableData,
  values: TypedArrayOrArray<unknown>,
  options?: { signal?: AbortSignal },
): Promise<TypedArrayOrArray<unknown>> {
  const numbers = ArrayBuffer.isView(values);
  const aligned: Float64Array | unknown[] = numbers
    ? new Float64Array(base.getSize()).fill(NaN)
    : new Array<unknown>(base.getSize()).fill(undefined);
  await TableUtils.forEachRow(
    base.getIds(),
    other,
    (row, i) => {
      if (row !== undefined) {
        aligned[i] = numbers ? Number(values[row]) : values[row];
      }
    },
    options,
  );
  return aligned;
}

/**
 * Reads columns for a data tool, all in the rows of the first column's table
 *
 * @param found - The columns
 * @param options - Optional abort signal
 * @returns The values of each column
 */
async function loadAligned(
  found: FoundColumn[],
  options?: { signal?: AbortSignal },
): Promise<TypedArrayOrArray<unknown>[]> {
  const base = (found[0] as FoundColumn).data;
  const columns: TypedArrayOrArray<unknown>[] = [];
  for (const { data, column } of found) {
    const values = await data.loadValues<unknown>(column, options);
    columns.push(
      data === base ? values : await alignRows(base, data, values, options),
    );
  }
  return columns;
}

/**
 * The numbers of a column, or `undefined` if it does not hold numbers
 *
 * @param values - The values
 * @returns The numbers, missing ones as NaN
 */
function numbersOf(
  values: TypedArrayOrArray<unknown>,
): Float64Array | undefined {
  return ArrayBuffer.isView(values)
    ? Float64Array.from(values as ArrayLike<number>)
    : undefined;
}

/** The groups of a column, and the group of each row */
type GroupedRows = {
  /** The groups, sorted like people expect */
  groups: string[];
  /** The index of the group of each row, or -1 when the value is missing */
  codes: Int32Array;
};

/**
 * Finds the groups of a column, as texts like TissUUmaps groups them
 *
 * @param values - The values
 * @returns The groups, or `undefined` with more than {@link maxCategories}
 */
function groupRows(
  values: TypedArrayOrArray<unknown>,
): GroupedRows | undefined {
  const index = new Map<string, number>();
  const codes = new Int32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    // the values of a table are texts, numbers or booleans; others are missing
    const key =
      typeof value === "string"
        ? value
        : (typeof value === "number" && !Number.isNaN(value)) ||
            typeof value === "boolean"
          ? String(value)
          : undefined;
    if (key === undefined) {
      codes[i] = -1;
      continue;
    }
    let code = index.get(key);
    if (code === undefined) {
      if (index.size >= maxCategories) {
        return undefined;
      }
      code = index.size;
      index.set(key, code);
    }
    codes[i] = code;
  }
  const groups = sortGroups(index.keys());
  const sortedIndex = new Int32Array(groups.length);
  groups.forEach((group, k) => {
    sortedIndex[index.get(group) as number] = k;
  });
  for (let i = 0; i < codes.length; i++) {
    const code = codes[i] as number;
    if (code >= 0) {
      codes[i] = sortedIndex[code] as number;
    }
  }
  return { groups, codes };
}

/**
 * Splits numbers by group, each part sorted and without missing values
 *
 * @param numbers - The numbers
 * @param codes - The group of each row, see {@link GroupedRows}
 * @param groupCount - The number of groups
 * @returns The sorted numbers of each group
 */
function splitByGroup(
  numbers: Float64Array,
  codes: Int32Array,
  groupCount: number,
): Float64Array[] {
  const sizes = new Int32Array(groupCount);
  for (let i = 0; i < numbers.length; i++) {
    const code = codes[i] as number;
    if (code >= 0 && !Number.isNaN(numbers[i])) {
      sizes[code] = (sizes[code] as number) + 1;
    }
  }
  const parts = Array.from(sizes, (size) => new Float64Array(size));
  const filled = new Int32Array(groupCount);
  for (let i = 0; i < numbers.length; i++) {
    const code = codes[i] as number;
    const value = numbers[i] as number;
    if (code >= 0 && !Number.isNaN(value)) {
      const position = filled[code] as number;
      (parts[code] as Float64Array)[position] = value;
      filled[code] = position + 1;
    }
  }
  for (const part of parts) {
    part.sort();
  }
  return parts;
}

/**
 * The numbers of the rows that pass a test, sorted, without missing ones
 *
 * @param numbers - The numbers
 * @param keep - Whether a row is kept, by its index
 * @returns The sorted numbers
 */
function numbersWhere(
  numbers: Float64Array,
  keep: (row: number) => boolean,
): Float64Array {
  const kept: number[] = [];
  for (let i = 0; i < numbers.length; i++) {
    const value = numbers[i] as number;
    if (!Number.isNaN(value) && keep(i)) {
      kept.push(value);
    }
  }
  return Float64Array.from(kept).sort();
}

/**
 * Finds a group of a column for a data tool
 *
 * @param grouped - The groups of the column
 * @param wanted - The group the model gave
 * @param column - The column, for the message
 * @returns The index of the group, or an error message for the model
 */
function analysisGroup(
  grouped: GroupedRows,
  wanted: unknown,
  column: string,
): number | { error: string } {
  const group = findGroup(grouped.groups, wanted);
  if (group === undefined) {
    const shown = grouped.groups.slice(0, maxGroups);
    const more =
      grouped.groups.length > shown.length
        ? `, ... (${grouped.groups.length} groups)`
        : "";
    return {
      error: `Error: ${column} has no group ${JSON.stringify(wanted)}. Its groups: ${shown.join(", ")}${more}.`,
    };
  }
  return grouped.groups.indexOf(group);
}

/**
 * How many groups or genes to list
 *
 * @param limit - The limit the model gave, if any
 * @returns The limit, from 1 to {@link maxListed}
 */
function listLimit(limit: unknown): number {
  return typeof limit === "number" && Number.isFinite(limit)
    ? Math.min(maxListed, Math.max(1, Math.round(limit)))
    : defaultListed;
}

/**
 * Lists the number of items of each group, the largest first (or the
 * smallest, if asked)
 *
 * @param title - What is counted, e.g. "obs/leiden of cell_labels"
 * @param counts - The count of each group
 * @param missing - How many items have no value
 * @param limit - How many groups to list
 * @param lowest - Whether to list the smallest first
 * @returns The text
 */
function describeCounts(
  title: string,
  counts: Map<string, number>,
  missing: number,
  limit: number,
  lowest: boolean,
): string {
  const total = [...counts.values()].reduce((sum, n) => sum + n, 0);
  const sorted = [...counts].sort((a, b) =>
    lowest ? a[1] - b[1] : b[1] - a[1],
  );
  const lines = [
    `${title}: ${count(total)} items in ${count(counts.size)} groups` +
      `${missing > 0 ? ` (and ${count(missing)} without a value)` : ""}, ${lowest ? "smallest" : "largest"} first:`,
    ...sorted
      .slice(0, limit)
      .map(([group, n]) => `- ${group}: ${count(n)} (${percent(n / total)})`),
  ];
  if (sorted.length > limit) {
    lines.push(`(+ ${count(sorted.length - limit)} more groups)`);
  }
  return lines.join("\n");
}

/**
 * column_stats for a column of a table too big to read: only what the table
 * itself can tell, its range or its counts
 *
 * @param title - What is described, e.g. "feature_name of transcripts"
 * @param found - The column
 * @param limit - How many groups to list
 * @param lowest - Whether to list the smallest first
 * @param options - Optional abort signal
 * @returns The result for the model
 */
async function bigColumnStats(
  title: string,
  found: FoundColumn,
  limit: number,
  lowest: boolean,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const { data, column, tableId } = found;
  const rows = data.getSize();
  const intro = `The table ${tableId} has ${count(rows)} rows, too many to read here`;
  if (data.loadValueRange !== undefined) {
    const range = await data.loadValueRange(column, options);
    if (range !== undefined) {
      return `${title}: ${intro}; the table only tells the range: min ${short(range[0])}, max ${short(range[1])}.`;
    }
  }
  if (data.loadUniqueValueCounts !== undefined) {
    const counts = new Map<string, number>();
    for (const [value, n] of await data.loadUniqueValueCounts<unknown>(
      column,
      options,
    )) {
      const group = String(value);
      counts.set(group, (counts.get(group) ?? 0) + n);
    }
    return `${intro}; the table counted its values itself.\n${describeCounts(title, counts, 0, limit, lowest)}`;
  }
  return tooBigError(tableId, rows);
}

/** The ways column_stats sorts groups */
type StatsSort = "count" | "mean" | "median" | "percent_above_0";

/**
 * The tool column_stats: counts the groups of a column, or summarizes its
 * numbers, also for each group of another column
 */
async function columnStats(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const object = analysisObject(context.projectStore.getState(), args.id);
  if ("error" in object) {
    return object.error;
  }
  if (typeof args.column !== "string" || args.column.trim() === "") {
    return 'Error: give column, e.g. "obs/leiden" or "EPCAM".';
  }
  const withBy = typeof args.by === "string" && args.by.trim() !== "";
  const found = await findColumns(
    context,
    object,
    withBy ? [args.column, args.by] : [args.column],
    options,
  );
  if ("error" in found) {
    return found.error;
  }
  const [column, by] = found as [FoundColumn, FoundColumn | undefined];
  const limit = listLimit(args.limit);
  const lowest = args.order === "lowest";
  const title = `${column.column} of ${object.name}`;
  const big = found.find(({ data }) => data.getSize() > maxRowsToRead);
  if (big !== undefined) {
    return by === undefined
      ? bigColumnStats(title, column, limit, lowest, options)
      : tooBigError(big.tableId, big.data.getSize());
  }
  const [values, byValues] = await loadAligned(found, options);
  const numbers = numbersOf(values as TypedArrayOrArray<unknown>);
  if (by === undefined || byValues === undefined) {
    const grouped =
      numbers === undefined ||
      new Set(numbers).size <= maxGroups + (numbers.some(Number.isNaN) ? 1 : 0)
        ? groupRows(values as TypedArrayOrArray<unknown>)
        : undefined;
    if (grouped !== undefined) {
      const counts = new Map(grouped.groups.map((group) => [group, 0]));
      let missing = 0;
      for (const code of grouped.codes) {
        if (code < 0) {
          missing++;
        } else {
          const group = grouped.groups[code] as string;
          counts.set(group, (counts.get(group) ?? 0) + 1);
        }
      }
      const text = describeCounts(title, counts, missing, limit, lowest);
      return numbers === undefined
        ? text
        : `${text}\nAs numbers: ${describeSummary(summarize(numbers))}.`;
    }
    if (numbers === undefined) {
      return `Error: ${title} has more than ${count(maxCategories)} different texts (e.g. names of single items): it cannot be counted by group.`;
    }
    const summary = summarize(numbers);
    const missing = numbers.length - summary.count;
    return (
      `${title}: ${count(summary.count)} items with a value` +
      `${missing > 0 ? ` (${count(missing)} without)` : ""}: ${describeSummary(summary)}.`
    );
  }
  const grouped = groupRows(byValues);
  if (grouped === undefined) {
    return `Error: ${by.column} has more than ${count(maxCategories)} groups: use a column with fewer groups as by.`;
  }
  const byTitle = `${title} by ${by.column} (${grouped.groups.length} groups)`;
  if (numbers === undefined) {
    // a column of groups within the groups of by, e.g. cell types per cluster
    const inner = groupRows(values as TypedArrayOrArray<unknown>);
    if (inner === undefined) {
      return `Error: ${title} has more than ${count(maxCategories)} different texts: it cannot be counted by group.`;
    }
    const table = grouped.groups.map(() => new Int32Array(inner.groups.length));
    const sizes = new Int32Array(grouped.groups.length);
    grouped.codes.forEach((code, i) => {
      const innerCode = inner.codes[i] as number;
      if (code >= 0 && innerCode >= 0) {
        const row = table[code] as Int32Array;
        row[innerCode] = (row[innerCode] as number) + 1;
        sizes[code] = (sizes[code] as number) + 1;
      }
    });
    const order = grouped.groups
      .map((_, k) => k)
      .sort((a, b) =>
        lowest
          ? (sizes[a] as number) - (sizes[b] as number)
          : (sizes[b] as number) - (sizes[a] as number),
      );
    const lines = [
      `${column.column} within each group of ${by.column} in ${object.name} (${grouped.groups.length} groups), ${lowest ? "smallest" : "largest"} groups first:`,
      ...order.slice(0, limit).map((k) => {
        const size = sizes[k] as number;
        const row = table[k] as Int32Array;
        const main = inner.groups
          .map((group, j) => [group, row[j] as number] as const)
          .filter(([, n]) => n > 0)
          .sort((a, b) => b[1] - a[1]);
        const shown = main
          .slice(0, 3)
          .map(([group, n]) => `${group} ${percent(n / size)}`);
        const more = main.length > 3 ? ` (+ ${main.length - 3} more)` : "";
        return `- ${grouped.groups[k]} (${count(size)} items): ${shown.join(", ") || "no values"}${more}`;
      }),
    ];
    if (order.length > limit) {
      lines.push(`(+ ${order.length - limit} more groups)`);
    }
    return lines.join("\n");
  }
  const parts = splitByGroup(numbers, grouped.codes, grouped.groups.length);
  const summaries = parts.map((part) => summarizeSorted(part));
  const sort: StatsSort = isOneOf(args.sort, [
    "count",
    "mean",
    "median",
    "percent_above_0",
  ])
    ? (args.sort as StatsSort)
    : "mean";
  const key = (summary: Summary): number =>
    sort === "percent_above_0" ? summary.positive : summary[sort];
  const order = summaries
    .map((_, k) => k)
    .filter((k) => (summaries[k] as Summary).count > 0)
    .sort((a, b) => {
      const difference =
        key(summaries[a] as Summary) - key(summaries[b] as Summary);
      return lowest ? difference : -difference;
    });
  const lines = [
    `${byTitle}, sorted by ${sort.replaceAll("_", " ")}, ${lowest ? "lowest" : "highest"} first:`,
    ...order.slice(0, limit).map((k) => {
      const summary = summaries[k] as Summary;
      return `- ${grouped.groups[k]}: ${describeSummary(summary)}, ${count(summary.count)} items`;
    }),
  ];
  if (order.length > limit) {
    lines.push(`(+ ${order.length - limit} more groups)`);
  }
  const all = summarize(numbers);
  lines.push(`All: ${describeSummary(all)}, ${count(all.count)} items.`);
  return lines.join("\n");
}

/**
 * Reads a column of numbers and a column of groups for compare_groups and
 * marker_genes, and finds the group and the reference
 *
 * @returns The groups and the indices of the group and of the reference
 * (`undefined` for all the other items), or an error message
 */
function pickGroups(
  byValues: TypedArrayOrArray<unknown>,
  by: FoundColumn,
  args: ToolArguments,
):
  | { grouped: GroupedRows; group: number; reference?: number }
  | { error: string } {
  const grouped = groupRows(byValues);
  if (grouped === undefined) {
    return {
      error: `Error: ${by.column} has more than ${count(maxCategories)} groups: use a column of groups as by.`,
    };
  }
  const group = analysisGroup(grouped, args.group, by.column);
  if (typeof group !== "number") {
    return group;
  }
  if (args.reference === undefined || args.reference === "") {
    return { grouped, group };
  }
  const reference = analysisGroup(grouped, args.reference, by.column);
  if (typeof reference !== "number") {
    return reference;
  }
  if (reference === group) {
    return {
      error:
        "Error: group and reference are the same group: give two different groups.",
    };
  }
  return { grouped, group, reference };
}

/**
 * The tool compare_groups: compares the numbers of a group with those of
 * another group, or of all the other items
 */
async function compareGroups(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const object = analysisObject(context.projectStore.getState(), args.id);
  if ("error" in object) {
    return object.error;
  }
  if (typeof args.column !== "string" || args.column.trim() === "") {
    return 'Error: give column, e.g. "obs/cell_area" or "EPCAM".';
  }
  const found = await findColumns(
    context,
    object,
    [args.column, args.by],
    options,
  );
  if ("error" in found) {
    return found.error;
  }
  const [column, by] = found as [FoundColumn, FoundColumn];
  const big = found.find(({ data }) => data.getSize() > maxRowsToRead);
  if (big !== undefined) {
    return tooBigError(big.tableId, big.data.getSize());
  }
  const [values, byValues] = (await loadAligned(found, options)) as [
    TypedArrayOrArray<unknown>,
    TypedArrayOrArray<unknown>,
  ];
  const numbers = numbersOf(values);
  if (numbers === undefined) {
    return `Error: ${column.column} does not hold numbers. To compare groups of it, use column_stats with by.`;
  }
  const picked = pickGroups(byValues, by, args);
  if ("error" in picked) {
    return picked.error;
  }
  const { grouped, group, reference } = picked;
  const { codes } = grouped;
  const xs = numbersWhere(numbers, (i) => codes[i] === group);
  const ys = numbersWhere(numbers, (i) =>
    reference === undefined ? codes[i] !== group : codes[i] === reference,
  );
  const name = `"${grouped.groups[group]}"`;
  const other =
    reference === undefined
      ? "the other items"
      : `"${grouped.groups[reference]}"`;
  if (xs.length === 0 || ys.length === 0) {
    return `Error: ${xs.length === 0 ? name : other} has no values of ${column.column}.`;
  }
  const a = summarizeSorted(xs);
  const b = summarizeSorted(ys);
  const test = mannWhitneySorted(xs, ys);
  const signed = (value: number) => `${value > 0 ? "+" : ""}${short(value)}`;
  return [
    `${column.column} of ${object.name}, group ${name} vs ${other} of ${by.column}:`,
    `- ${name}: ${count(a.count)} items, ${describeSummary(a)}`,
    `- ${other}: ${count(b.count)} items, ${describeSummary(b)}`,
    `Difference ${name} − ${other}: mean ${signed(a.mean - b.mean)}, median ${signed(a.median - b.median)}. ` +
      `An item of ${name} is higher than one of ${other} in ${percent(test.auc)} of the pairs (ties count half; 50% = no difference).`,
    `Wilcoxon rank-sum test (Mann-Whitney U, ties corrected): U = ${short(test.u)}, p = ${formatP(test.p)}.`,
  ].join("\n");
}

/**
 * The tool correlation: the Spearman correlation of two columns, over all
 * the items or within one group
 */
async function correlation(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const object = analysisObject(context.projectStore.getState(), args.id);
  if ("error" in object) {
    return object.error;
  }
  const names = [args.column_a, args.column_b];
  if (names.some((name) => typeof name !== "string" || name.trim() === "")) {
    return 'Error: give column_a and column_b, e.g. "EPCAM" and "KRT7".';
  }
  const inGroup = args.group !== undefined && args.group !== "";
  const found = await findColumns(
    context,
    object,
    inGroup ? [...names, args.by] : names,
    options,
  );
  if ("error" in found) {
    return found.error;
  }
  const big = found.find(({ data }) => data.getSize() > maxRowsToRead);
  if (big !== undefined) {
    return tooBigError(big.tableId, big.data.getSize());
  }
  const [a, b, by] = found as [FoundColumn, FoundColumn, FoundColumn?];
  const [valuesA, valuesB, byValues] = await loadAligned(found, options);
  const x = numbersOf(valuesA as TypedArrayOrArray<unknown>);
  const y = numbersOf(valuesB as TypedArrayOrArray<unknown>);
  if (x === undefined || y === undefined) {
    return `Error: ${x === undefined ? a.column : b.column} does not hold numbers: a correlation needs numbers or genes.`;
  }
  let where = "";
  if (by !== undefined && byValues !== undefined) {
    const grouped = groupRows(byValues);
    if (grouped === undefined) {
      return `Error: ${by.column} has more than ${count(maxCategories)} groups.`;
    }
    const group = analysisGroup(grouped, args.group, by.column);
    if (typeof group !== "number") {
      return group.error;
    }
    grouped.codes.forEach((code, i) => {
      if (code !== group) {
        x[i] = NaN;
      }
    });
    where = ` within group "${grouped.groups[group]}" of ${by.column}`;
  }
  const result = spearman(x, y);
  const title = `Spearman correlation of ${a.column} and ${b.column} in ${object.name}${where}`;
  if (Number.isNaN(result.rho)) {
    return `${title}: not available over ${count(result.n)} items, since ${result.n < 3 ? "there are fewer than 3" : "one of the columns is constant there"}.`;
  }
  return `${title}: rho ${short(result.rho)} over ${count(result.n)} items, p = ${formatP(result.p)}.`;
}

/** The results of marker_genes for one gene */
type GeneResult = {
  gene: string;
  z: number;
  p: number;
  meanIn: number;
  meanOut: number;
  /** The fractions of the items that express the gene (above 0) */
  fractionIn: number;
  fractionOut: number;
};

/** The results of marker_genes for one group, for all the genes */
type MarkerResults = {
  genes: GeneResult[];
  /** Benjamini-Hochberg adjusted p-values, in the order of the genes */
  adjusted: Float64Array;
  /** Whether the values look log1p-transformed (not whole numbers) */
  logged: boolean;
  inCount: number;
  outCount: number;
  /** How long the reading and the tests took, in seconds */
  seconds: number;
  /** When the results were computed */
  at: Date;
};

/**
 * The results of marker_genes, by table and by question, so that a later
 * question about the same group is answered at once; a table loaded again is
 * a new object, which starts with no results
 */
const markerCache = new WeakMap<TableData, Map<string, MarkerResults>>();

/**
 * Finds the matrix of genes of labels or points, e.g. X
 *
 * @param context - What the tool works with
 * @param object - The labels or points
 * @param wanted - The matrix the model gave, if any; X without it
 * @param options - Optional abort signal
 * @returns The table, the matrix and the columns of its genes, or an error
 */
async function findMatrix(
  context: ToolContext,
  object: ItemsObject,
  wanted: unknown,
  options?: { signal?: AbortSignal },
): Promise<
  | { tableId: string; data: TableData; matrix: string; columns: string[] }
  | { error: string }
> {
  const name = typeof wanted === "string" && wanted !== "" ? wanted : "X";
  const matrices: string[] = [];
  for (const tableId of objectTables(object, context.tables)) {
    const ref = context.dataStore.getState().tableDataRefs.get(tableId);
    if (ref === undefined) {
      continue;
    }
    const data = await ref.promise;
    for (const column of await listColumns(data, options)) {
      if (!column.matrix || !geneMatrix.test(column.name)) {
        continue;
      }
      matrices.push(column.name);
      if (column.name.toLowerCase() === name.toLowerCase()) {
        const columns = await matrixColumns(data, column.name, options);
        return { tableId, data, matrix: column.name, columns };
      }
    }
  }
  return {
    error:
      matrices.length === 0
        ? `Error: ${object.name} has no matrix of genes in its tables.`
        : `Error: ${object.name} has no matrix "${name}"; it has ${list(matrices)}.`,
  };
}

/**
 * The tool marker_genes: tests every gene of a matrix between a group and
 * the other items (or another group), like scanpy's rank_genes_groups with
 * the Wilcoxon test; shows its progress, can be stopped, and keeps its
 * results for later questions
 */
async function markerGenes(
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
): Promise<string> {
  const { signal } = options ?? {};
  const object = analysisObject(context.projectStore.getState(), args.id);
  if ("error" in object) {
    return object.error;
  }
  const matrix = await findMatrix(context, object, args.matrix, options);
  if ("error" in matrix) {
    return matrix.error;
  }
  const { data } = matrix;
  const found = await findColumns(context, object, [args.by], options);
  if ("error" in found) {
    return found.error;
  }
  const by = found[0] as FoundColumn;
  for (const table of [matrix, by]) {
    if (table.data.getSize() > maxRowsToRead) {
      return tooBigError(table.tableId, table.data.getSize());
    }
  }
  // the groups in the rows of the matrix
  const byValues = await by.data.loadValues<unknown>(by.column, options);
  const groupValues =
    by.data === data
      ? byValues
      : await alignRows(data, by.data, byValues, options);
  const picked = pickGroups(groupValues, by, args);
  if ("error" in picked) {
    return picked.error;
  }
  const { grouped, group, reference } = picked;
  const inRows: number[] = [];
  const outRows: number[] = [];
  grouped.codes.forEach((code, i) => {
    if (code === group) {
      inRows.push(i);
    } else if (reference === undefined || code === reference) {
      outRows.push(i);
    }
  });
  const name = `"${grouped.groups[group]}"`;
  const other =
    reference === undefined
      ? "all the other items"
      : `"${grouped.groups[reference]}"`;
  if (inRows.length === 0 || outRows.length === 0) {
    return `Error: ${inRows.length === 0 ? name : other} has no items.`;
  }
  const key = JSON.stringify([
    matrix.matrix,
    by.tableId,
    by.column,
    grouped.groups[group],
    reference === undefined ? null : grouped.groups[reference],
  ]);
  const cached = markerCache.get(data)?.get(key);
  const results =
    cached ??
    (await testGenes(matrix.columns, data, inRows, outRows, context, signal));
  if (cached === undefined) {
    const tableCache =
      markerCache.get(data) ?? new Map<string, MarkerResults>();
    tableCache.set(key, results);
    markerCache.set(data, tableCache);
  }
  const down = args.direction === "down";
  const order = results.genes
    .map((_, k) => k)
    .sort((a, b) => {
      const difference =
        (results.genes[a] as GeneResult).z - (results.genes[b] as GeneResult).z;
      return down ? difference : -difference;
    });
  const significant = order.filter((k) => {
    const gene = results.genes[k] as GeneResult;
    return (
      (down ? gene.z < 0 : gene.z > 0) && (results.adjusted[k] as number) < 0.05
    );
  }).length;
  const prefix = `${matrix.matrix}[`;
  const geneName = (column: string) =>
    column.startsWith(prefix) ? column.slice(prefix.length, -1) : column;
  const limit = listLimit(args.limit);
  const lines = [
    `Marker genes of group ${name} of ${by.column} in ${object.name}: ${count(results.inCount)} items vs ${other} ` +
      `(${count(results.outCount)}), over ${count(results.genes.length)} genes of ${matrix.matrix}.`,
    "Wilcoxon rank-sum test (Mann-Whitney U, ties corrected), p-values adjusted with Benjamini-Hochberg over all the genes. " +
      (results.logged
        ? "The values look log1p-transformed: log2 fold change of expm1 of the means, as scanpy."
        : "The values are whole numbers (counts): log2 fold change of the means."),
    `${down ? "Lower" : "Higher"} in the group, by z-score as scanpy sorts:`,
    ...order.slice(0, limit).map((k, rank) => {
      const gene = results.genes[k] as GeneResult;
      const fold = log2FoldChange(gene.meanIn, gene.meanOut, results.logged);
      return (
        `${rank + 1}. ${geneName(gene.gene)}: log2FC ${short(fold)}, ` +
        `in ${percent(gene.fractionIn)} of the group vs ${percent(gene.fractionOut)} of the others, ` +
        `adj. p ${formatP(results.adjusted[k] as number)}`
      );
    }),
    `${count(significant)} of ${count(results.genes.length)} genes are ${down ? "lower" : "higher"} in the group with adj. p < 0.05.`,
    cached === undefined
      ? `Took ${short(results.seconds)} s to read and test the genes.`
      : `From the results kept since ${results.at.toLocaleTimeString("en-GB")} (they took ${short(results.seconds)} s).`,
  ];
  return lines.join("\n");
}

/**
 * Tests every gene of a matrix between the rows of a group and the others,
 * one gene at a time, so that only one column is in memory; tells its
 * progress and lets the page breathe between the genes
 *
 * @param columns - The columns of the genes
 * @param data - The table of the matrix
 * @param inRows - The rows of the group
 * @param outRows - The rows it is compared with
 * @param context - What the tool works with, for the progress
 * @param signal - Stops the tests
 * @returns The results
 */
async function testGenes(
  columns: string[],
  data: TableData,
  inRows: number[],
  outRows: number[],
  context: ToolContext,
  signal?: AbortSignal,
): Promise<MarkerResults> {
  const started = performance.now();
  let lastBreath = started;
  const genes: GeneResult[] = [];
  const xs = new Float64Array(inRows.length);
  const ys = new Float64Array(outRows.length);
  let wholeNumbers = true;
  const fill = (
    target: Float64Array,
    rows: number[],
    values: ArrayLike<number>,
  ) => {
    let sum = 0;
    let positive = 0;
    rows.forEach((row, k) => {
      const value = Number(values[row]);
      target[k] = value;
      sum += value;
      if (value > 0) {
        positive++;
      }
      if (wholeNumbers && !Number.isInteger(value)) {
        wholeNumbers = false;
      }
    });
    target.sort();
    return { mean: sum / rows.length, fraction: positive / rows.length };
  };
  const tellProgress = (done: number) => {
    const seconds = Math.round((performance.now() - started) / 100) / 10;
    context.progress?.(
      `testing the genes: ${done} of ${columns.length} (${Math.round((100 * done) / columns.length)}%), ${seconds} s`,
    );
  };
  tellProgress(0);
  for (const [index, column] of columns.entries()) {
    signal?.throwIfAborted();
    const values = (await data.loadValues<number>(column, {
      signal,
    })) as ArrayLike<number>;
    const inside = fill(xs, inRows, values);
    const outside = fill(ys, outRows, values);
    const test = mannWhitneySorted(xs, ys);
    genes.push({
      gene: column,
      z: test.z,
      p: test.p,
      meanIn: inside.mean,
      meanOut: outside.mean,
      fractionIn: inside.fraction,
      fractionOut: outside.fraction,
    });
    // the progress is shown, and Stop can be clicked, between the genes
    if (performance.now() - lastBreath > 100) {
      tellProgress(index + 1);
      await new Promise((resolve) => setTimeout(resolve, 0));
      lastBreath = performance.now();
    }
  }
  return {
    genes,
    adjusted: benjaminiHochberg(genes.map((gene) => gene.p)),
    logged: !wholeNumbers,
    inCount: inRows.length,
    outCount: outRows.length,
    seconds: (performance.now() - started) / 1000,
    at: new Date(),
  };
}

// --- Project summary ---

/**
 * Names a table column, e.g. 'column "cell_type" of table tables/table'
 *
 * @param ref - The column
 * @returns The text
 */
function describeColumnRef(ref: TableColumnRef): string {
  return `column "${ref.column}"${ref.table === undefined ? "" : ` of table ${ref.table}`}`;
}

/**
 * Describes a color map with its first colors, e.g. 'color map "Chat colors –
 * cell_labels": 3 red, others light grey'
 *
 * @param map - The color map
 * @returns The text
 */
function describeColorMap(map: GroupValueMap<Color>): string {
  const groups = Object.entries(map.values);
  const colors = groups
    .slice(0, 10)
    .map(([group, color]) => `${group} ${colorName(color)}`);
  if (groups.length > 10) {
    colors.push(`... (${groups.length} groups)`);
  }
  if (map.default !== undefined) {
    colors.push(`others ${colorName(map.default)}`);
  }
  return `color map "${map.name}": ${colors.join(", ") || "no colors yet"}`;
}

/**
 * Describes a color setting in a few words, e.g. "fixed red", 'by groups of
 * column "cell_type", palette distinct' or 'from the numbers in column "area",
 * palette viridis, range 0 to 500'
 *
 * @param config - The color setting
 * @param colorMaps - The project's color maps
 * @returns The text
 */
function describeColor(
  config: ColorConfig,
  colorMaps: GroupValueMap<Color>[],
): string {
  const source = getActiveConfigSource(config);
  if (source === "constant" && isConstantConfig<Color>(config)) {
    return `fixed ${colorName(config.constant.value)}`;
  }
  if (
    source === "from" &&
    isFromConfig<{ range?: [number, number]; palette: string }>(config)
  ) {
    // without a range, the lowest value gets the first color and the highest the last
    const range =
      config.from.range === undefined
        ? "range from the lowest to the highest value"
        : `range ${config.from.range[0]} to ${config.from.range[1]}`;
    return `from the numbers in ${describeColumnRef(config.from)}, palette ${config.from.palette}, ${range}`;
  }
  if (
    source === "groupBy" &&
    isGroupByConfig<false, { palette?: string }>(config)
  ) {
    const { map, palette } = config.groupBy;
    // what TissUUmaps does without a map or palette
    let how = "no palette or color map, so all in one color";
    if (map !== undefined) {
      const colorMap = colorMaps.find((m) => m.id === map);
      how = describeColorMap(colorMap ?? { id: map, name: map, values: {} });
    } else if (palette !== undefined) {
      how = `palette ${palette}`;
    }
    return `by groups of ${describeColumnRef(config.groupBy)}, ${how}`;
  }
  if (source === "random" && isRandomConfig<{ palette: string }>(config)) {
    return `random colors, palette ${config.random.palette}`;
  }
  return "default colors";
}

/**
 * Describes which items of labels or points are shown
 *
 * @param config - The visibility setting
 * @param visibilityMaps - The project's visibility maps
 * @returns The text, e.g. 'only the groups 1, 3 of column "leiden" are shown',
 * or `undefined` when all items are shown
 */
function describeFilter(
  config: VisibilityConfig,
  visibilityMaps: GroupValueMap<boolean>[],
): string | undefined {
  const source = getActiveConfigSource(config);
  if (source === "constant" && isConstantConfig<boolean>(config)) {
    return config.constant.value ? undefined : "all items hidden";
  }
  if (source === "from" && isFromConfig(config)) {
    return `only the items where ${describeColumnRef(config.from)} is not 0 are shown`;
  }
  if (!(source === "groupBy" && isGroupByConfig<true>(config))) {
    return undefined;
  }
  const map = visibilityMaps.find((m) => m.id === config.groupBy.map);
  if (map === undefined) {
    return `by groups of ${describeColumnRef(config.groupBy)}`;
  }
  const groups = Object.entries(map.values);
  const shown = groups.filter(([, visible]) => visible).map(([group]) => group);
  const hidden = groups
    .filter(([, visible]) => !visible)
    .map(([group]) => group);
  return map.default === false
    ? `only the groups ${list(shown)} of ${describeColumnRef(config.groupBy)} are shown (show_all_groups removes this filter)`
    : `the groups ${list(hidden)} of ${describeColumnRef(config.groupBy)} are hidden (show_all_groups removes this filter)`;
}

/**
 * Describes the base point size in a few words
 *
 * @param pointSize - The size setting
 * @returns The text
 */
function describePointSize(pointSize: SizeConfig): string {
  const source = getActiveConfigSource(pointSize);
  if (source === "constant" && isConstantConfig<number>(pointSize)) {
    return `${pointSize.constant.value}`;
  }
  const ref = columnRefOf(pointSize);
  if (ref === undefined) {
    return "default";
  }
  return source === "from"
    ? `from table column "${ref.column}"`
    : `for each group of table column "${ref.column}"`;
}

/**
 * Describes layers, images or shapes, one line each, e.g.
 * '- id: img1, name: "DAPI", visible: yes, opacity: 1'
 *
 * @param title - The section title
 * @param objects - The objects
 * @param colorMaps - The project's color maps
 * @param channelsOf - For images, the line about their channels, if any
 * @returns The text
 */
function describeObjects(
  title: string,
  objects: readonly (Layer | Image | Shapes)[],
  colorMaps: GroupValueMap<Color>[],
  channelsOf?: (image: Image) => string | undefined,
): string {
  if (objects.length === 0) {
    return `## ${title}\nnone`;
  }
  const lines = objects.map((object) => {
    let line = `- id: ${object.id}, name: "${object.name}"`;
    line += `, visible: ${object.visibility ? "yes" : "no"}`;
    line += `, opacity: ${round(object.opacity)}`;
    if ("shapeFillColor" in object) {
      line += `, fill color: ${describeColor(object.shapeFillColor, colorMaps)}`;
      line += `, outline color: ${describeColor(object.shapeStrokeColor, colorMaps)}`;
    }
    const table = annotatingTable(object);
    if (table !== undefined) {
      line += `, annotated by table: ${table}`;
    }
    const channels =
      channelsOf !== undefined && "dataSource" in object
        ? channelsOf(object as Image)
        : undefined;
    return channels === undefined ? line : `${line}\n${channels}`;
  });
  return `## ${title}\n${lines.join("\n")}`;
}

/**
 * Describes the channels of an image as the Images panel shows them, with
 * their indices, e.g. "  - 4 channels, composite (the visible channels
 * blended): 0 DAPI (visible, white), 1 ATP1A1 (hidden, red), ..."
 *
 * @param image - The image
 * @param data - The data of the image, if it is loaded
 * @returns The line, or `undefined` for an image without channels
 */
function describeChannels(
  image: Image,
  data: ImageData | undefined,
): string | undefined {
  const channels = shownChannels(image, data);
  if (channels.length === 0) {
    return undefined;
  }
  const mode = image.channelViewMode ?? ImageChannelViewMode.composite;
  const active = ImageUtils.getActiveChannel(image, channels.length);
  const how =
    mode === ImageChannelViewMode.composite
      ? "composite (the visible channels blended)"
      : `${mode} (only channel ${active}, ${channels[active]?.name ?? ""}, is shown)`;
  const shown = channels.slice(0, 30).map((channel, c) => {
    const settings = [channel.visibility ? "visible" : "hidden"];
    settings.push(colorName(channel.color));
    if (channel.opacity !== 1) {
      settings.push(`opacity ${round(channel.opacity)}`);
    }
    return `${c} ${channel.name} (${settings.join(", ")})`;
  });
  if (channels.length > 30) {
    shown.push("...");
  }
  return `  - ${channels.length} channel${channels.length === 1 ? "" : "s"}, ${how}: ${shown.join(", ")}`;
}

/**
 * The lines about the useful columns of one table of labels or points
 *
 * @param object - The labels or points
 * @param tableId - The table
 * @param info - What was found out about the table, if anything
 * @returns The lines
 */
function describeTableColumns(
  object: ItemsObject,
  tableId: string,
  info: TableInfo | undefined,
): string[] {
  const own = tableId === annotatingTable(object);
  const which = own
    ? `table ${tableId}`
    : `another table, ${tableId}, with rows for the same items`;
  if (info === undefined) {
    return [`  - columns: the table ${tableId} is not loaded yet`];
  }
  if (info.status === "reading") {
    return [`  - columns of the table ${tableId}: still being read`];
  }
  if (info.status === "error") {
    return [
      `  - columns: the table ${tableId} could not be read (${info.error})`,
    ];
  }
  const position = own ? positionColumnsOf(object) : [];
  const useful = info.useful.filter(
    (column) => !position.includes(column.name),
  );
  const other = info.other + info.useful.length - useful.length;
  const lines = [
    `  - columns you can use (${which}, ${info.rows} rows):${useful.length === 0 ? " none" : ""}`,
    ...useful.map((column) => `    - ${column.name} (${column.text})`),
  ];
  if (other > 0) {
    const why = info.tooBig ? ", not read because the table is very big" : "";
    lines.push(`  - (+ ${other} other column${other === 1 ? "" : "s"}${why})`);
  }
  return lines;
}

/**
 * The cluster columns labels or points can use, e.g. [obs/leiden, obs/kmeans]
 *
 * @param object - The labels or points
 * @param tables - What the panel knows about the tables
 * @returns The column names
 */
function clusterColumns(object: ItemsObject, tables: TableKnowledge): string[] {
  return objectTables(object, tables).flatMap((tableId) => {
    const info = tables.info.get(tableId);
    if (info?.status !== "ready") {
      return [];
    }
    return info.useful
      .filter(
        (column) =>
          column.groups !== undefined && clusterColumn.test(column.name),
      )
      .map((column) => column.name);
  });
}

/**
 * Describes labels or points, each with its colors, its filter and the columns
 * it can use, e.g.
 *
 * ```
 * - **cell_labels** (labels, hidden, opacity 1, id: labels/cell_labels@global)
 *   - color: by groups of column "leiden", palette distinct
 *   - columns you can use (table tables/table, 1000 rows):
 *     - leiden (16 groups: 0, 1, 2, ..., 15)
 *   - (+ 4 other columns)
 * ```
 *
 * @param title - The section title
 * @param kind - The section
 * @param project - The project
 * @param tables - What the panel knows about the tables
 * @returns The text
 */
function describeItemsObjects(
  title: string,
  kind: ItemsKind,
  project: ProjectStore,
  tables: TableKnowledge,
): string {
  const objects: readonly ItemsObject[] = project[kind];
  if (objects.length === 0) {
    return `## ${title}\nnone`;
  }
  const blocks = objects.map((object) => {
    let line = `- **${object.name}** (${kind}, ${object.visibility ? "visible" : "hidden"}, opacity ${round(object.opacity)}`;
    if ("pointSize" in object) {
      line += `, point size ${describePointSize(object.pointSize)}`;
      line += `, point size factor ${round(object.pointSizeFactor)}`;
    }
    line += `, id: ${object.id})`;
    const lines = [
      line,
      `  - color: ${describeColor(colorConfigOf(object), project.colorMaps)}`,
    ];
    const filter = describeFilter(
      visibilityConfigOf(object),
      project.visibilityMaps,
    );
    if (filter !== undefined) {
      lines.push(`  - filter: ${filter}`);
    }
    const others = sameItemsAs(project, kind, object);
    if (others.length > 0) {
      const names = others.map(
        (other) => `${other.name} (${other.visibility ? "visible" : "hidden"})`,
      );
      lines.push(`  - same items as: ${list(names)}, drawn in the same place`);
    }
    const tableIds = objectTables(object, tables);
    if (tableIds.length === 0) {
      lines.push("  - columns: none (no table is linked)");
    }
    for (const tableId of tableIds) {
      lines.push(
        ...describeTableColumns(object, tableId, tables.info.get(tableId)),
      );
    }
    // several cluster columns: the summary says which one "clusters" means
    const clusters = clusterColumns(object, tables);
    if (clusters.length > 1) {
      const colorColumn = groupByColumnOf(object)?.column;
      lines.push(
        colorColumn !== undefined && clusters.includes(colorColumn)
          ? `  - "clusters" means ${colorColumn} here: the colors show it now`
          : `  - cluster columns: ${list(clusters)}. If the user just says "clusters", do not call a tool: ask which one.`,
      );
    }
    return lines.join("\n");
  });
  return `## ${title}\n${blocks.join("\n")}`;
}

/**
 * What the model cannot change with the tools; the strong models are told
 * {@link actionLimits} instead
 */
const toolLimits =
  "You cannot change anything else (for example deleting or renaming, or the colors of\nimages and shapes).";

/** What the strong models cannot change, also with call_action */
const actionLimits = [
  "With call_action you can also change the other settings that its allowed actions",
  "change (e.g. the colors of image channels, the markers of points, transforms, the order",
  "of objects). You cannot delete or add objects, or change their data. Rename something",
  "(setName, or the name of an object) only when the user asks to rename it.",
].join("\n");

/**
 * The instructions for the model, before the project summary
 */
const instructions = [
  "You are an assistant inside TissUUmaps, a viewer for tissue images and spatial data.",
  "Always answer in the language the user writes in (e.g. Italian if the user writes Italian).",
  "Answer questions about the project that is open right now, using the project summary.",
  "You can change how images, labels, points and shapes are shown with these tools.",
  "Every tool takes either ids (a list of ids from the summary; put several objects in",
  'ONE call) or category ("images", "labels", "points" or "shapes") for all the objects',
  "of that section, never both.",
  '- When the user names objects (e.g. "the transcripts", "the cell labels", "the H&E image"),',
  "  use ids, even if that object is the only one of its section.",
  "- Use category only when the user names a whole section without saying which objects",
  '  ("labels", "the images", "all points"): do not ask which ones.',
  "- set_visibility: to show or hide objects without changing the others",
  '  (e.g. "show the transcripts", "hide all points").',
  '- show_only: ONLY when the user says "only", "just", "nothing else" or',
  '  "hide everything else": all the other objects are hidden.',
  '  "Hide everything" is show_only without ids.',
  "- set_opacity: to make objects more or less transparent. Opacity goes from 0",
  '  (invisible) to 1 (fully opaque): "50% transparent" is 0.5, "30% transparent" is 0.7,',
  '  "80% opaque" is 0.8.',
  "- set_point_size: to make the points of points objects bigger or smaller than they",
  "  are NOW. Pass change_percent: just copy the percentage the user said, with a plus sign",
  "  for bigger and a minus sign for smaller. Do NOT do any math and do NOT compute a final",
  '  size: the plugin does that. "10% smaller" is -10, "50% bigger" is 50,',
  '  "twice as big" is 100, "half the size" is -50, "a bit bigger" is 20, "a bit smaller" is -20.',
  '  If the amount is a number without a clear unit (e.g. "reduce by 1", "make it 3 bigger"),',
  "  do not guess and do not call the tool: ask the user what they mean",
  '  (for example "1% smaller" or "half the size").',
  "  Call it only once per request. It works for every points object, also when the size",
  "  comes from a table column. Only for points.",
  '- color_by_column: to color labels or points by a column (see "columns you can use",',
  "  also those of another table): groups get a color each, numbers a color scale. For a gene,",
  '  pass its name as column (e.g. "EPCAM"). Pass palette and range only if the user says them.',
  '- set_group_colors: to give chosen colors to some groups ("cluster 3 in red").',
  "  Leave out others if the user does not name a color for the other groups.",
  '- set_color: one color for all the items ("color the cell circles green").',
  '- show_only_groups: to show only some groups ("only clusters 1 and 3",',
  '  "only the EPCAM transcripts"). show_all_groups removes that filter',
  '  ("show all the cells again").',
  "  In set_group_colors and show_only_groups, leave out column when the groups belong to",
  "  the column the object is colored by now.",
  '  For "clusters", follow the line about clusters under the object in the summary.',
  '  Filters on numbers (e.g. "area > 50") are not possible.',
  '  If a tool result says "Tell the user", tell the user that in your answer.',
  "set_opacity, set_point_size and the color and filter tools also make hidden objects visible, so you don't need",
  "to show them separately. To get a hidden object with a new opacity or size, change it",
  "first and hide it afterwards.",
  "Ask only when the words point to different sections at once or to no object",
  '(e.g. "the labels of the borders" could mean cell_labels, nucleus_labels,',
  "cell_boundaries or nucleus_boundaries): then do not call any tool, ask the user which",
  "one they mean, and list the possible objects by name.",
  'If the user names an object that is not in the summary (e.g. "transcripts" when no object',
  "is called like that), do not call any tool and do not use another object or category",
  "instead: say that this project has no such object, and name the objects it has.",
  "Use the tools only when the user asks to change how something is shown.",
  "When the user asks for a change, always call the tool, even if the summary says",
  "it is already like that: the tool result tells you what really happened.",
  'Never say that you changed something unless a tool result for it starts with "Done".',
  "The project summary always shows the project as it was when the user wrote the latest",
  "message, and the tool results after that message say what changed since then: check",
  "them before saying what is visible. Older messages, summaries and tool results in the",
  "conversation can be out of date, because the user may have clicked Undo or changed",
  "things by hand.",
  toolLimits,
  "You cannot zoom or move the view yet. If the user asks to zoom in or out, to move,",
  "center or fit the view, say that you can't do it yet, and how to do it by hand: the",
  "mouse wheel zooms, dragging the image moves it, and the button with the house at the",
  'top right of the viewer ("Reset view") goes back to the whole view.',
  "If a request has parts you can do and parts you can't, do the parts you can",
  "with the tools, then say clearly which parts you couldn't do",
  '(e.g. "I showed the points, but I can\'t rename them").',
  "If nothing in the request is possible, say so and don't call any tool.",
  "After using a tool, always answer with a short sentence in your own words,",
  "using object names; don't copy the tool result.",
  'If a tool result says "Nothing changed" or "Already", tell the user that those',
  "objects were already in that state, instead of saying that you changed them.",
  "If the answer is not in the summary, say that you don't know.",
  'Lines that start with "> " in a message of the user quote one of your earlier answers.',
].join("\n");

/**
 * The rules that keep a personality to the tone, after its description
 */
const personaRules = [
  "The personality only changes your tone. Every rule of your instructions still applies",
  "in full: tell the truth about what the tools did, ask when a request is ambiguous, keep",
  "answers short, and never be rude to the user. If earlier answers in this conversation",
  "used another name or tone, use this name and tone from now on. Speak of yourself as",
  '"I", never by your name in the third person.',
].join("\n");

/**
 * What the strong models are told besides the instructions: the types of the
 * app state, which never change between requests, so that the services keep
 * them in their prompt cache
 */
const wholeAppInstructions = [
  "# The app state of TissUUmaps",
  "Below are the TypeScript types of the app state of TissUUmaps, as its plugins see it:",
  "the project store (the open project and its actions), the model of the project, and",
  "the parts of the app store that only change the view. They are resolved into plain",
  "property lists, with the first sentence of each comment. Each turn comes with the",
  "current values of this state, as JSON, after the project summary.",
  "Use them to answer questions about the project: its objects and their settings, what",
  "can be changed, and how. To change something, use the specific tools when they can do",
  "it (showing, hiding, opacity, point size, colors and filters by a column), and",
  "call_action for every other change that its allowed actions can make, with the",
  "arguments of the action as in the types and the ids from the app state. For transform",
  "and channels, give only what changes. The summary lists the channels of each image,",
  "with their indices. To show only one channel of an image, set channelViewMode to",
  '"color" and activeChannel to its index; to show several, keep "composite" and set the',
  "visibility of every channel (false for those to hide). A call_action result lists each",
  "setting it changed, with the value before and after, and those that already were so:",
  "tell the user only what it lists as changed. If a call_action result is an error, read why,",
  "and fix the arguments or tell the user. When even call_action cannot make a change",
  "(e.g. deleting or adding objects), do not pretend to make it: say so, and say which",
  "setting would do it, so that the user can change it in the panels of TissUUmaps.",
  'When the user asks what can be changed (e.g. "which properties of the labels can I',
  'change?"), answer about TissUUmaps: list all the settings in the types, then say which',
  "of them you can change yourself with your tools.",
  "You know the settings, not the controls of the panels: name the panel (Images, Labels,",
  "Points, Shapes, or Project for the layers) and the setting, but do not describe buttons,",
  "fields or menus that you do not know.",
  "```ts",
  storeTypes,
  "```",
  "",
  "# Statistics",
  "column_stats, marker_genes, compare_groups and correlation compute statistics on the",
  "columns of labels and points (e.g. the cells), in the browser. They change nothing: to",
  'color or filter by a result (e.g. "color that cluster in red"), use the other tools',
  "afterwards. When you report a result:",
  "- Give the size of the effect first (log2 fold change, % of cells expressing, difference",
  "  of means or medians, rho), then the p-value.",
  "- With thousands of cells almost every difference is significant: judge by the size of",
  "  the effect, not by the p-value, and say so when the effect is small.",
  "- When genes are tested between clusters computed from the same expression data (e.g.",
  '  leiden or kmeans clusters), warn about "double dipping": the clusters were made to',
  "  differ, so the p-values are too optimistic.",
  "- Say that the cells are not independent (neighbors in the same tissue, one sample), so",
  "  the results are exploratory, not for a publication.",
  "- Use only the numbers of the results. If a result says that a table is too big, tell the",
  "  user.",
].join("\n");

/**
 * The app state as JSON, for the strong models: the project store and the
 * parts of the app store that only change the view, without the actions
 *
 * The OpenSeadragon options are left out: they are long and say nothing about
 * the project.
 *
 * @param project - The project store
 * @param app - The app store
 * @returns The JSON text
 */
function stateJson(project: ProjectStore, app: AppStore): string {
  const view = {
    interactionMode: app.interactionMode,
    imageChannelPreview: app.imageChannelPreview,
    highlightedItemGroup: app.highlightedItemGroup,
    activePanelId: app.activePanelId,
    expandedImageIds: app.expandedImageIds,
    expandedLabelsIds: app.expandedLabelsIds,
    expandedPointsIds: app.expandedPointsIds,
    expandedShapesIds: app.expandedShapesIds,
    expandedTableIds: app.expandedTableIds,
  };
  return JSON.stringify(
    { projectStore: project, appStore: view },
    (key, value: unknown) =>
      typeof value === "function" || key === "osOptions" ? undefined : value,
  );
}

/**
 * Builds what changes between the requests to the model: the personality and
 * a summary of the open project. The instructions, which never change, are
 * sent apart from it (see `providers.ts`).
 *
 * @param project - The project
 * @param dataStore - The data store, for the channels of the images
 * @param tables - What the panel knows about the tables
 * @param persona - The personality of the assistant
 * @returns The text
 */
function buildContext(
  project: ProjectStore,
  dataStore: DataStoreApi,
  tables: TableKnowledge,
  persona: Persona,
): string {
  return [
    "# Your personality",
    persona.tone,
    personaRules,
    "",
    `# Project "${project.name}"`,
    describeObjects("Layers", project.layers, project.colorMaps),
    describeObjects("Images", project.images, project.colorMaps, (image) =>
      describeChannels(image, loadedImageData(dataStore, image)),
    ),
    describeItemsObjects("Labels", "labels", project, tables),
    describeItemsObjects("Points", "points", project, tables),
    describeObjects("Shapes", project.shapes, project.colorMaps),
    "## Color palettes in TissUUmaps",
    `- for groups: ${groupPalettes.join(", ")}`,
    `- for numbers: ${numberPalettes.join(", ")}`,
  ].join("\n");
}

/**
 * Example requests for the open project, shown above the text box; each one
 * names an object of the project and only columns and groups of that
 * object's tables
 *
 * @param project - The project
 * @param tables - What the panel knows about the tables
 * @returns Up to four requests
 */
function exampleRequests(
  project: ProjectStore,
  tables: TableKnowledge,
): string[] {
  let clusters: string | undefined;
  let filter: string | undefined;
  let gene: string | undefined;
  const objects: ItemsObject[] = [...project.labels, ...project.points];
  for (const object of objects) {
    const groupBy = groupByColumnOf(object);
    for (const tableId of objectTables(object, tables)) {
      const info = tables.info.get(tableId);
      if (info?.status !== "ready") {
        continue;
      }
      for (const column of info.useful) {
        const short = column.name.replace(/^.*\//, "");
        if (
          clusters === undefined &&
          column.groups !== undefined &&
          clusterColumn.test(column.name)
        ) {
          clusters = `Color the ${object.name} by ${short}`;
        }
        // only with gene names: "Color ... by 0 expression" would make no sense
        const [firstGene] = column.genes ?? [];
        if (gene === undefined && firstGene !== undefined) {
          gene = `Color the ${object.name} by ${firstGene} expression`;
        }
        // many groups that give the colors now, e.g. a color for each gene of
        // the transcripts: only the groups of exactly that column
        const colorsFromHere =
          groupBy?.column === column.name &&
          (groupBy.table ?? annotatingTable(object)) === tableId;
        const [first, second] = column.groups ?? [];
        if (
          filter === undefined &&
          column.many === true &&
          colorsFromHere &&
          second !== undefined
        ) {
          filter = `Show only ${first} and ${second} in ${object.name}`;
        }
      }
    }
  }
  const [image] = project.images;
  return [
    clusters,
    filter,
    gene,
    "What is shown now?",
    image === undefined ? undefined : `Make the ${image.name} 50% transparent`,
    project.points.length > 0 ? "Hide all points" : undefined,
  ]
    .filter((example) => example !== undefined)
    .slice(0, 4);
}

// --- Simple Markdown for the model's answers ---
// The model's text is never used as HTML: the elements are created here and
// the text is put in them as plain text, so that a tag like <script> in an
// answer is shown as text and never run.

/** `code`, **bold** or *italic*; the * must touch the word, so "2 * 3" is not italic */
const inlineMarkdown = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/;

/**
 * Creates an element that contains the given text as plain text
 *
 * @param tag - The tag name
 * @param text - The text
 * @returns The element
 */
function styled(tag: string, text: string): HTMLElement {
  const element = document.createElement(tag);
  element.textContent = text;
  return element;
}

/**
 * Splits a line into plain text and `code`, **bold** and *italic* pieces;
 * underscores are left alone, so that IDs like cell_labels are not made italic
 *
 * @param line - The line
 * @returns The pieces, as text or elements
 */
function renderInline(line: string): (string | HTMLElement)[] {
  // split() with a group in the pattern keeps the matches, at odd positions
  return line.split(inlineMarkdown).map((part, index) => {
    if (index % 2 === 0) {
      return part;
    }
    if (part.startsWith("`")) {
      const code = styled("code", part.slice(1, -1));
      code.style.cssText =
        "font-family: monospace; background: var(--muted); padding: 0 4px; border-radius: 4px;";
      return code;
    }
    if (part.startsWith("**")) {
      return styled("b", part.slice(2, -2));
    }
    return styled("i", part.slice(1, -1));
  });
}

/**
 * Turns an answer of the model into elements: "- " or "* " lines become a
 * bulleted list (indented lines further to the right), "1. " lines a numbered
 * list, "#" lines a bold heading, and every other line a paragraph
 *
 * @param text - The answer
 * @returns The element holding the answer
 */
function renderMarkdown(text: string): HTMLElement {
  const box = document.createElement("div");
  let currentList: HTMLUListElement | HTMLOListElement | null = null;
  // models sometimes write "sample\_1" for "sample_1"
  for (const line of text.replaceAll("\\_", "_").split("\n")) {
    const bullet = /^(\s*)[-*]\s+(.*)/.exec(line);
    const numbered = /^\s*(\d+)[.)]\s+(.*)/.exec(line);
    const heading = /^#{1,6}\s+(.*)/.exec(line);
    if (heading !== null) {
      currentList = null;
      const title = styled("b", heading[1] ?? "");
      title.style.cssText = "display: block; margin-top: 8px;";
      box.append(title);
    } else if (bullet !== null || numbered !== null) {
      const tag = bullet !== null ? "ul" : "ol";
      if (currentList === null || currentList.tagName.toLowerCase() !== tag) {
        currentList = document.createElement(tag);
        // TissUUmaps' styles remove the list bullets
        currentList.style.cssText = `margin: 0; padding-left: 20px; list-style: ${tag === "ul" ? "disc" : "decimal"};`;
        if (currentList instanceof HTMLOListElement) {
          currentList.start = Number(numbered?.[1] ?? 1);
        }
        box.append(currentList);
      }
      const item = document.createElement("li");
      item.append(
        ...renderInline((bullet !== null ? bullet[2] : numbered?.[2]) ?? ""),
      );
      if (bullet !== null) {
        item.style.marginLeft = `${(bullet[1] ?? "").length * 8}px`;
      }
      currentList.append(item);
    } else {
      currentList = null;
      const paragraph = document.createElement("div");
      // keeps empty lines visible
      paragraph.style.minHeight = "1em";
      paragraph.append(...renderInline(line));
      box.append(paragraph);
    }
  }
  return box;
}

// --- Undo ---

/** Added to an undo line when the user had changed the value by hand */
const handNote = " (it had been changed by hand after the chat changed it)";

/**
 * Describes how a value changed, e.g. "he_image opacity 0.5 → 1" or
 * "transcripts hidden → visible"
 *
 * @param change - The change
 * @param from - The value before
 * @param to - The value after
 * @returns The text
 */
function describeValueChange(
  change: Change,
  from: unknown,
  to: unknown,
): string {
  if (change.type === "added") {
    return `${change.name} (the map added by the chat is deleted)`;
  }
  if (change.type === "moved") {
    return `${change.name} back at position ${Number(to) + 1}`;
  }
  if (change.kind === "app") {
    return change.property === "highlightedItemGroup"
      ? "the highlighted group"
      : "the expanded entries";
  }
  // the settings of colors and filters are too long to show
  if (change.property.endsWith("Color")) {
    return `${change.name} colors`;
  }
  if (change.property.endsWith("Visibility")) {
    return `${change.name} group filter`;
  }
  const format = (value: unknown): string => {
    if (change.property === "visibility") {
      return value === true ? "visible" : "hidden";
    }
    if (typeof value === "string") {
      return `"${value}"`;
    }
    return String(round(Number(value)));
  };
  const labels: Record<string, string> = {
    visibility: "",
    opacity: " opacity",
    pointSizeFactor: " point size factor",
  };
  const label = labels[change.property] ?? ` ${change.property}`;
  // e.g. a transform or the channels: only named
  if (typeof from === "object" || typeof to === "object") {
    return `${change.name}${label}`;
  }
  return `${change.name}${label} ${format(from)} → ${format(to)}`;
}

// --- User interface ---
// All the colors are TissUUmaps' theme variables, which change with the light
// and the dark theme, so that the panel stays readable with both.

/** The shared look of the menus, the text fields and the buttons */
const controlStyle =
  "padding: 4px 8px; border: 1px solid var(--input); border-radius: 6px;" +
  "background: var(--background); color: var(--foreground); font: inherit;";

/** The look of the small buttons, e.g. "Undo" and the suggestions */
const smallButtonStyle = `${controlStyle}padding: 2px 8px; font-size: 12px;`;

/** The look of small, quieter text, e.g. the thinking word and "Made by" */
const quietTextStyle = "color: var(--muted-foreground); font-size: 12px;";

/** The kinds of messages in the chat, each with its own look */
type MessageKind = "user" | "assistant" | "tool" | "note" | "error" | "info";

/** The look of each kind of message */
const messageStyles: Record<MessageKind, string> = {
  user: "background: var(--secondary); color: var(--secondary-foreground); border-color: var(--border);",
  assistant: "background: transparent; border-color: var(--border);",
  tool:
    "background: transparent; color: var(--muted-foreground); border-color: var(--border);" +
    "border-style: dashed; font-size: 12px;",
  note: "background: var(--accent); color: var(--accent-foreground); border-color: var(--ring);",
  error:
    "background: color-mix(in oklch, var(--destructive) 12%, transparent);" +
    "border-color: var(--destructive);",
  info: "background: transparent; border-color: var(--border);",
};

/** How long each word is shown while the model thinks, in milliseconds */
const thinkingWordTime = 2000;

/** The line at the bottom of the panel */
const credits = "Made by Alessia Giustolisi";

/**
 * Creates a button
 *
 * @param text - The button text
 * @param onClick - The click action
 * @param style - The look of the button
 * @returns The button
 */
function makeButton(
  text: string,
  onClick: () => void,
  style = controlStyle,
): HTMLButtonElement {
  const button = document.createElement("button");
  button.style.cssText = style;
  button.textContent = text;
  button.addEventListener("click", onClick);
  return button;
}

/**
 * Creates the avatar shown next to the assistant's name: the image of the
 * personality, or an empty round placeholder until there is one
 *
 * @param persona - The personality
 * @returns The element
 */
function makeAvatar(persona: Persona): HTMLElement {
  const style =
    "width: 20px; height: 20px; border-radius: 50%; flex: none; object-fit: cover;";
  if (persona.avatar !== undefined) {
    const image = document.createElement("img");
    image.src = persona.avatar;
    // the name is next to it
    image.alt = "";
    image.style.cssText = style;
    return image;
  }
  const placeholder = document.createElement("span");
  placeholder.setAttribute("aria-hidden", "true");
  placeholder.style.cssText = `${style}background: var(--muted); border: 1px solid var(--border);`;
  return placeholder;
}

/**
 * Picks a word to show while the model thinks, at random
 *
 * @param previous - The word shown until now, which is not picked again
 * @returns The word with its translation
 */
function randomThinkingWord(
  previous?: (typeof thinkingWords)[number],
): (typeof thinkingWords)[number] {
  const others = thinkingWords.filter((entry) => entry !== previous);
  return others[Math.floor(Math.random() * others.length)] as {
    word: string;
    translation: string;
  };
}

/**
 * Makes a dot blink, unless the user asked the system for less motion or the
 * browser cannot animate
 *
 * @param dot - The dot
 * @param index - Its position, so that the dots blink one after the other
 */
function blink(dot: HTMLElement, index: number): void {
  const lessMotion =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (lessMotion || typeof dot.animate !== "function") {
    return;
  }
  dot.animate([{ opacity: 0.25 }, { opacity: 1 }, { opacity: 0.25 }], {
    duration: 1200,
    delay: index * 200,
    iterations: Infinity,
  });
}

/**
 * Creates the indicator shown while the model thinks: three blinking dots and
 * an Italian word that changes every two seconds, in random order. With the
 * mouse over the word, or the keyboard focus on it, its English translation
 * is shown and the word stays.
 *
 * @returns The element, and a function that stops the changes
 */
function makeThinkingIndicator(): { element: HTMLElement; stop: () => void } {
  const element = document.createElement("div");
  element.style.cssText = `display: flex; align-items: center; gap: 6px; ${quietTextStyle}`;
  const dots = document.createElement("span");
  dots.setAttribute("aria-hidden", "true");
  for (let index = 0; index < 3; index++) {
    const dot = document.createElement("span");
    dot.textContent = "•";
    dot.style.margin = "0 1px";
    dots.append(dot);
    blink(dot, index);
  }
  const word = document.createElement("span");
  word.tabIndex = 0;
  word.style.cssText =
    "font-style: italic; cursor: help; border-bottom: 1px dotted var(--muted-foreground);";
  const translation = document.createElement("span");
  translation.setAttribute("role", "tooltip");
  translation.style.cssText =
    "position: absolute; left: calc(100% + 6px); top: 50%; transform: translateY(-50%);" +
    "display: none; z-index: 1; white-space: nowrap; padding: 2px 6px; font-size: 12px;" +
    "border: 1px solid var(--border); border-radius: 6px; background: var(--popover);" +
    "color: var(--popover-foreground); box-shadow: 0 1px 3px rgb(0 0 0 / 20%);";
  const wordBox = document.createElement("span");
  wordBox.style.position = "relative";
  wordBox.append(word, translation);
  element.append(dots, wordBox);

  let current = randomThinkingWord();
  let paused = false;
  function showWord(): void {
    word.textContent = current.word;
    translation.textContent = current.translation;
  }
  function pause(on: boolean): void {
    paused = on;
    translation.style.display = on ? "" : "none";
  }
  showWord();
  word.addEventListener("mouseenter", () => pause(true));
  word.addEventListener("mouseleave", () => pause(false));
  word.addEventListener("focus", () => pause(true));
  word.addEventListener("blur", () => pause(false));
  const timer = setInterval(() => {
    if (!paused) {
      current = randomThinkingWord(current);
      showWord();
    }
  }, thinkingWordTime);
  return { element, stop: () => clearInterval(timer) };
}

/**
 * Formats a tool call for the chat, e.g. "set_visibility(category: points,
 * visible: false)" or 'set_group_colors(colors: {"3":"red"})'
 *
 * @param name - The tool name
 * @param args - The arguments
 * @returns The text
 */
function describeCall(name: string, args: ToolArguments): string {
  const values = Object.entries(args).map(([key, value]) => {
    if (Array.isArray(value)) {
      // objects in full, so that two different calls never read the same
      const items = value.map((item: unknown) =>
        typeof item === "string" ? item : JSON.stringify(item),
      );
      return `${key}: [${items.join(", ")}]`;
    }
    if (typeof value === "object" && value !== null) {
      return `${key}: ${JSON.stringify(value)}`;
    }
    return `${key}: ${String(value)}`;
  });
  return `${name}(${values.join(", ")})`;
}

/**
 * Writes an amount in dollars, with enough digits for fractions of a cent
 *
 * @param amount - The amount
 * @returns E.g. "$0.0004" or "$0.012"
 */
function dollars(amount: number): string {
  return `$${amount.toFixed(amount < 0.01 ? 4 : 3)}`;
}

/**
 * Whether a service is reached through Ollama on the user's computer, which
 * needs no key but has to be running
 *
 * @param provider - The service
 * @returns Whether it is one of the Ollama services
 */
function usesOllama(provider: Provider): boolean {
  return !provider.needsKey;
}

/**
 * The message when OpenAI or Anthropic cannot be reached
 *
 * @param provider - The service
 * @returns The message
 */
function notReachable(provider: Provider): string {
  return `${provider.name} does not answer. Check the internet connection, then click "Check again".`;
}

/**
 * Mounts the chat panel
 *
 * @param container - The element to mount the panel into
 * @param stores - The stores of the application
 * @returns A callback that unmounts the panel
 */
function mountAiChat(
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
  settingsBox.append(serviceRow, keyRow, statusRow);

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
    history.append(message);
    history.scrollTop = history.scrollHeight;
    return body;
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
   * What a change is about, as it is now: the value of the property, the
   * added map, or the index of the moved entry; `undefined` if the entry is
   * not in the project any more
   */
  function currentValue(change: Change): { value: unknown } | undefined {
    const project = projectStore.getState();
    if (change.kind === "project" || change.kind === "app") {
      const state: object =
        change.kind === "project" ? project : appStore.getState();
      return { value: (state as Record<string, unknown>)[change.property] };
    }
    if (change.type === "moved") {
      const index = project[change.kind].findIndex(
        (entry) => entry.id === change.id,
      );
      return index === -1 ? undefined : { value: index };
    }
    const entry = findById(project, change.kind, change.id);
    if (entry === undefined) {
      return undefined;
    }
    return {
      value:
        change.type === "added"
          ? entry
          : (entry as unknown as Record<string, unknown>)[change.property],
    };
  }

  /** Puts back what one change changed, with the store actions */
  function takeBack(change: Change): void {
    const project = projectStore.getState();
    if (change.kind === "project") {
      setProjectSetting(project, change.property, change.before);
    } else if (change.kind === "app") {
      setView(appStore.getState(), change.property, change.before);
    } else if (change.type === "added" && isMapKind(change.kind)) {
      deleteMap(project, change.kind, change.id);
    } else if (change.type === "moved" && !isMapKind(change.kind)) {
      applyMove(project, change.kind, change.id, change.before as number);
    } else {
      applyUpdate(project, change.kind, change.id, {
        [change.property]: change.before,
      });
    }
  }

  /**
   * Puts back the values from before one request, from its last change to
   * its first, and returns one line per change for the chat
   */
  function undoRequest(changes: Change[]): string[] {
    const lines: string[] = [];
    for (const change of [...changes].reverse()) {
      const now = currentValue(change);
      if (now === undefined) {
        lines.push(
          `${change.name} is not in the project any more, so it was skipped`,
        );
        continue;
      }
      takeBack(change);
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
        touched.set(key, { change, from: currentValue(change)?.value });
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
      const to = currentValue(change)?.value;
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

  /**
   * The tools for a service: for the strong models, also call_action and the
   * data tools
   */
  function toolsFor(provider: Provider) {
    return [
      ...toolDefinitions,
      ...(provider.wholeApp
        ? [callActionDefinition, ...analysisDefinitions]
        : []),
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
    const offered = !wholeAppToolNames.includes(name) || provider.wholeApp;
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
        { projectStore, dataStore, appStore, tables, changes, progress },
        { signal: stopper.signal },
      );
    } catch (error) {
      signal.throwIfAborted();
      result = stopper.signal.aborted
        ? "Stopped by the user before the end: nothing was computed. Tell the user."
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
    try {
      startThinking();
      // the same for every step of this turn
      const context = await turnContext(provider);
      for (let step = 1; step <= maxSteps; step++) {
        // the answer is shown while it arrives, in a message made when the
        // first piece of text comes
        let setText: ((text: string) => void) | null = null;
        if (thinking === null) {
          startThinking();
        }
        let answer: ChatAnswer;
        try {
          answer = await askModel(
            provider,
            model,
            key,
            context,
            (textSoFar) => {
              if (setText === null) {
                stopThinking();
                setText = addMessage(persona.name, "assistant", true);
              }
              setText(textSoFar);
            },
          );
        } finally {
          stopThinking();
        }
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
            outcome = await runTool(provider, name, args, changes, shownCall);
          }
          if (outcome.result.startsWith("Error")) {
            failedCalls.push(callText);
          }
          show("Tool", `${shownCall}\n→ ${outcome.result}`);
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

/**
 * A chat panel that lets the user ask a model about the open project and
 * change how its objects are shown: show and hide objects, change their
 * opacity and point size, and color and filter labels and points by the
 * columns of their tables
 *
 * The model runs at OpenAI or Anthropic, with the user's own API key, or
 * through Ollama, on the user's computer or online. Each request sends the
 * model a summary of the project. The changes go through the project store,
 * and "Undo" takes them back.
 */
export const aiChatPlugin: Plugin = {
  id: "ai-chat",
  name: "AI chat",
  mount: mountAiChat,
};
