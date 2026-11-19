import type {
  AppStoreApi,
  DataStoreApi,
  Image,
  Labels,
  Points,
  ProjectStoreApi,
  Shapes,
} from "@tissuumaps/core";

import type { ChatPluginState } from "./chat-plugins";
import type { Skill } from "./skills";

/** The arguments of a tool call, as the model sends them */
export type ToolArguments = Record<string, unknown>;

/** The sections of the project that the tools work on */
export type ObjectKind = "images" | "labels" | "points" | "shapes";

/** The sections whose items can be colored and filtered by a table column */
export type ItemsKind = "labels" | "points";

/**
 * The lists of maps of the project: the color and filter tools write color
 * and visibility maps, call_action all of them
 */
export type MapKind =
  "markerMaps" | "sizeMaps" | "colorMaps" | "visibilityMaps" | "opacityMaps";

/** The lists of the project whose entries can be changed */
export type ListKind = ObjectKind | "layers" | MapKind;

/** An object of the project that the tools can change */
export type ProjectObject = Image | Labels | Points | Shapes;

/** An object whose items can be colored and filtered by a table column */
export type ItemsObject = Labels | Points;

/** An object that a tool works on, with its section */
export type Target = { kind: ObjectKind; object: ProjectObject };

/** An object whose items a color or filter tool works on, with its section */
export type ItemsTarget = { kind: ItemsKind; object: ItemsObject };

/** One change made by a tool, remembered so that "Undo" can take it back */
export type Change = {
  /**
   * Without it, one property changed; "added": a map was added, which Undo
   * deletes; "moved": an entry was moved to another index; "computed": code
   * saved a column into a table computed by the chat, whose data source was
   * before and after the source of a version (before: none for a new table,
   * which Undo deletes)
   */
  type?: "added" | "moved" | "computed";
  /**
   * The list of the changed entry; "project" for a setting of the project
   * itself, "app" for a setting of the view in the app store, "tables" for
   * a table computed by the chat
   */
  kind: ListKind | "project" | "app" | "tables";
  /** The ID of the entry, or "" for the project and the app */
  id: string;
  name: string;
  /**
   * The property, or "index" for a move, or "" for an added map, or the
   * column of a computed table
   */
  property: string;
  before: unknown;
  after: unknown;
};

/** What a tool works with, besides the arguments of the model */
export type ToolContext = {
  projectStore: ProjectStoreApi;
  dataStore: DataStoreApi;
  /** For call_action, which sets the view */
  appStore: AppStoreApi;
  tables: TableKnowledge;
  /** The changes of the current request, which the tool adds to */
  changes: Change[];
  /** The skills that are on, for load_skill, read_skill_file and run_script */
  skills: Skill[];
  /**
   * Shows the progress of a long tool in a row of the chat, with a button
   * that stops it (through the abort signal of the tool)
   */
  progress: (text: string) => void;
  /**
   * Runs a script of a skill in the sandbox, after asking the user for the
   * scripts of a skill of the project, and shows its code in the chat
   *
   * @returns The result for the model
   */
  runSkillScript: (
    skill: Skill,
    path: string,
    args: unknown,
    options?: RunOptions,
  ) => Promise<string>;
  /**
   * Runs code that the model wrote in the sandbox, and shows it in the chat
   *
   * @returns The result for the model
   */
  runCode: (
    title: string,
    code: string,
    language: "javascript" | "python",
    options?: RunOptions & { skill?: Skill },
  ) => Promise<string>;
  /**
   * Reads a skill from GitHub, shows the user what it holds, and adds it if
   * the user says yes
   *
   * @returns The result for the model
   */
  addGitHubSkill: (
    address: string,
    options?: { signal?: AbortSignal; progress?: (text: string) => void },
  ) => Promise<string>;
  /**
   * Shows the user new instructions for a skill, and uses them if the user
   * says yes
   *
   * @returns The result for the model
   */
  proposeSkillMd: (
    skill: Skill,
    skillMd: string,
    options?: { signal?: AbortSignal },
  ) => Promise<string>;
  /**
   * Keeps a table computed by the chat as a CSV file next to the project, after
   * asking the user, or downloads it when it cannot
   *
   * @returns The result for the model
   */
  keepComputedTable: (
    table: string,
    options?: { fileName?: string },
  ) => Promise<string>;
  /**
   * Shows the user a plugin that the model wrote, with what it uses and its
   * delicate parts, and runs it and opens its panel if the user says yes
   *
   * @param plugin - Its id (with "chat-"), name, description and code
   * @param options - The skill it is based on, if any; an optional abort
   * signal, which answers no
   * @returns The result for the model
   */
  /** The plugins made by the chat, for read_plugin and propose_plugin */
  chatPlugins: () => ChatPluginState[];
  proposePlugin: (
    plugin: { id: string; name: string; description: string; code: string },
    options?: { skill?: Skill; signal?: AbortSignal },
  ) => Promise<string>;
  /**
   * Gives the model the image of a chart of this chat, with the result of
   * the tool, for the current turn only
   *
   * @param chart - Its number, as the chat shows it; without it, the last
   * @returns The result for the model
   */
  lookAtChart: (chart?: number) => string;
};

/** How code runs in the sandbox */
export type RunOptions = {
  /** Stops the code */
  signal?: AbortSignal;
  /** Shows how far it is, in a row of the chat */
  progress?: (text: string) => void;
  /** The changes of the current request, for the columns the code saves */
  changes?: Change[];
};

/**
 * A tool the model can call
 *
 * It returns a short text for the model, starting with "Done", "Nothing
 * changed" or "Error". If something is wrong, it changes nothing.
 */
export type Tool = (
  args: ToolArguments,
  context: ToolContext,
  options?: { signal?: AbortSignal },
) => string | Promise<string>;

/** What was found out about one table */
export type TableInfo =
  | { status: "reading" }
  | { status: "error"; error: string }
  | {
      status: "ready";
      /** e.g. that the table was computed by the chat, and is temporary */
      note?: string;
      rows: number;
      tooBig: boolean;
      useful: ColumnInfo[];
      other: number;
      /**
       * The columns of the x and the y of each item, for spatial analyses,
       * e.g. ["obsm/spatial[0]", "obsm/spatial[1]"]
       */
      coordinates?: [string, string];
    };

/** A column useful for colors and filters */
export type ColumnInfo = {
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
export type TableKnowledge = {
  /** What was found out about each table, by table ID */
  info: Map<string, TableInfo>;
  /**
   * The other tables each object's settings have used, by object ID, so that
   * after e.g. coloring the cell circles by their own table, the columns of
   * the AnnData table can still be used
   */
  remembered: Map<string, Set<string>>;
};
