import type {
  AppStoreApi,
  DataStoreApi,
  Image,
  Labels,
  Points,
  ProjectStoreApi,
  Shapes,
} from "@tissuumaps/core";

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
export type ToolContext = {
  projectStore: ProjectStoreApi;
  dataStore: DataStoreApi;
  /** For call_action, which sets the view */
  appStore: AppStoreApi;
  tables: TableKnowledge;
  /** The changes of the current request, which the tool adds to */
  changes: Change[];
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
      rows: number;
      tooBig: boolean;
      useful: ColumnInfo[];
      other: number;
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
