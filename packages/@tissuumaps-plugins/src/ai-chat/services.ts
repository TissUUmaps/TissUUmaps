import {
  type DataStoreApi,
  type PluginStores,
  createTable,
} from "@tissuumaps/core";

import {
  type ChatPlugins,
  type PluginStorage,
  makeChatPlugins,
} from "./chat-plugins";
import { registerChatShapes } from "./chat-shapes";
import {
  type ComputedValues,
  columnValues,
  computedTableType,
  forgetVersions,
  keepVersion,
  maxComputedColumns,
  registerComputedTables,
  valuesByRow,
  versionOf,
  withComputedTables,
} from "./computed-tables";
import { pluginStorage } from "./plugin-storage";
import { loadProjectSkills } from "./project-skills";
import {
  type DownloadProgress,
  type PythonFiles,
  makePythonFiles,
} from "./python-files";
import { type PythonKernel, startPythonKernel } from "./python-sandbox";
import type { ScriptQuestion } from "./sandbox";
import { type SkillList, makeSkillList } from "./skill-list";
import {
  forgetStoredSkill,
  loadStoredSkills,
  storeSkill,
} from "./skill-storage";
import {
  columnsForCode,
  describeMatrixReading,
  matrixColumns,
  maxRowsToRead,
} from "./tables";
import type { Change } from "./types";

// What the chat keeps while TissUUmaps runs, also while its panel is closed:
// the files of Python, the skills, the tables computed by the chat, and the
// answers to what code in the sandbox asks for. The plugins made by the chat
// use them too, so they live in the setup of the plugin, not in its panel.

/** What code in the sandbox asks for to read the data of the project */
export type DataQuestion = Extract<
  ScriptQuestion,
  { kind: "tables" | "columns" | "values" }
>;

/** What code asks for to save a column */
export type SaveColumnQuestion = Extract<
  ScriptQuestion,
  { kind: "save_column" }
>;

/** A column that code saved into a table computed by the chat */
export type SavedColumn = {
  /** What the code gets back */
  result: { table: string; column: string; rows: number; values: string };
  /** The change, which the chat can take back with Undo */
  change: Change;
  /** The table computed by the chat, and its name */
  tableId: string;
  tableName: string;
  /** The column, and its values */
  column: string;
  values: ComputedValues;
};

/** Python that stays for the next runs of the same code */
export type KernelSlot = {
  /**
   * Gives the Python of this slot, started first if needed; code from
   * another source gets a new one
   *
   * @param source - Where the code comes from, e.g. "chat"
   * @param options - Optional abort signal, which stops the start
   * @returns The kernel
   * @throws If Python cannot start, or is stopped
   */
  kernel: (
    source: string,
    options?: { signal?: AbortSignal },
  ) => Promise<PythonKernel>;
  /** Stops Python: the next code starts it again */
  stop: () => void;
};

/** What the chat keeps while TissUUmaps runs */
export type ChatServices = {
  /** The stores of the app */
  stores: PluginStores;
  /** The data store, with the tables computed by the chat */
  dataStore: DataStoreApi;
  /** The files of Python, downloaded and checked once */
  pythonFiles: PythonFiles;
  /** The download of the files of Python in progress, if any */
  pythonDownload: () => DownloadProgress | undefined;
  /**
   * Calls a function whenever the download of Python goes on or ends
   *
   * @returns A function that stops calling it
   */
  onPythonDownload: (listener: () => void) => () => void;
  /** Makes a place for a Python of its own, e.g. for one plugin */
  makeKernelSlot: () => KernelSlot;
  /** The skills: of the open project, and those added by hand */
  skillList: SkillList;
  /**
   * What the panel does when the skills change, and when the user clicks
   * "Add skill from GitHub…"; set while the panel is open
   */
  hooks: { skillsChanged?: () => void; addSkillFromGitHub?: () => void };
  /**
   * Answers a question of code about the data: the tables, their columns,
   * the values of a column
   *
   * @throws If it cannot be answered, e.g. an unknown table
   */
  readData: (
    question: DataQuestion,
    options?: { signal?: AbortSignal },
  ) => Promise<unknown>;
  /**
   * Saves a column that code computed into the table "(chat)" of a table
   *
   * @throws If it cannot be saved, with the reason for the code
   */
  saveColumn: (
    question: SaveColumnQuestion,
    options?: { signal?: AbortSignal },
  ) => Promise<SavedColumn>;
  /** The plugins made by the chat */
  plugins: ChatPlugins;
  /** Called when the chat stops: aborted then */
  signal: AbortSignal;
  /** Stops everything, e.g. when the AI chat is unregistered */
  dispose: () => void;
};

/**
 * Makes what the chat keeps while TissUUmaps runs
 *
 * @param stores - The stores of the app
 * @param options - For the tests: where the plugins made by the chat are
 * remembered, and how their code is imported
 * @returns The services
 */
export function makeChatServices(
  stores: PluginStores,
  options?: {
    pluginStorage?: PluginStorage;
    loadPlugin?: (code: string) => Promise<unknown>;
  },
): ChatServices {
  const { projectStore, appStore } = stores;
  const dataStore = withComputedTables(stores.dataStore, projectStore);
  const abortController = new AbortController();
  const { signal } = abortController;
  const hooks: ChatServices["hooks"] = {};

  let download: DownloadProgress | undefined;
  const downloadListeners = new Set<() => void>();
  const pythonFiles = makePythonFiles((progress) => {
    download = progress;
    for (const listener of downloadListeners) {
      listener();
    }
  });
  /** The Pythons that are running, stopped with the chat */
  const slots = new Set<KernelSlot>();

  const skillList = makeSkillList(
    { load: loadStoredSkills, save: storeSkill, remove: forgetStoredSkill },
    () => hooks.skillsChanged?.(),
    { signal, addFromGitHub: () => hooks.addSkillFromGitHub?.() },
  );

  /** The project and workspace whose skills were read last */
  let skillsRead: { instanceId: string; workspace: unknown } | null = null;

  /**
   * Reads the skills of the open project again when another project is
   * opened or another folder connected; a later read wins over an earlier one
   */
  function readProjectSkills(): void {
    const { source, instanceId } = projectStore.getState();
    const workspace = appStore.getState().workspace ?? null;
    if (
      skillsRead?.instanceId === instanceId &&
      skillsRead.workspace === workspace
    ) {
      return;
    }
    const read = { instanceId, workspace };
    skillsRead = read;
    loadProjectSkills(source, workspace, { signal }).then(
      (project) => {
        if (skillsRead === read) {
          skillList.setProjectSkills(project);
        }
      },
      (error: unknown) => {
        if (!signal.aborted && skillsRead === read) {
          skillList.setProjectSkills(
            error instanceof Error ? error : new Error(String(error)),
          );
        }
      },
    );
  }

  function makeKernelSlot(): KernelSlot {
    let current: { kernel: PythonKernel; source: string } | null = null;
    const slot: KernelSlot = {
      kernel: async (source, options) => {
        if (
          current === null ||
          current.kernel.isStopped() ||
          current.source !== source
        ) {
          slot.stop();
          slots.add(slot);
          current = {
            kernel: await startPythonKernel(pythonFiles, options),
            source,
          };
        }
        return current.kernel;
      },
      stop: () => {
        current?.kernel.stop();
        current = null;
        slots.delete(slot);
      },
    };
    return slot;
  }

  async function readData(
    question: DataQuestion,
    options?: { signal?: AbortSignal },
  ): Promise<unknown> {
    const { signal: stop } = options ?? {};
    stop?.throwIfAborted();
    if (question.kind === "tables") {
      return projectStore
        .getState()
        .tables.map(({ id, name }) => ({ id, name }));
    }
    const table = projectStore
      .getState()
      .tables.find(
        ({ id, name }) => id === question.table || name === question.table,
      );
    const tableRef =
      table === undefined
        ? undefined
        : dataStore.getState().tableDataRefs.get(table.id);
    if (table === undefined || tableRef === undefined) {
      throw new Error(
        `there is no loaded table "${question.table}"; tm.tables() lists the tables`,
      );
    }
    const data = await tableRef.promise;
    stop?.throwIfAborted();
    if (question.kind === "columns") {
      if (question.matrix !== undefined) {
        return matrixColumns(data, question.matrix, { signal: stop });
      }
      return columnsForCode(data, { signal: stop });
    }
    // the name of a whole matrix, e.g. "obsm/spatial", reads nothing: say how
    // to read its columns
    const inner = await matrixColumns(data, question.column, {
      signal: stop,
    }).catch(() => []);
    if (inner.length > 0) {
      throw new Error(describeMatrixReading(question.column, inner));
    }
    const rows = data.getSize();
    if (rows > maxRowsToRead) {
      throw new Error(
        `the table "${table.name}" has ${rows} rows, too many to read here (at most ${maxRowsToRead})`,
      );
    }
    const values = await data.loadValues(question.column, { signal: stop });
    // a copy of these values only, not of a buffer they may share
    return values.slice();
  }

  /**
   * Saves a column into the table "(chat)" of a table: a new version of it,
   * so that TissUUmaps reads the values again. The original table is never
   * changed.
   */
  async function saveColumn(
    question: SaveColumnQuestion,
    options?: { signal?: AbortSignal },
  ): Promise<SavedColumn> {
    const { signal: stop } = options ?? {};
    const project = projectStore.getState();
    const given = project.tables.find(
      ({ id, name }) => id === question.table || name === question.table,
    );
    if (given === undefined) {
      throw new Error(
        `there is no table "${question.table}"; tm.tables() lists the tables`,
      );
    }
    // a table computed by the chat stands for its original table
    const originId = versionOf(given.dataSource.source)?.originId ?? given.id;
    const origin = project.tables.find(({ id }) => id === originId);
    const column =
      typeof question.column === "string" ? question.column.trim() : "";
    if (
      column === "" ||
      column.length > 60 ||
      /[[\]\r\n]/.test(column) ||
      column.toLowerCase() === "id"
    ) {
      throw new Error(
        `${JSON.stringify(question.column)} cannot be the name of a column: give a short name, without [ ], other than "id"`,
      );
    }
    const tableRef = dataStore.getState().tableDataRefs.get(originId);
    if (origin === undefined || tableRef === undefined) {
      throw new Error(`the table "${question.table}" is not loaded`);
    }
    const data = await tableRef.promise;
    stop?.throwIfAborted();
    const rows = data.getSize();
    if (rows > maxRowsToRead) {
      throw new Error(
        `the table "${origin.name}" has ${rows} rows, too many for a computed column (at most ${maxRowsToRead})`,
      );
    }
    const values = columnValues(question.values);
    const byRow =
      typeof values === "string" || question.ids === undefined
        ? values
        : valuesByRow(data.getIds(), question.ids, values);
    if (typeof byRow === "string") {
      throw new Error(byRow);
    }
    if (byRow.length !== rows) {
      throw new Error(
        `the table "${origin.name}" has ${rows} rows, but there are ${byRow.length} values: give one value per row, in the order of tm.read, or give the ids of the values`,
      );
    }
    // the project may have changed while the values were read
    const now = projectStore.getState();
    const tableId = `${computedTableType}:${originId}`;
    const existing = now.tables.find(({ id }) => id === tableId);
    if (
      existing !== undefined &&
      existing.dataSource.type !== computedTableType
    ) {
      throw new Error(
        `the table "${existing.name}" was kept as a CSV file, so the chat cannot add columns to it`,
      );
    }
    const previous = versionOf(existing?.dataSource.source);
    const saved = now.tables
      .filter(({ dataSource }) => dataSource.type === computedTableType)
      .reduce(
        (count, table) =>
          count + (versionOf(table.dataSource.source)?.columns.size ?? 0),
        0,
      );
    if (
      !(previous?.columns.has(column) ?? false) &&
      saved >= maxComputedColumns
    ) {
      throw new Error(
        `the tables computed by the chat have ${maxComputedColumns} columns already, the most there can be: tell the user, who can take some back with Undo`,
      );
    }
    const columns = new Map(previous?.columns);
    columns.set(column, byRow);
    const source = keepVersion({
      originId,
      ids: data.getIds(),
      names: data.getNames?.(),
      columns,
    });
    const name = existing?.name ?? `${origin.name} (chat)`;
    if (existing === undefined) {
      now.addTable(
        createTable({
          id: tableId,
          name,
          dataSource: { type: computedTableType, source },
        }),
      );
    } else {
      now.updateTable(tableId, {
        dataSource: { ...existing.dataSource, source },
      });
    }
    return {
      result: {
        table: name,
        column,
        rows,
        values: byRow instanceof Float64Array ? "numbers" : "text",
      },
      change: {
        type: "computed",
        kind: "tables",
        id: tableId,
        name,
        property: column,
        before: existing?.dataSource.source,
        after: source,
      },
      tableId,
      tableName: name,
      column,
      values: byRow,
    };
  }

  // TissUUmaps loads the tables computed by the chat from memory
  registerComputedTables(appStore);
  registerChatShapes(appStore);
  const unsubscribeProject = projectStore.subscribe((state, previous) => {
    if (state.instanceId !== previous.instanceId) {
      // the computed tables of the other project, unless this one has them too
      forgetVersions(
        new Set(
          state.tables
            .filter(({ dataSource }) => dataSource.type === computedTableType)
            .map(({ dataSource }) => dataSource.source ?? ""),
        ),
      );
    }
    readProjectSkills();
  });
  const unsubscribeWorkspace = appStore.subscribe(readProjectSkills);
  readProjectSkills();

  const host = {
    stores,
    pythonDownload: () => download,
    onPythonDownload: (listener: () => void) => {
      downloadListeners.add(listener);
      return () => {
        downloadListeners.delete(listener);
      };
    },
    makeKernelSlot,
    skillList,
    readData,
    saveColumn,
    signal,
  };
  const plugins = makeChatPlugins(
    host,
    options?.pluginStorage ?? pluginStorage,
    options?.loadPlugin === undefined ? {} : { load: options.loadPlugin },
  );

  return {
    ...host,
    dataStore,
    pythonFiles,
    hooks,
    plugins,
    dispose: () => {
      plugins.dispose();
      abortController.abort();
      unsubscribeProject();
      unsubscribeWorkspace();
      for (const slot of [...slots]) {
        slot.stop();
      }
      downloadListeners.clear();
    },
  };
}
