import type {
  AppStoreApi,
  DataRef,
  DataStoreApi,
  IDArray,
  ProjectStoreApi,
  TableColumnQuerySuggestion,
  TableData,
  TableDataProvider,
  TableDataSource,
} from "@tissuumaps/core";

// Code in the sandbox can save a column of values, one per row of a table of
// the project (tm.saveColumn, or tm.save_column in Python), e.g. the number
// of neighbors of each cell. The panel keeps the columns in a table of its
// own, "cells (chat)" for the table "cells", with the same item IDs, so that
// the colors and filters of TissUUmaps can use it like the table itself. The
// original data is never changed. The values live in memory: the table has
// the data source type below, which this file loads from memory, until the
// user keeps it as a CSV file. Every change of the columns gives the table a
// new source, since TissUUmaps keeps the values it has read of each source.

/** The data source type of the tables computed by the chat */
export const computedTableType = "ai-chat";

/** The most columns that the computed tables of a project may have in all */
export const maxComputedColumns = 20;

/** The values of a computed column: numbers, or text */
export type ComputedValues = Float64Array | string[];

/** One version of a computed table: its rows and columns */
export type ComputedTable = {
  /** The table whose rows these are */
  originId: string;
  /** The item IDs of the rows, those of the original table */
  ids: IDArray;
  /** The item names of the rows, if the original table has them */
  names: string[] | undefined;
  /** The columns, in the order they were saved */
  columns: Map<string, ComputedValues>;
};

/**
 * The versions of the computed tables, by data source; they stay while the
 * page is open, so that Undo can go back to one, and a project saved and
 * opened again in the same page still finds them
 */
const versions = new Map<string, ComputedTable>();

/**
 * Keeps a version of a computed table
 *
 * @param table - The version
 * @returns Its source, for the data source of the table
 */
export function keepVersion(table: ComputedTable): string {
  const source = `${computedTableType}:${crypto.randomUUID()}`;
  versions.set(source, table);
  return source;
}

/**
 * A version of a computed table
 *
 * @param source - Its source
 * @returns The version, or undefined if the page does not have it
 */
export function versionOf(
  source: string | undefined,
): ComputedTable | undefined {
  return source === undefined ? undefined : versions.get(source);
}

/**
 * Forgets the versions that no table uses any more, e.g. after another
 * project was opened
 *
 * @param used - The sources still in use
 */
export function forgetVersions(used: Set<string>): void {
  for (const source of versions.keys()) {
    if (!used.has(source)) {
      versions.delete(source);
    }
  }
}

/**
 * A computed table, as TissUUmaps reads tables
 *
 * @param table - The version
 * @returns The table data
 */
export function computedTableData(table: ComputedTable): TableData {
  const names = [...table.columns.keys()];
  return {
    // the versions are kept for Undo, so there is nothing to release
    close: () => {},
    getIds: () => table.ids,
    getSize: () => table.ids.length,
    ...(table.names === undefined ? {} : { getNames: () => table.names }),
    suggestColumnQueries: (currentQuery: string) => {
      const lower = currentQuery.toLowerCase();
      const matching = names.filter((name) =>
        name.toLowerCase().startsWith(lower),
      );
      return Promise.resolve(
        [...matching, ...names.filter((name) => !matching.includes(name))].map(
          (query): TableColumnQuerySuggestion => ({ query }),
        ),
      );
    },
    resolveColumnQuery: (query: string) =>
      Promise.resolve(names.includes(query) ? query : null),
    loadValues: <T>(column: string) => {
      const values = table.columns.get(column);
      if (values === undefined) {
        return Promise.reject(new Error(`the table has no column "${column}"`));
      }
      return Promise.resolve(values as unknown as T[]);
    },
  } as TableData;
}

/** Loads the computed tables from memory, for TissUUmaps */
export const computedTableProvider: TableDataProvider<
  TableDataSource,
  TableData
> = {
  name: "AI chat (computed)",
  schema: {
    type: "object",
    properties: { source: { type: "string", readOnly: true } },
  },
  uischema: { type: "VerticalLayout", elements: [] },
  normalize: (dataSource) => dataSource,
  load: (dataSource) => {
    const table = versionOf(dataSource.source);
    return table === undefined
      ? Promise.reject(
          new Error(
            "this table was computed by the AI chat in an earlier session, and was not kept as a CSV file, so its values are gone",
          ),
        )
      : Promise.resolve(computedTableData(table));
  },
};

/**
 * The data store as the panel reads it, with the tables computed by the chat
 * always loaded: TissUUmaps loads a table only once something it shows uses
 * it, but the tools must find the columns of a new computed table at once
 *
 * @param dataStore - The data store of TissUUmaps
 * @param projectStore - The project store, with the computed tables
 * @returns The data store, as the panel reads it
 */
export function withComputedTables(
  dataStore: DataStoreApi,
  projectStore: ProjectStoreApi,
): DataStoreApi {
  // one reference per version, so that the panel reads each version once
  const refs = new Map<string, DataRef<TableData>>();
  const getState = () => {
    const state = dataStore.getState();
    const computed = projectStore
      .getState()
      .tables.filter(({ dataSource }) => dataSource.type === computedTableType);
    if (computed.length === 0) {
      return state;
    }
    const tableDataRefs = new Map(state.tableDataRefs);
    for (const { id, dataSource } of computed) {
      const table = versionOf(dataSource.source);
      if (table === undefined) {
        continue;
      }
      let ref = refs.get(dataSource.source!);
      if (ref === undefined) {
        const data = computedTableData(table);
        ref = { status: "loaded", data, promise: Promise.resolve(data) };
        refs.set(dataSource.source!, ref);
      }
      tableDataRefs.set(id, ref);
    }
    return { ...state, tableDataRefs };
  };
  return { ...dataStore, getState };
}

/**
 * Lets TissUUmaps load the computed tables, once
 *
 * @param appStore - The app store
 */
export function registerComputedTables(appStore: AppStoreApi): void {
  const state = appStore.getState();
  if (
    typeof state.registerTableDataProvider === "function" &&
    state.tableDataProviders?.get(computedTableType) === undefined
  ) {
    state.registerTableDataProvider(computedTableType, computedTableProvider);
  }
}

/**
 * Checks and converts the values that code saves as a column: numbers, or
 * text if any value is text
 *
 * @param values - The values, as the sandbox sent them
 * @returns The values, or why they cannot be a column
 */
export function columnValues(values: unknown): ComputedValues | string {
  if (ArrayBuffer.isView(values) && !(values instanceof DataView)) {
    return Float64Array.from(values as unknown as ArrayLike<number>);
  }
  if (!Array.isArray(values)) {
    return "the values must be a list (an array), one value per row";
  }
  const items: unknown[] = values;
  if (
    items.every(
      (value) =>
        typeof value === "number" || value === null || value === undefined,
    )
  ) {
    return Float64Array.from(items, (value) =>
      typeof value === "number" ? value : NaN,
    );
  }
  const text: string[] = [];
  for (const value of items) {
    if (value === null || value === undefined) {
      text.push("");
    } else if (
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
    ) {
      text.push(String(value));
    } else {
      return "each value must be a number or a text";
    }
  }
  return text;
}

/**
 * Puts values given for some items into the order of all the rows
 *
 * @param ids - The item IDs of all the rows
 * @param givenIds - The item IDs of the values
 * @param values - The values
 * @returns The values of all the rows, NaN or "" for the others; or why not
 */
export function valuesByRow(
  ids: IDArray,
  givenIds: unknown,
  values: ComputedValues,
): ComputedValues | string {
  if (!Array.isArray(givenIds) && !ArrayBuffer.isView(givenIds)) {
    return "ids must be a list, one item ID per value";
  }
  const given = Array.from(givenIds as ArrayLike<unknown>);
  if (given.length !== values.length) {
    return `there are ${given.length} ids but ${values.length} values`;
  }
  const rows = new Map<unknown, number>();
  Array.from(ids as ArrayLike<number | string>).forEach((id, row) =>
    rows.set(id, row),
  );
  const result: ComputedValues =
    values instanceof Float64Array
      ? new Float64Array(ids.length).fill(NaN)
      : new Array<string>(ids.length).fill("");
  const unknown: unknown[] = [];
  given.forEach((id, i) => {
    // the same ID may come as a number or as text
    const row = rows.get(id) ?? rows.get(String(id)) ?? rows.get(Number(id));
    if (row === undefined) {
      unknown.push(id);
    } else {
      result[row] = values[i]!;
    }
  });
  if (unknown.length > 0) {
    return `${unknown.length} ids are not in the table, e.g. ${JSON.stringify(unknown.slice(0, 3))}`;
  }
  return result;
}

/**
 * Whether the item IDs would come back the same from a CSV file: numbers do,
 * and text does unless it looks like a number, which a CSV file turns into one
 *
 * @param ids - The item IDs
 * @returns Whether they survive a CSV file
 */
export function idsSurviveCsv(ids: IDArray): boolean {
  if (!Array.isArray(ids)) {
    return true;
  }
  return (ids as unknown[]).every(
    (id) =>
      typeof id !== "string" ||
      id.trim() === "" ||
      !Number.isFinite(Number(id)),
  );
}

/**
 * Writes a computed table as CSV, with the item IDs in the column "id"
 *
 * @param table - The version
 * @returns The text of the file
 */
export function computedTableCsv(table: ComputedTable): string {
  const quote = (text: string) =>
    /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  const columns = [...table.columns];
  const lines = [["id", ...columns.map(([name]) => name)].map(quote).join(",")];
  for (let row = 0; row < table.ids.length; row++) {
    const cells = [String(table.ids[row])];
    for (const [, values] of columns) {
      const value = values[row]!;
      cells.push(
        typeof value === "number"
          ? Number.isNaN(value)
            ? ""
            : String(value)
          : value,
      );
    }
    lines.push(cells.map(quote).join(","));
  }
  return `${lines.join("\n")}\n`;
}

/**
 * A useful name for the CSV file of a computed table: its original table, its
 * columns and the day, e.g. "cell_circles-c3_group-2026-11-05.csv"
 *
 * @param tableName - The name of the table, e.g. "cell_circles (chat)"
 * @param columns - Its columns
 * @param day - The day, today by default
 * @returns The file name
 */
export function suggestCsvName(
  tableName: string,
  columns: string[],
  day = new Date(),
): string {
  // only letters, digits, "." and "_" in each part, which "-" separates
  const part = (text: string) =>
    text.replace(/[^\p{L}\p{N}._]+/gu, "_").replace(/^[_.]+|[_.]+$/g, "");
  const date = [
    day.getFullYear(),
    String(day.getMonth() + 1).padStart(2, "0"),
    String(day.getDate()).padStart(2, "0"),
  ].join("-");
  const origin = part(tableName.replace(/\s*\(chat\)$/, "")) || "table";
  const columnsPart = part(part(columns.join("_")).slice(0, 40));
  return `${[origin, columnsPart, date].filter((text) => text !== "").join("-")}.csv`;
}

/**
 * Checks the name of a CSV file, and adds ".csv" if it is missing
 *
 * @param given - The name, e.g. "vicini_cluster3"
 * @returns The name, e.g. "vicini_cluster3.csv", or why it is not valid
 */
export function csvFileName(
  given: string,
): { name: string } | { error: string } {
  const trimmed = given.trim();
  const name = /\.csv$/i.test(trimmed) ? trimmed : `${trimmed}.csv`;
  const stem = name.slice(0, -".csv".length);
  const valid =
    name.length <= 100 &&
    /^[\p{L}\p{N}][\p{L}\p{N} ._()-]*$/u.test(stem) &&
    !/[. ]$/.test(stem) &&
    // names that Windows keeps for itself
    !/^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$/i.test(stem);
  return valid
    ? { name }
    : {
        error: `"${given}" is not a valid file name: use letters, digits, spaces, "-", "_", "." and "( )", without folders, and at most 100 characters`,
      };
}
