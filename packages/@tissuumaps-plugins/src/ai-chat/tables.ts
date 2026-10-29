import type {
  DataStoreApi,
  ProjectStore,
  TableColumnRef,
  TableData,
  TypedArrayOrArray,
} from "@tissuumaps/core";

import {
  annotatingTable,
  columnRefOf,
  columnSettingsOf,
  groupByColumnOf,
  list,
  round,
} from "./objects";
import type { ColumnInfo, ItemsObject, TableKnowledge } from "./types";

/** Up to this many different values, all of them are listed (e.g. clusters) */
export const maxGroups = 20;

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
export const maxRowsToRead = 5_000_000;

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
export const clusterColumn = /clust|leiden|louvain|kmeans|k_means|phenograph/i;

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
export function objectTables(
  object: ItemsObject,
  tables: TableKnowledge,
): string[] {
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
export function linkedTables(
  project: ProjectStore,
  tables: TableKnowledge,
): string[] {
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
export function findUsedColumns(
  project: ProjectStore,
  tableId: string,
): Set<string> {
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
export async function listColumns(
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
export async function matrixColumns(
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
export async function describeMatrix(
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
export function describeValues(
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
export async function findColumn(
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
export async function readColumn(
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
export function findGroup(
  groups: string[],
  wanted: unknown,
): string | undefined {
  const lower = String(wanted).trim().toLowerCase();
  const lastWord = lower.split(/\s+/).at(-1);
  return (
    groups.find((group) => group.toLowerCase() === lower) ??
    groups.find((group) => group.toLowerCase() === lastWord)
  );
}
