import type { TableColumnQuerySuggestion, TableData } from "@tissuumaps/core";

/**
 * Finds the column of a section by its name
 *
 * The exact name wins, else the only name equal to the text ignoring case:
 * names differing in case may name different columns (e.g. human and mouse
 * gene symbols). The matching suggestion is a query, which is resolved to its
 * column, so that only columns reach the callers (e.g. `describeColumn` or
 * `mapColumn`).
 *
 * @param tableData - The table data
 * @param sectionId - The `id` of the leaf section to search
 * @param name - The name to find
 * @param options - Optional abort signal
 * @returns The column and its name, or `null` if no column of the section has
 * the name
 */
export async function findColumnByName(
  tableData: Pick<TableData, "suggestColumnQueries" | "resolveColumnQuery">,
  sectionId: string,
  name: string,
  options?: { signal?: AbortSignal },
): Promise<{ column: string; name: string } | null> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const suggestions = await tableData.suggestColumnQueries(name, {
    sectionId,
    signal,
  });
  const getName = (suggestion: TableColumnQuerySuggestion) =>
    suggestion.name ?? suggestion.query;
  const lowerCaseName = name.toLowerCase();
  const caseMatches = suggestions.filter(
    (suggestion) => getName(suggestion).toLowerCase() === lowerCaseName,
  );
  const match =
    suggestions.find((suggestion) => getName(suggestion) === name) ??
    (caseMatches.length === 1 ? caseMatches[0] : undefined);
  if (match === undefined) {
    return null;
  }
  const column = await tableData.resolveColumnQuery(match.query, { signal });
  return column !== null ? { column, name: getName(match) } : null;
}
