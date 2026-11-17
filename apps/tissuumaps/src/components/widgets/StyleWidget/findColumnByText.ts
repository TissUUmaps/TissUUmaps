import type { TableData } from "@tissuumaps/core";

/**
 * Finds the column of a query group whose name (or query) is exactly a text
 *
 * The matching suggestion is a query, which is resolved to its column, so that
 * only columns reach the callers (e.g. `describeColumn` or `mapColumn`).
 *
 * @param tableData - The table data
 * @param queryGroupId - The `id` of the leaf query group to search
 * @param text - The text to find
 * @param options - Optional abort signal
 * @returns The column and the name of its suggestion, or `null` if no
 * column of the query group has the text as its name or query
 */
export async function findColumnByText(
  tableData: Pick<TableData, "suggestColumnQueries" | "resolveColumnQuery">,
  queryGroupId: string,
  text: string,
  options?: { signal?: AbortSignal },
): Promise<{ column: string; name: string } | null> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const suggestions = await tableData.suggestColumnQueries(text, {
    queryGroupId,
    signal,
  });
  const match = suggestions.find(
    (suggestion) => suggestion.name === text || suggestion.query === text,
  );
  if (match === undefined) {
    return null;
  }
  const column = await tableData.resolveColumnQuery(match.query, { signal });
  return column !== null ? { column, name: match.name ?? match.query } : null;
}
