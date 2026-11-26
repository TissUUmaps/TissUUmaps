import { describe, expect, it, vi } from "vitest";

import type { TableColumnQuerySuggestion } from "@tissuumaps/core";

import { findColumnByName } from "./findColumnByName";

/**
 * A table whose suggested queries differ from the columns they resolve to
 *
 * @param suggestions - The suggestions of the section
 */
function createTableData(
  suggestions: TableColumnQuerySuggestion[] = [
    { query: "raw/X[cd3e]", name: "CD3E" },
  ],
) {
  return {
    suggestColumnQueries: vi.fn().mockResolvedValue(suggestions),
    resolveColumnQuery: vi.fn((query: string) =>
      Promise.resolve(`${query} (column)`),
    ),
  };
}

describe("findColumnByName", () => {
  it("resolves the matching suggestion to its column", async () => {
    const tableData = createTableData();
    await expect(findColumnByName(tableData, "raw/X", "CD3E")).resolves.toEqual(
      { column: "raw/X[cd3e] (column)", name: "CD3E" },
    );
    expect(tableData.suggestColumnQueries).toHaveBeenCalledWith("CD3E", {
      sectionId: "raw/X",
      signal: undefined,
    });
  });

  it("prefers the exact name", async () => {
    const tableData = createTableData([
      { query: "X[0]", name: "Cd3e" },
      { query: "X[1]", name: "CD3E" },
    ]);
    await expect(findColumnByName(tableData, "X", "CD3E")).resolves.toEqual({
      column: "X[1] (column)",
      name: "CD3E",
    });
  });

  it("matches the only name equal ignoring case", async () => {
    const tableData = createTableData();
    await expect(findColumnByName(tableData, "raw/X", "cd3e")).resolves.toEqual(
      { column: "raw/X[cd3e] (column)", name: "CD3E" },
    );
  });

  it("matches no name if several are equal ignoring case", async () => {
    const tableData = createTableData([
      { query: "X[0]", name: "Cd3e" },
      { query: "X[1]", name: "CD3E" },
    ]);
    await expect(findColumnByName(tableData, "X", "cd3e")).resolves.toBeNull();
  });

  it("does not match queries", async () => {
    const tableData = createTableData();
    await expect(
      findColumnByName(tableData, "raw/X", "raw/X[cd3e]"),
    ).resolves.toBeNull();
  });
});
