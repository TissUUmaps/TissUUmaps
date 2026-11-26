import { describe, expect, it, vi } from "vitest";

import { findColumnByText } from "./findColumnByText";

/** A table whose suggested query differs from the column it resolves to */
function createTableData() {
  return {
    suggestColumnQueries: vi
      .fn()
      .mockResolvedValue([{ query: "raw/X[cd3e]", name: "CD3E" }]),
    resolveColumnQuery: vi.fn((query: string) =>
      Promise.resolve(
        query === "raw/X[cd3e]" ? "raw/X[ENSG00000198851]" : null,
      ),
    ),
  };
}

describe("findColumnByText", () => {
  it("resolves the matching suggestion to its column", async () => {
    const tableData = createTableData();
    await expect(findColumnByText(tableData, "raw/X", "CD3E")).resolves.toEqual(
      { column: "raw/X[ENSG00000198851]", name: "CD3E" },
    );
    expect(tableData.suggestColumnQueries).toHaveBeenCalledWith("CD3E", {
      sectionId: "raw/X",
      signal: undefined,
    });
  });

  it("matches names exactly", async () => {
    const tableData = createTableData();
    await expect(
      findColumnByText(tableData, "raw/X", "cd3e"),
    ).resolves.toBeNull();
  });
});
