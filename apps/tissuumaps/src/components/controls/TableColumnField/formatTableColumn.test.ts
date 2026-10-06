import { describe, expect, it } from "vitest";

import { createTable } from "@tissuumaps/core";

import { formatTableColumn } from "./formatTableColumn";

const own = createTable({
  id: "own",
  name: "Own",
  dataSource: { type: "test" },
});
const cells = createTable({
  id: "cells",
  name: "Cells",
  dataSource: { type: "test" },
});
const tables = [own, cells];

describe("formatTableColumn", () => {
  it("formats a column of the default table by name alone", () => {
    expect(formatTableColumn({ column: "gene" }, tables)).toBe("gene");
  });

  it("prefixes another table by name", () => {
    expect(formatTableColumn({ table: "cells", column: "gene" }, tables)).toBe(
      "Cells:gene",
    );
  });

  it("prefixes an unknown table by ID", () => {
    expect(formatTableColumn({ table: "gone", column: "gene" }, tables)).toBe(
      "gone:gene",
    );
  });
});
