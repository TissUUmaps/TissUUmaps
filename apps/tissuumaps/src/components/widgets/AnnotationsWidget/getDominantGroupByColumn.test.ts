import { describe, expect, it } from "vitest";

import type {
  ConstantConfig,
  FromConfig,
  GroupByConfig,
} from "@tissuumaps/core";

import { getDominantGroupByColumn } from "./getDominantGroupByColumn";

describe("getDominantGroupByColumn", () => {
  it("picks the column that most configurations group by", () => {
    const configs: GroupByConfig<false>[] = [
      { groupBy: { column: "cluster", map: undefined } },
      { groupBy: { column: "gene", map: undefined } },
      { groupBy: { column: "gene", map: undefined } },
    ];

    expect(getDominantGroupByColumn(configs)).toEqual({ column: "gene" });
  });

  it("picks the first configuration's column on a tie", () => {
    const configs: GroupByConfig<false>[] = [
      { groupBy: { column: "cluster", map: undefined } },
      { groupBy: { column: "gene", map: undefined } },
    ];

    expect(getDominantGroupByColumn(configs)).toEqual({ column: "cluster" });
  });

  it("ignores configurations that do not group", () => {
    const configs: (
      ConstantConfig<number> | FromConfig | GroupByConfig<false>
    )[] = [
      { constant: { value: 1 } },
      { from: { column: "gene" } },
      { groupBy: { column: "cluster", map: undefined } },
    ];

    expect(getDominantGroupByColumn(configs)).toEqual({ column: "cluster" });
  });

  it("keeps the columns of different tables apart", () => {
    const configs: GroupByConfig<false>[] = [
      { groupBy: { column: "gene", map: undefined } },
      { groupBy: { table: "other", column: "gene", map: undefined } },
      { groupBy: { table: "other", column: "gene", map: undefined } },
    ];

    expect(getDominantGroupByColumn(configs, "own")).toEqual({
      table: "other",
      column: "gene",
    });
  });

  it("returns null without any grouping configuration", () => {
    expect(getDominantGroupByColumn([])).toBeNull();
    const constantConfig: ConstantConfig<number> = { constant: { value: 1 } };
    expect(getDominantGroupByColumn([constantConfig])).toBeNull();
  });
});
