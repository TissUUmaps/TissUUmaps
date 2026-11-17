import { describe, expect, it } from "vitest";

import {
  type ColorConfig,
  labelsDefaults,
  pointsDefaults,
} from "@tissuumaps/core";

import {
  hide,
  isHiddenByConstant,
  readColorBy,
  showIfHidden,
  withColorColumn,
  withoutColorColumn,
} from "./colorBy";

const white = { r: 255, g: 255, b: 255 };

describe("readColorBy", () => {
  it("reads a column of the own table grouped into categories", () => {
    // configurations keep the branches of their inactive sources
    const config = {
      constant: { value: white },
      source: "groupBy",
      groupBy: { column: "gene", map: "m1" },
    } as ColorConfig;
    expect(readColorBy(config, "t1")).toEqual({
      kind: "column",
      column: "gene",
      encoding: "categories",
    });
    expect(
      readColorBy(
        { ...config, groupBy: { table: "t1", column: "gene", map: undefined } },
        "t1",
      ),
    ).toMatchObject({ kind: "column", column: "gene" });
  });

  it("reads a column of the own table mapped to a gradient", () => {
    expect(
      readColorBy(
        { source: "from", from: { column: "qv", palette: "viridis" } },
        "t1",
      ),
    ).toEqual({ kind: "column", column: "qv", encoding: "gradient" });
  });

  it("reads the default coloring without a column, and a hidden one, as none", () => {
    expect(readColorBy({ constant: { value: white } }, "t1")).toEqual({
      kind: "none",
    });
    expect(
      readColorBy(labelsDefaults.labelColor, "t1", { noneSource: "random" }),
    ).toEqual({ kind: "none" });
    expect(
      readColorBy(
        { source: "from", from: { column: "qv", palette: "viridis" } },
        "t1",
        { isHidden: true },
      ),
    ).toEqual({ kind: "none" });
  });

  it("reads another coloring without a column as custom", () => {
    expect(
      readColorBy({ constant: { value: white } }, "t1", {
        noneSource: "random",
      }),
    ).toEqual({ kind: "custom", label: "Single color" });
    expect(readColorBy(labelsDefaults.labelColor, "t1")).toEqual({
      kind: "custom",
      label: "Random colors",
    });
  });

  it("reads columns of other tables, or without a table, as custom", () => {
    expect(
      readColorBy(
        { groupBy: { table: "t2", column: "gene", map: undefined } },
        "t1",
      ),
    ).toMatchObject({ kind: "custom" });
    expect(
      readColorBy({ groupBy: { column: "gene", map: undefined } }, null),
    ).toMatchObject({ kind: "custom" });
  });
});

describe("withColorColumn", () => {
  it("groups by a new column with the default palette", () => {
    expect(
      withColorColumn(
        { constant: { value: white } },
        "gene",
        "categories",
        "t1",
      ),
    ).toEqual({
      constant: { value: white },
      source: "groupBy",
      groupBy: { column: "gene", palette: "distinct", map: undefined },
    });
  });

  it("maps a new column to the gradient's palette, without a range", () => {
    expect(
      withColorColumn(
        {
          source: "from",
          from: { column: "qv", palette: "batlow", range: [0, 1] },
        },
        "area",
        "gradient",
        "t1",
      ),
    ).toEqual({
      source: "from",
      from: { column: "area", palette: "batlow" },
    });
    expect(
      withColorColumn({ constant: { value: white } }, "area", "gradient", "t1"),
    ).toEqual({
      constant: { value: white },
      source: "from",
      from: { column: "area", palette: "viridis" },
    });
  });

  it("groups a new column with the categories' palette, without a map", () => {
    expect(
      withColorColumn(
        {
          source: "groupBy",
          groupBy: { column: "leiden", palette: "batlowS", map: "m1" },
        },
        "cell_type",
        "categories",
        "t1",
      ),
    ).toEqual({
      source: "groupBy",
      groupBy: { column: "cell_type", palette: "batlowS", map: undefined },
    });
  });

  it("keeps a branch naming the own table by its ID", () => {
    const config = {
      source: "from",
      from: { table: "t1", column: "qv", palette: "batlow", range: [0, 1] },
    } as ColorConfig;
    expect(withColorColumn(config, "qv", "gradient", "t1")).toEqual(config);
    // a branch of another table is not the own column: only its palette stays
    expect(withColorColumn(config, "qv", "gradient", "t2")).toEqual({
      source: "from",
      from: { column: "qv", palette: "batlow" },
    });
  });

  it("keeps the branches of the same column when switching", () => {
    const config = {
      source: "from",
      from: { column: "qv", palette: "batlow", range: [0, 1] },
      groupBy: { column: "qv", map: "m1" },
    } as ColorConfig;
    const categories = withColorColumn(config, "qv", "categories", "t1");
    expect(categories).toEqual({ ...config, source: "groupBy" });
    expect(withColorColumn(categories, "qv", "gradient", "t1")).toEqual(config);
  });
});

describe("withoutColorColumn", () => {
  it("returns to the default source, keeping its branch", () => {
    expect(
      withoutColorColumn(
        {
          constant: { value: { r: 1, g: 2, b: 3 } },
          source: "groupBy",
          groupBy: { column: "gene", map: undefined },
        } as ColorConfig,
        pointsDefaults.pointColor,
      ),
    ).toMatchObject({
      source: "constant",
      constant: { value: { r: 1, g: 2, b: 3 } },
    });
    expect(
      withoutColorColumn(
        { groupBy: { column: "gene", map: undefined } },
        labelsDefaults.labelColor,
      ),
    ).toMatchObject({ source: "random", random: { palette: "distinct" } });
  });
});

describe("showIfHidden", () => {
  it("shows a constantly hidden fill only", () => {
    expect(isHiddenByConstant({ constant: { value: false } })).toBe(true);
    expect(showIfHidden({ constant: { value: false } })).toEqual({
      source: "constant",
      constant: { value: true },
    });
    const fromConfig = { from: { column: "visible" } };
    expect(showIfHidden(fromConfig)).toBe(fromConfig);
    expect(hide(fromConfig)).toEqual({
      ...fromConfig,
      source: "constant",
      constant: { value: false },
    });
  });
});
