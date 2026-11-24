import { describe, expect, it } from "vitest";

import {
  type ColorConfig,
  Marker,
  type MarkerConfig,
  labelsDefaults,
  pointsDefaults,
} from "@tissuumaps/core";

import {
  type ColorBy,
  hide,
  isHiddenByConstant,
  isMarkerByCategory,
  markerByCategory,
  markerFollowingColor,
  markerWithoutCategories,
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

describe("markerByCategory", () => {
  const constantMarker: MarkerConfig = { constant: { value: Marker.Disc } };

  it("groups the marker by a column", () => {
    const grouped = markerByCategory(
      constantMarker,
      "gene",
      pointsDefaults.pointMarker,
      "t1",
    );
    expect(grouped).toEqual({
      constant: { value: Marker.Disc },
      source: "groupBy",
      groupBy: { column: "gene", map: undefined },
    });
    expect(isMarkerByCategory(grouped, "gene", "t1")).toBe(true);
    expect(isMarkerByCategory(grouped, "qv", "t1")).toBe(false);
    expect(isMarkerByCategory(grouped, "gene", null)).toBe(false);
  });

  it("keeps the grouping of the same column", () => {
    const grouped = {
      source: "constant",
      constant: { value: Marker.Disc },
      groupBy: { column: "gene", map: "m1" },
    } as MarkerConfig;
    expect(
      markerByCategory(grouped, "gene", pointsDefaults.pointMarker, "t1"),
    ).toEqual({ ...grouped, source: "groupBy" });
  });

  it("returns to a constant marker", () => {
    expect(
      markerByCategory(
        { groupBy: { column: "gene", map: undefined } },
        null,
        pointsDefaults.pointMarker,
        "t1",
      ),
    ).toMatchObject({ source: "constant", constant: { value: Marker.Disc } });
  });
});

describe("markerFollowingColor", () => {
  const byLeiden: MarkerConfig = {
    source: "groupBy",
    groupBy: { column: "leiden", map: undefined },
  };
  const colorByLeiden: ColorConfig = {
    source: "groupBy",
    groupBy: { column: "leiden", map: undefined },
  };
  const isTumor: ColorBy = {
    kind: "column",
    column: "is_tumor",
    encoding: "categories",
  };
  const qv: ColorBy = { kind: "column", column: "qv", encoding: "gradient" };

  function follow(
    marker: MarkerConfig,
    previousColor: ColorConfig,
    next: ColorBy,
  ): MarkerConfig {
    return markerFollowingColor(
      marker,
      previousColor,
      next,
      pointsDefaults.pointMarker,
      "t1",
    );
  }

  it("moves the grouping to the next column colored by categories", () => {
    expect(follow(byLeiden, colorByLeiden, isTumor)).toMatchObject({
      source: "groupBy",
      groupBy: { column: "is_tumor" },
    });
  });

  it("returns to a constant marker for a gradient or no color", () => {
    expect(follow(byLeiden, colorByLeiden, qv)).toMatchObject({
      source: "constant",
    });
    expect(follow(byLeiden, colorByLeiden, { kind: "none" })).toMatchObject({
      source: "constant",
    });
  });

  it("stays on through a gradient detour", () => {
    const marker = follow(byLeiden, colorByLeiden, qv);
    // the color keeps its grouping by leiden as an inactive branch
    const colorByQv = withColorColumn(colorByLeiden, "qv", "gradient", "t1");
    expect(follow(marker, colorByQv, isTumor)).toMatchObject({
      source: "groupBy",
      groupBy: { column: "is_tumor" },
    });
    expect(
      follow(marker, colorByQv, {
        kind: "column",
        column: "leiden",
        encoding: "categories",
      }),
    ).toMatchObject({ source: "groupBy", groupBy: { column: "leiden" } });
  });

  it("stays off through a gradient detour once switched off", () => {
    const marker = markerWithoutCategories(
      byLeiden,
      pointsDefaults.pointMarker,
    );
    expect(marker).toEqual({
      source: "constant",
      constant: { value: Marker.Disc },
    });
    const colorByQv = withColorColumn(colorByLeiden, "qv", "gradient", "t1");
    const afterGradient = follow(marker, colorByLeiden, qv);
    expect(follow(afterGradient, colorByQv, isTumor)).toEqual(marker);
  });

  it("leaves a marker that does not follow the color unchanged", () => {
    const constantMarker: MarkerConfig = { constant: { value: Marker.Disc } };
    expect(follow(constantMarker, colorByLeiden, isTumor)).toBe(constantMarker);
    const byOtherTable: MarkerConfig = {
      source: "groupBy",
      groupBy: { table: "t2", column: "leiden", map: undefined },
    };
    expect(follow(byOtherTable, colorByLeiden, isTumor)).toBe(byOtherTable);
  });
});

describe("markerWithoutCategories", () => {
  it("keeps the constant and gradient branches only", () => {
    expect(
      markerWithoutCategories(
        {
          source: "groupBy",
          constant: { value: Marker.Cross },
          from: { column: "marker" },
          groupBy: { column: "leiden", map: "m1" },
        } as MarkerConfig,
        pointsDefaults.pointMarker,
      ),
    ).toEqual({
      source: "constant",
      constant: { value: Marker.Cross },
      from: { column: "marker" },
    });
  });
});
