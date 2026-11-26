import { describe, expect, it } from "vitest";

import {
  type ColorConfig,
  Marker,
  type MarkerConfig,
  type SizeConfig,
  labelsDefaults,
  pointsDefaults,
} from "@tissuumaps/core";

import {
  type StylePart,
  type StyleProperty,
  findStyleBy,
  followColumn,
  getFollowState,
  getFollowingNames,
  getPartToggledNames,
  getToggledNames,
  hide,
  isHiddenByConstant,
  isSameColumnStyleBy,
  isStyleItemOn,
  isStyleItemSupported,
  readStyleBy,
  showIfHidden,
  withColorColumn,
  withFromColumn,
  withMarkerColumn,
  withoutColumn,
} from "./styleBy";

const white = { r: 255, g: 255, b: 255 };

describe("readStyleBy", () => {
  it("reads a column of the own table grouped into categories as categorical", () => {
    // configurations keep the branches of their inactive sources
    const config = {
      constant: { value: white },
      source: "groupBy",
      groupBy: { column: "gene", map: "m1" },
    } as ColorConfig;
    expect(readStyleBy<ColorConfig>(config, "t1")).toEqual({
      kind: "column",
      column: "gene",
      columnType: "categorical",
    });
    expect(
      readStyleBy<ColorConfig>(
        { ...config, groupBy: { table: "t1", column: "gene", map: undefined } },
        "t1",
      ),
    ).toMatchObject({ kind: "column", column: "gene" });
  });

  it("reads a column of the own table taken as values as numerical", () => {
    expect(
      readStyleBy<ColorConfig>(
        { source: "from", from: { column: "qv", palette: "viridis" } },
        "t1",
      ),
    ).toEqual({ kind: "column", column: "qv", columnType: "numerical" });
  });

  it("reads the default coloring without a column, and a hidden one, as none", () => {
    expect(
      readStyleBy<ColorConfig>({ constant: { value: white } }, "t1"),
    ).toEqual({
      kind: "none",
    });
    expect(
      readStyleBy<ColorConfig>(labelsDefaults.labelColor, "t1", {
        defaultConfig: labelsDefaults.labelColor,
      }),
    ).toEqual({ kind: "none" });
    expect(
      readStyleBy<ColorConfig>(
        { source: "from", from: { column: "qv", palette: "viridis" } },
        "t1",
        { isHidden: true },
      ),
    ).toEqual({ kind: "none" });
  });

  it("reads another coloring without a column as custom", () => {
    expect(
      readStyleBy<ColorConfig>({ constant: { value: white } }, "t1", {
        defaultConfig: labelsDefaults.labelColor,
      }),
    ).toEqual({ kind: "custom", label: "Single color" });
    expect(readStyleBy<ColorConfig>(labelsDefaults.labelColor, "t1")).toEqual({
      kind: "custom",
      label: "Random colors",
    });
  });

  it("reads columns of other tables, or without a table, as custom", () => {
    expect(
      readStyleBy<ColorConfig>(
        { groupBy: { table: "t2", column: "gene", map: undefined } },
        "t1",
      ),
    ).toMatchObject({ kind: "custom" });
    expect(
      readStyleBy<ColorConfig>(
        { groupBy: { column: "gene", map: undefined } },
        null,
      ),
    ).toMatchObject({ kind: "custom" });
  });
});

describe("withColorColumn", () => {
  it("groups by a new column with the default palette", () => {
    expect(
      withColorColumn(
        { constant: { value: white } },
        "gene",
        "categorical",
        "t1",
      ),
    ).toEqual({
      constant: { value: white },
      source: "groupBy",
      groupBy: { column: "gene", palette: "distinct", map: undefined },
    });
  });

  it("maps a new numerical column to the palette, without a range", () => {
    expect(
      withColorColumn(
        {
          source: "from",
          from: { column: "qv", palette: "batlow", range: [0, 1] },
        },
        "area",
        "numerical",
        "t1",
      ),
    ).toEqual({
      source: "from",
      from: { column: "area", palette: "batlow" },
    });
    expect(
      withColorColumn(
        { constant: { value: white } },
        "area",
        "numerical",
        "t1",
      ),
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
        "categorical",
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
    expect(withColorColumn(config, "qv", "numerical", "t1")).toEqual(config);
    // a branch of another table is not the own column: only its palette stays
    expect(withColorColumn(config, "qv", "numerical", "t2")).toEqual({
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
    const categories = withColorColumn(config, "qv", "categorical", "t1");
    expect(categories).toEqual({ ...config, source: "groupBy" });
    expect(withColorColumn(categories, "qv", "numerical", "t1")).toEqual(
      config,
    );
  });
});

describe("withoutColumn for colors", () => {
  it("returns to the default's color source, keeping its branch", () => {
    expect(
      withoutColumn<ColorConfig>(
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
      withoutColumn<ColorConfig>(
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

describe("findStyleBy", () => {
  it("follows the first column, else the first property", () => {
    const gene = {
      kind: "column",
      column: "gene",
      columnType: "categorical",
    } as const;
    expect(findStyleBy([{ kind: "none" }, gene])).toBe(gene);
    expect(
      findStyleBy([
        { kind: "custom", label: "Single color" },
        { kind: "none" },
      ]),
    ).toEqual({ kind: "custom", label: "Single color" });
  });

  it("tells a property apart that follows the same column as another type", () => {
    const categorical = {
      kind: "column",
      column: "qv",
      columnType: "categorical",
    } as const;
    const numerical = { ...categorical, columnType: "numerical" } as const;
    expect(isSameColumnStyleBy(categorical, { ...categorical })).toBe(true);
    expect(isSameColumnStyleBy(categorical, numerical)).toBe(false);
    expect(isSameColumnStyleBy({ kind: "none" }, { kind: "none" })).toBe(false);
  });
});

describe("withMarkerColumn", () => {
  it("groups the marker by a new column, keeping the same column's map", () => {
    const constantMarker: MarkerConfig = { constant: { value: Marker.Disc } };
    expect(withMarkerColumn(constantMarker, "gene", "t1")).toEqual({
      constant: { value: Marker.Disc },
      source: "groupBy",
      groupBy: { column: "gene", map: undefined },
    });
    const grouped = {
      source: "constant",
      constant: { value: Marker.Disc },
      groupBy: { column: "gene", map: "m1" },
    } as MarkerConfig;
    expect(withMarkerColumn(grouped, "gene", "t1")).toEqual({
      ...grouped,
      source: "groupBy",
    });
  });
});

describe("withFromColumn", () => {
  it("takes the values of a new column in the unit of the constant", () => {
    expect(
      withFromColumn<SizeConfig>(
        { constant: { value: 3, unit: "layer" } },
        "area",
        "t1",
      ),
    ).toEqual({
      constant: { value: 3, unit: "layer" },
      source: "from",
      from: { column: "area", unit: "layer" },
    });
  });

  it("keeps the branch of the same column", () => {
    const config = {
      source: "constant",
      constant: { value: 3 },
      from: { column: "area", unit: "world" },
    } as SizeConfig;
    expect(withFromColumn(config, "area", "t1")).toEqual({
      ...config,
      source: "from",
    });
  });
});

describe("withoutColumn", () => {
  it("returns to the constant, keeping the other branches", () => {
    expect(
      withoutColumn<MarkerConfig>(
        {
          source: "groupBy",
          constant: { value: Marker.Cross },
          groupBy: { column: "leiden", map: "m1" },
        } as MarkerConfig,
        pointsDefaults.pointMarker,
      ),
    ).toEqual({
      source: "constant",
      constant: { value: Marker.Cross },
      groupBy: { column: "leiden", map: "m1" },
    });
    expect(
      withoutColumn<SizeConfig>(
        { source: "from", from: { column: "area" } },
        pointsDefaults.pointSize,
      ),
    ).toMatchObject({
      source: "constant",
      constant: pointsDefaults.pointSize.constant,
    });
  });
});

describe("getFollowState", () => {
  const gene = {
    kind: "column",
    column: "gene",
    columnType: "categorical",
  } as const;
  const qv = { kind: "column", column: "qv", columnType: "numerical" } as const;

  it("follows the new column with the named properties only", () => {
    const followState = getFollowState(
      { color: gene, marker: gene, size: { kind: "none" } },
      qv,
      ["color"],
    );
    expect(followState("color")).toEqual({ column: qv, wasFollowing: true });
    expect(followState("marker")).toEqual({ column: null, wasFollowing: true });
    expect(followState("size")).toEqual({ column: null, wasFollowing: false });
  });

  it("does not count a property on another column as following", () => {
    const followState = getFollowState(
      { color: gene, opacity: qv },
      { kind: "none" },
      [],
    );
    expect(followState("opacity")).toEqual({
      column: null,
      wasFollowing: false,
    });
  });
});

describe("followColumn", () => {
  const options = {
    withColumn: ({ column }: { column: string }) => `with ${column}`,
    withoutColumn: () => "without",
  };
  const qv = { kind: "column", column: "qv", columnType: "numerical" } as const;

  it("follows a column, stops following, or keeps the configuration", () => {
    expect(
      followColumn("config", { column: qv, wasFollowing: false }, options),
    ).toBe("with qv");
    expect(
      followColumn("config", { column: null, wasFollowing: true }, options),
    ).toBe("without");
    expect(
      followColumn("config", { column: null, wasFollowing: false }, options),
    ).toBe("config");
  });
});

describe("style items", () => {
  const color = {
    name: "color",
    columnTypes: ["categorical", "numerical"],
    isOn: true,
  } satisfies StyleProperty;
  const marker = {
    name: "marker",
    columnTypes: ["categorical"],
    isOn: true,
  } satisfies StyleProperty;
  const size = {
    name: "size",
    columnTypes: ["numerical"],
    isOn: false,
  } satisfies StyleProperty;
  const fill = {
    name: "fill",
    properties: [
      {
        name: "fillColor",
        columnTypes: ["categorical", "numerical"],
        isOn: false,
      },
      { name: "fillOpacity", columnTypes: ["numerical"], isOn: false },
    ],
  } satisfies StylePart;

  it("keeps the properties that can follow a new column type", () => {
    expect(getFollowingNames([color, marker, size], "numerical")).toEqual([
      "color",
    ]);
    expect(
      getFollowingNames(
        [{ ...color, isOn: false }, { ...marker, isOn: false }, size],
        "numerical",
      ),
    ).toEqual(["color"]);
    expect(
      getFollowingNames([{ ...marker, isOn: false }, size], "numerical"),
    ).toEqual(["size"]);
  });

  it("starts a newly toggled part with its first property for the type", () => {
    expect(
      getToggledNames([color, marker, fill], ["color", "fill"], "numerical"),
    ).toEqual(["color", "fillColor"]);
    expect(getToggledNames([color, marker], [], "categorical")).toBeNull();
  });

  it("keeps the properties of a part that stays on", () => {
    const fillOn = {
      ...fill,
      properties: [
        { ...fill.properties[0], isOn: false },
        { ...fill.properties[1], isOn: true },
      ],
    } satisfies StylePart;
    expect(getToggledNames([fillOn, size], ["fill"], "numerical")).toEqual([
      "fillOpacity",
    ]);
    expect(isStyleItemOn(fillOn)).toBe(true);
    expect(isStyleItemSupported(fill, "categorical")).toBe(true);
    expect(isStyleItemSupported(size, "categorical")).toBe(false);
  });

  it("sets the toggles of a part, keeping the other properties", () => {
    expect(
      getPartToggledNames([color, fill], fill, ["fillColor", "fillOpacity"]),
    ).toEqual(["color", "fillColor", "fillOpacity"]);
    expect(
      getPartToggledNames([{ ...color, isOn: false }, fill], fill, []),
    ).toBeNull();
  });
});
