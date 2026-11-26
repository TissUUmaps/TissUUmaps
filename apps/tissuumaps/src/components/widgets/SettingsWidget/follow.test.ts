import { describe, expect, it } from "vitest";

import {
  type ColorConfig,
  type Config,
  Marker,
  type MarkerConfig,
  type OpacityConfig,
  type SizeConfig,
  type TableColumnType,
  type VisibilityConfig,
  labelsDefaults,
  pointsDefaults,
} from "@tissuumaps/core";

import type { SettingsWidgetPart, SettingsWidgetProperty } from ".";
import {
  type FollowedColumn,
  describeColorWithoutColumn,
  findFollowedColumn,
  getFollower,
  getFollowingNames,
  getPartToggledNames,
  getToggledNames,
  isItemOn,
  isItemSupported,
  isSameFollowedColumn,
  readFollowedColumn,
  withColorColumn,
  withFromColumn,
  withMarkerColumn,
  withOpacityColumn,
} from "./follow";

const white = { r: 255, g: 255, b: 255 };

describe("readFollowedColumn", () => {
  const colorColumnTypes: TableColumnType[] = ["categorical", "numerical"];

  it("reads a column of the own table grouped into categories as categorical", () => {
    // configurations keep the branches of their inactive sources
    const config = {
      constant: { value: white },
      source: "groupBy",
      groupBy: { column: "gene", map: "m1" },
    } as ColorConfig;
    expect(
      readFollowedColumn<ColorConfig>(config, "t1", colorColumnTypes),
    ).toEqual({
      column: "gene",
      columnType: "categorical",
    });
    expect(
      readFollowedColumn<ColorConfig>(
        { ...config, groupBy: { table: "t1", column: "gene", map: undefined } },
        "t1",
        colorColumnTypes,
      ),
    ).toMatchObject({ column: "gene" });
  });

  it("reads a column of the own table taken as values as numerical", () => {
    expect(
      readFollowedColumn<ColorConfig>(
        { source: "from", from: { column: "qv", palette: "viridis" } },
        "t1",
        colorColumnTypes,
      ),
    ).toEqual({ column: "qv", columnType: "numerical" });
  });

  it("reads a coloring without an own-table column as no column", () => {
    expect(
      readFollowedColumn<ColorConfig>(
        { constant: { value: white } },
        "t1",
        colorColumnTypes,
      ),
    ).toBeNull();
    expect(
      readFollowedColumn<ColorConfig>(
        labelsDefaults.labelColor,
        "t1",
        colorColumnTypes,
      ),
    ).toBeNull();
    expect(
      readFollowedColumn<ColorConfig>(
        { groupBy: { table: "t2", column: "gene", map: undefined } },
        "t1",
        colorColumnTypes,
      ),
    ).toBeNull();
    expect(
      readFollowedColumn<ColorConfig>(
        { groupBy: { column: "gene", map: undefined } },
        null,
        colorColumnTypes,
      ),
    ).toBeNull();
  });

  it("reads a column the property cannot follow as that type as no column", () => {
    // values per category from the group table
    expect(
      readFollowedColumn<VisibilityConfig>(
        {
          source: "groupBy",
          constant: { value: true },
          groupBy: { column: "cluster", map: "m1" },
        } as VisibilityConfig,
        "t1",
        ["numerical"],
      ),
    ).toBeNull();
    expect(
      readFollowedColumn<MarkerConfig>(
        { source: "from", from: { column: "marker" } },
        "t1",
        ["categorical"],
      ),
    ).toBeNull();
  });
});

describe("describeColorWithoutColumn", () => {
  it("describes the color shown while no column is picked", () => {
    const constant: ColorConfig = { constant: { value: white } };
    expect(
      describeColorWithoutColumn(constant, pointsDefaults.pointColor),
    ).toBe("None");
    expect(
      describeColorWithoutColumn(constant, labelsDefaults.labelColor),
    ).toBe("Single color");
    expect(
      describeColorWithoutColumn(
        labelsDefaults.labelColor,
        labelsDefaults.labelColor,
      ),
    ).toBe("Random colors");
    expect(
      describeColorWithoutColumn(
        { groupBy: { table: "t2", column: "gene", map: undefined } },
        pointsDefaults.pointColor,
      ),
    ).toBe("Custom (see Advanced)");
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

describe("findFollowedColumn", () => {
  it("follows the first column, else the first property", () => {
    const gene = { column: "gene", columnType: "categorical" } as const;
    expect(findFollowedColumn([null, gene])).toBe(gene);
    expect(findFollowedColumn([null, null])).toBeNull();
  });

  it("tells a property apart that follows the same column as another type", () => {
    const categorical = { column: "qv", columnType: "categorical" } as const;
    const numerical = { ...categorical, columnType: "numerical" } as const;
    expect(isSameFollowedColumn(categorical, { ...categorical })).toBe(true);
    expect(isSameFollowedColumn(categorical, numerical)).toBe(false);
    expect(isSameFollowedColumn(null, null)).toBe(false);
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

describe("withOpacityColumn", () => {
  it("starts a new column at the range from 0 to the column's maximum", () => {
    expect(withOpacityColumn({ constant: { value: 0.5 } }, "qv", "t1")).toEqual(
      {
        constant: { value: 0.5 },
        source: "from",
        from: { column: "qv", range: [0, null] },
      },
    );
  });

  it("keeps the branch of the same column with its range", () => {
    const config = {
      source: "constant",
      constant: { value: 0.5 },
      from: { column: "qv", range: [2, 8] },
    } as OpacityConfig;
    expect(withOpacityColumn(config, "qv", "t1")).toEqual({
      ...config,
      source: "from",
    });
  });

  it("restarts another column at the range from 0", () => {
    expect(
      withOpacityColumn(
        { source: "from", from: { column: "qv", range: [2, 8] } },
        "area",
        "t1",
      ),
    ).toEqual({ source: "from", from: { column: "area", range: [0, null] } });
  });
});

describe("getFollower", () => {
  const gene = { column: "gene", columnType: "categorical" } as const;
  const qv = { column: "qv", columnType: "numerical" } as const;
  const notFollowing = () => {
    throw new Error("The property does not follow the column");
  };

  /**
   * @returns The configuration of a property that stops following the column
   * of the widget
   */
  const stopFollowing = <TConfig extends Config<string>>(
    config: TConfig,
    defaultConfig: TConfig,
  ) =>
    getFollower({ property: gene }, null, [])(
      "property",
      config,
      defaultConfig,
      notFollowing,
    );

  it("follows the new column with the named properties only", () => {
    const follow = getFollower({ color: gene, marker: gene, size: null }, qv, [
      "color",
    ]);
    const color: ColorConfig = { constant: { value: white } };
    expect(
      follow(
        "color",
        color,
        pointsDefaults.pointColor,
        ({ column, columnType }) =>
          withColorColumn(color, column, columnType, "t1"),
      ),
    ).toMatchObject({ source: "from", from: { column: "qv" } });
    const marker = {
      source: "groupBy",
      constant: { value: Marker.Cross },
      groupBy: { column: "gene", map: undefined },
    } as MarkerConfig;
    expect(
      follow("marker", marker, pointsDefaults.pointMarker, notFollowing),
    ).toMatchObject({ source: "constant" });
    const size: SizeConfig = { constant: { value: 3 } };
    expect(follow("size", size, pointsDefaults.pointSize, notFollowing)).toBe(
      size,
    );
  });

  it("does not count a property on another column as following", () => {
    const follow = getFollower({ color: gene, size: qv }, null, []);
    const size = { source: "from", from: { column: "qv" } } as SizeConfig;
    expect(follow("size", size, pointsDefaults.pointSize, notFollowing)).toBe(
      size,
    );
  });

  it("leaves values per category from the group table alone", () => {
    const color: ColorConfig = {
      source: "groupBy",
      groupBy: { column: "cluster", map: undefined },
    };
    const visibility = {
      source: "groupBy",
      groupBy: { column: "cluster", map: "m1" },
    } as VisibilityConfig;
    const previousFollowedColumns = {
      color: readFollowedColumn(color, "t1", ["categorical", "numerical"]),
      visibility: readFollowedColumn(visibility, "t1", ["numerical"]),
    };
    const withColumn = ({ column }: FollowedColumn) =>
      withFromColumn(visibility, column, "t1");
    // a new column, or none
    expect(
      getFollower(previousFollowedColumns, qv, ["color"])(
        "visibility",
        visibility,
        pointsDefaults.pointVisibility,
        withColumn,
      ),
    ).toBe(visibility);
    expect(
      getFollower(previousFollowedColumns, null, [])(
        "visibility",
        visibility,
        pointsDefaults.pointVisibility,
        withColumn,
      ),
    ).toBe(visibility);
  });

  it("returns to the constant, keeping the other branches", () => {
    expect(
      stopFollowing<MarkerConfig>(
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
      stopFollowing<SizeConfig>(
        { source: "from", from: { column: "area" } },
        pointsDefaults.pointSize,
      ),
    ).toMatchObject({
      source: "constant",
      constant: pointsDefaults.pointSize.constant,
    });
  });

  it("returns to the default's color source, keeping its branch", () => {
    expect(
      stopFollowing<ColorConfig>(
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
      stopFollowing<ColorConfig>(
        { groupBy: { column: "gene", map: undefined } },
        labelsDefaults.labelColor,
      ),
    ).toMatchObject({ source: "random", random: { palette: "distinct" } });
  });
});

describe("items", () => {
  const color = {
    name: "color",
    label: "",
    icon: null,
    columnTypes: ["categorical", "numerical"],
    isOn: true,
  } satisfies SettingsWidgetProperty;
  const marker = {
    name: "marker",
    label: "",
    icon: null,
    columnTypes: ["categorical"],
    isOn: true,
  } satisfies SettingsWidgetProperty;
  const size = {
    name: "size",
    label: "",
    icon: null,
    columnTypes: ["numerical"],
    isOn: false,
  } satisfies SettingsWidgetProperty;
  const fill = {
    name: "fill",
    label: "",
    icon: null,
    properties: [
      {
        name: "fillColor",
        label: "",
        icon: null,
        columnTypes: ["categorical", "numerical"],
        isOn: false,
      },
      {
        name: "fillOpacity",
        label: "",
        icon: null,
        columnTypes: ["numerical"],
        isOn: false,
      },
    ],
  } satisfies SettingsWidgetPart;

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
    } satisfies SettingsWidgetPart;
    expect(getToggledNames([fillOn, size], ["fill"], "numerical")).toEqual([
      "fillOpacity",
    ]);
    expect(isItemOn(fillOn)).toBe(true);
    expect(isItemSupported(fill, "categorical")).toBe(true);
    expect(isItemSupported(size, "categorical")).toBe(false);
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
