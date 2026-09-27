import { describe, expect, it } from "vitest";

import type {
  ConstantConfig,
  GroupByConfig,
  TableColumnRef,
} from "../model/configs";
import type { GroupValueMap } from "../model/primitives";
import { ConfigUtils } from "./ConfigUtils";
import { HashUtils } from "./HashUtils";

describe("ConfigUtils", () => {
  describe("findGroupByMap", () => {
    const maps: GroupValueMap<number>[] = [
      { id: "map1", name: "Map 1", values: { A: 1 } },
    ];

    it("returns the map of an active group-by source as it is", () => {
      const config: GroupByConfig<false> = {
        groupBy: { column: "cluster", map: "map1" },
      };

      expect(ConfigUtils.findGroupByMap(config, maps)).toBe(maps[0]);
    });

    it("returns undefined if group-by is not the active source", () => {
      const config: ConstantConfig<number> &
        Pick<GroupByConfig<false>, "groupBy"> = {
        source: "constant",
        constant: { value: 1 },
        groupBy: { column: "cluster", map: "map1" },
      };

      expect(ConfigUtils.findGroupByMap(config, maps)).toBeUndefined();
    });

    it("returns undefined if the map does not exist", () => {
      const config: GroupByConfig<false> = {
        groupBy: { column: "cluster", map: "missing" },
      };

      expect(ConfigUtils.findGroupByMap(config, maps)).toBeUndefined();
    });
  });

  describe("getGroupByMapIds", () => {
    it("returns the map IDs of the group-by configurations, whatever their active source", () => {
      const configs: (
        | ConstantConfig<number>
        | GroupByConfig<false>
        | (ConstantConfig<number> & Pick<GroupByConfig<false>, "groupBy">)
      )[] = [
        { groupBy: { column: "cluster", map: "colorMap" } },
        {
          source: "constant",
          constant: { value: 1 },
          groupBy: { column: "cluster", map: "opacityMap" },
        },
        { groupBy: { column: "cluster", map: undefined } },
        { constant: { value: 1 } },
      ];

      expect(ConfigUtils.getGroupByMapIds(configs)).toEqual(
        new Set(["colorMap", "opacityMap"]),
      );
    });
  });

  describe("createGroupValueGetter", () => {
    const map1: GroupValueMap<number> = {
      id: "map1",
      name: "Map 1",
      values: { A: 1 },
      default: 9,
    };
    const map2: GroupValueMap<number> = {
      id: "map2",
      name: "Map 2",
      values: { A: 1 },
    };
    const palette = [10, 20, 30];

    it("reads a group's value in the referenced map", () => {
      const config: GroupByConfig<false> = {
        groupBy: { column: "cluster", map: "map1" },
      };

      expect(
        ConfigUtils.createGroupValueGetter(config, map1, 0, palette)("A"),
      ).toBe(1);
    });

    it("falls back to the map's default, then to the default value", () => {
      const getValue = ConfigUtils.createGroupValueGetter(
        { groupBy: { column: "cluster", map: "map1" } },
        map1,
        0,
      );
      const getValueWithoutMapDefault = ConfigUtils.createGroupValueGetter(
        { groupBy: { column: "cluster", map: "map2" } },
        map2,
        0,
      );

      expect(getValue("B")).toBe(9);
      expect(getValueWithoutMapDefault("B")).toBe(0);
    });

    it("does not read inherited object properties as map values", () => {
      const getValue = ConfigUtils.createGroupValueGetter(
        { groupBy: { column: "cluster", map: "map2" } },
        map2,
        0,
      );

      expect(getValue("constructor")).toBe(0);
    });

    it("gives every group the default value if the map does not exist", () => {
      const getValue = ConfigUtils.createGroupValueGetter(
        { groupBy: { column: "cluster", map: "missing" } },
        undefined,
        0,
        palette,
      );

      expect(getValue("A")).toBe(0);
    });

    it("picks a palette value by hash without a map", () => {
      const getValue = ConfigUtils.createGroupValueGetter(
        { groupBy: { column: "cluster", map: undefined } },
        undefined,
        0,
        palette,
      );

      expect(getValue("A")).toBe(palette[HashUtils.hash("A") % palette.length]);
    });

    it("gives every group the default value without a map or palette", () => {
      const getValue = ConfigUtils.createGroupValueGetter(
        { groupBy: { column: "cluster", map: undefined } },
        undefined,
        0,
      );

      expect(getValue("A")).toBe(0);
    });
  });

  describe("withGroupByMap", () => {
    it("groups by the column with the map, keeping the other sources", () => {
      const config: ConstantConfig<boolean> = { constant: { value: false } };

      expect(
        ConfigUtils.withGroupByMap(config, { column: "cluster" }, "map1"),
      ).toEqual({
        constant: { value: false },
        source: "groupBy",
        groupBy: { column: "cluster", map: "map1" },
      });
    });

    it("keeps the extra fields of the group-by specification", () => {
      const config: GroupByConfig<false, { palette?: string }> = {
        groupBy: { column: "gene", map: undefined, palette: "viridis" },
      };

      expect(
        ConfigUtils.withGroupByMap(config, { column: "cluster" }, "map1"),
      ).toEqual({
        source: "groupBy",
        groupBy: { column: "cluster", map: "map1", palette: "viridis" },
      });
    });

    it("carries the unit of the active source over", () => {
      const config: ConstantConfig<number, { unit: "data" }> &
        GroupByConfig<false, { unit: "world" }> = {
        constant: { value: 1, unit: "data" },
        groupBy: { column: "cluster", map: "map1", unit: "world" },
      };

      expect(
        ConfigUtils.withGroupByMap(config, { column: "cluster" }, "map2")
          .groupBy,
      ).toEqual({
        column: "cluster",
        map: "map2",
        unit: "data",
      });
    });

    it("drops a group-by unit if the active source has none", () => {
      const config = {
        source: "constant" as const,
        constant: { value: 1 },
        groupBy: { column: "cluster", map: "map1", unit: "world" as const },
      };

      expect(
        ConfigUtils.withGroupByMap(config, { column: "cluster" }, "map2")
          .groupBy,
      ).not.toHaveProperty("unit", "world");
    });

    it("returns a configuration already grouping by the column with the map as is", () => {
      const config: GroupByConfig<true> = {
        groupBy: { column: "cluster", map: "map1" },
      };

      expect(
        ConfigUtils.withGroupByMap(config, { column: "cluster" }, "map1"),
      ).toBe(config);
    });
  });

  describe("getGroupByColumn", () => {
    it("returns the column of an active group-by source", () => {
      const config: GroupByConfig<false> = {
        groupBy: { column: "cluster", map: "map1" },
      };

      expect(ConfigUtils.getGroupByColumn(config)).toEqual({
        column: "cluster",
      });
    });

    it("returns undefined if group-by is not the active source", () => {
      const config: ConstantConfig<number> &
        Pick<GroupByConfig<false>, "groupBy"> = {
        source: "constant",
        constant: { value: 1 },
        groupBy: { column: "cluster", map: "map1" },
      };

      expect(ConfigUtils.getGroupByColumn(config)).toBeUndefined();
    });
  });

  describe("getUnit", () => {
    it("returns the unit of the active source", () => {
      const config: ConstantConfig<number, { unit: "data" }> &
        Pick<GroupByConfig<false, { unit: "world" }>, "groupBy"> = {
        source: "constant",
        constant: { value: 1, unit: "data" },
        groupBy: { column: "cluster", map: "map1", unit: "world" },
      };

      expect(ConfigUtils.getUnit(config)).toBe("data");
    });

    it("returns undefined if the active source has no unit", () => {
      const config: ConstantConfig<number> = { constant: { value: 1 } };

      expect(ConfigUtils.getUnit(config)).toBeUndefined();
    });
  });

  describe("getTableColumnRef", () => {
    it("keeps only the table and the column", () => {
      expect(
        ConfigUtils.getTableColumnRef({
          table: "cells",
          column: "cluster",
          map: "map1",
        } as TableColumnRef),
      ).toEqual({ table: "cells", column: "cluster" });
    });
  });

  describe("isSameTableColumn", () => {
    it("matches references to the same column of the same table", () => {
      expect(
        ConfigUtils.isSameTableColumn(
          { table: "cells", column: "cluster" },
          { table: "cells", column: "cluster" },
        ),
      ).toBe(true);
      expect(
        ConfigUtils.isSameTableColumn(
          { table: "cells", column: "cluster" },
          { table: "genes", column: "cluster" },
        ),
      ).toBe(false);
    });

    it("reads a reference without a table as one to the default table", () => {
      expect(
        ConfigUtils.isSameTableColumn(
          { column: "cluster" },
          { table: "cells", column: "cluster" },
          "cells",
        ),
      ).toBe(true);
    });
  });
});
