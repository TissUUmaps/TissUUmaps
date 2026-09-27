import { describe, expect, it, vi } from "vitest";

import type {
  ConstantConfig,
  FromConfig,
  GroupByConfig,
  GroupValueMap,
} from "@tissuumaps/core";

import type { GroupValuesAdapter } from "./adapter";
import { createGroupValues } from "./createGroupValues";
import type { GroupTableState } from "./useGroupTable";

type NumberConfig = ConstantConfig<number> | FromConfig | GroupByConfig<false>;

const groupTable: GroupTableState = {
  objectName: "cells",
  column: "cluster",
  setColumn: () => {},
  groupCounts: new Map([
    ["A", 2],
    ["B", 1],
  ]),
};

function createProperty(
  config: NumberConfig,
  maps: GroupValueMap<number>[] = [],
) {
  const adapter: GroupValuesAdapter<number, NumberConfig> = {
    maps,
    addMap: vi.fn(),
    updateMap: vi.fn(),
    columnSize: 60,
    renderCell: () => null,
    getPalette: () => [10, 20, 30],
  };
  return {
    name: "size",
    default: 1,
    config,
    onConfigChange: vi.fn(),
    adapter,
  };
}

describe("createGroupValues", () => {
  it("returns undefined while the table has no column or groups", () => {
    const property = createProperty({ constant: { value: 5 } });

    expect(
      createGroupValues({ ...groupTable, column: null }, property),
    ).toBeUndefined();
    expect(
      createGroupValues({ ...groupTable, groupCounts: null }, property),
    ).toBeUndefined();
  });

  it("merges an edit into the map the property is grouped by", () => {
    const map = { id: "map1", name: "Map 1", values: { A: 4, B: 6 } };
    const config: NumberConfig = {
      groupBy: { column: "cluster", map: "map1" },
    };
    const property = createProperty(config, [map]);

    const groupValues = createGroupValues(groupTable, property)!;
    groupValues.setValues({ A: 8 });

    expect(groupValues.getValue("A")).toBe(4);
    expect(groupValues.isInactive).toBe(false);
    expect(property.adapter.updateMap).toHaveBeenCalledWith("map1", {
      values: { A: 8, B: 6 },
    });
    expect(property.adapter.addMap).not.toHaveBeenCalled();
    expect(property.onConfigChange).toHaveBeenCalledWith(config);
  });

  it("writes every group's current value to a new map", () => {
    const property = createProperty({
      groupBy: { column: "cluster", map: undefined },
    });

    const groupValues = createGroupValues(groupTable, property)!;
    const valueB = groupValues.getValue("B");
    groupValues.setValues({ A: 8 });

    expect(property.adapter.addMap).toHaveBeenCalledWith({
      id: expect.any(String) as string,
      name: "cells cluster size",
      values: { A: 8, B: valueB },
    });
    const newMap = vi.mocked(property.adapter.addMap).mock.calls[0]![0];
    expect(property.onConfigChange).toHaveBeenCalledWith({
      source: "groupBy",
      groupBy: { column: "cluster", map: newMap.id },
    });
  });

  it("shows the value of a constant property for every group", () => {
    const property = createProperty({ constant: { value: 5 } });

    const groupValues = createGroupValues(groupTable, property)!;
    groupValues.setValues({ B: 7 });

    expect(groupValues.getValue("A")).toBe(5);
    expect(groupValues.isInactive).toBe(false);
    expect(property.adapter.addMap).toHaveBeenCalledWith(
      expect.objectContaining({ values: { A: 5, B: 7 } }),
    );
  });

  it("grays out a property that is neither constant nor grouped by the column", () => {
    const property = createProperty({ from: { column: "area" } });

    const groupValues = createGroupValues(groupTable, property)!;

    expect(groupValues.getValue("A")).toBe(1);
    expect(groupValues.isInactive).toBe(true);
  });
});
