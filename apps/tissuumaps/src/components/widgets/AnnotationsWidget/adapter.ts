import type { ReactNode } from "react";

import type { Config, GroupByConfig, GroupValueMap } from "@tissuumaps/core";

/** Initial width of a column of markers, colors or eye buttons, in pixels */
export const defaultGroupColumnSize = 60;

/** Initial width of a column of number inputs, in pixels */
export const defaultNumericGroupColumnSize = 90;

/**
 * What the group table needs to know about one type of group value
 *
 * Each value type has its own hook that returns its adapter (e.g.
 * `useColorGroupValues`), so that the shared group table hooks never name the
 * value types.
 */
export type GroupValuesAdapter<TValue, TConfig extends Config<string>> = {
  /** The project's maps of this value type */
  maps: GroupValueMap<TValue>[];

  addMap: (map: GroupValueMap<TValue>) => void;

  updateMap: (
    mapId: string,
    updates: Partial<Omit<GroupValueMap<TValue>, "id">>,
  ) => void;

  /** Initial width of the value's column, in pixels */
  columnSize: number;

  /** The value that the rows sort by; the column is not sortable without */
  getSortValue?: (value: TValue) => number | string;

  renderCell: (
    value: TValue,
    onValueChange: (value: TValue) => void,
  ) => ReactNode;

  /** Returns the values that a configuration without a map picks from */
  getPalette?: (
    config: Extract<TConfig, GroupByConfig<false>>,
  ) => readonly TValue[] | undefined;
};
