import type { ReactNode } from "react";

import type { Config, GroupByConfig, GroupValueMap } from "@tissuumaps/core";

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

  /** Shows a value in a cell of the group table */
  renderValue: (value: TValue) => ReactNode;

  /** Returns the values that a configuration without a map picks from */
  getPalette?: (
    config: Extract<TConfig, GroupByConfig<false>>,
  ) => readonly TValue[] | undefined;
};
