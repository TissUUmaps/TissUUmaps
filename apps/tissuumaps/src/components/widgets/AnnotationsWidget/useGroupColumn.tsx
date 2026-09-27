import { useMemo } from "react";

import type { Config, GroupByConfig } from "@tissuumaps/core";

import { useLatestCallback } from "@/hooks/useLatestCallback";

import type { GroupAnnotationsTableColumnDef } from "./GroupAnnotationsTable";
import type { GroupValuesAdapter } from "./adapter";
import { MutedCell } from "./cells/MutedCell";
import { createGroupValues } from "./createGroupValues";
import { isGroupedByColumn } from "./isGroupedByColumn";
import type { GroupTableState } from "./useGroupTable";

/** A property of an annotated object that can take a value per group */
export type GroupProperty<TValue, TConfig extends Config<string>> = {
  /** The name of the property, as a column header and in map names */
  name: string;

  /**
   * Whether the column is shown by default even while the property is not
   * grouped by the table's column
   */
  isShownByDefault?: boolean;

  /** The value of a group that nothing assigns one to */
  default: NoInfer<TValue>;

  /** The configuration of the property */
  config: NoInfer<TConfig>;

  /** Called with the configuration pointed at the map that an edit wrote */
  onConfigChange: (config: GroupByConfig<true>) => void;

  /** The adapter of the property's value type, which also sets `TValue` */
  adapter: GroupValuesAdapter<TValue, TConfig>;
};

/**
 * Returns the group table column of a property, showing and editing the value
 * of each group
 *
 * @param groupTable - The state of the group table
 * @param property - The property
 * @returns The column, or `undefined` while the table has no column or groups
 */
export function useGroupColumn<TValue, TConfig extends Config<string>>(
  groupTable: GroupTableState,
  property: GroupProperty<TValue, TConfig>,
): GroupAnnotationsTableColumnDef | undefined {
  const {
    name,
    isShownByDefault,
    default: defaultValue,
    config,
    adapter,
  } = property;
  const onConfigChange = useLatestCallback(property.onConfigChange);

  return useMemo(() => {
    const groupValues = createGroupValues(groupTable, {
      name,
      default: defaultValue,
      config,
      onConfigChange,
      adapter,
    });
    if (groupValues === undefined) {
      return undefined;
    }
    const { getValue, setValues, isInactive } = groupValues;
    const { getSortValue } = adapter;
    return {
      id: name,
      header: name.charAt(0).toUpperCase() + name.slice(1),
      size: adapter.columnSize,
      meta: {
        isShownByDefault:
          isShownByDefault === true || isGroupedByColumn(config, groupTable),
      },
      ...(getSortValue !== undefined && {
        accessorFn: (row) => getSortValue(getValue(row.group)),
      }),
      cell: ({ row }) => (
        <MutedCell isMuted={isInactive}>
          {adapter.renderCell(getValue(row.original.group), (value) =>
            setValues({ [row.original.group]: value }),
          )}
        </MutedCell>
      ),
    };
  }, [
    groupTable,
    name,
    isShownByDefault,
    defaultValue,
    config,
    onConfigChange,
    adapter,
  ]);
}
