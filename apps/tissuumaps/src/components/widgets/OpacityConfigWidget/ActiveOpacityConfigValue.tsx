import {
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
} from "@tissuumaps/core";

import { formatTableColumn } from "@/components/controls/TableColumnField/formatTableColumn";
import { percentFormat } from "@/lib/format";
import { useProjectStore } from "@/stores/project";

import type { OpacityConfigWidgetAdapter } from "./adapter";

export type ActiveOpacityConfigValueProps = {
  adapter: OpacityConfigWidgetAdapter;
  className?: string;
};

export function ActiveOpacityConfigValue({
  adapter,
  className,
}: ActiveOpacityConfigValueProps) {
  const { activeSource, opacityConfig, defaultOpacity, tableId } = adapter;

  const tables = useProjectStore((state) => state.tables);

  if (activeSource === "constant" && isConstantConfig(opacityConfig)) {
    return (
      <div className={className}>
        {percentFormat.format(opacityConfig.constant.value)}
      </div>
    );
  }

  if (
    activeSource === "from" &&
    isFromConfig(opacityConfig) &&
    tableId !== null
  ) {
    return (
      <div className={className}>
        {formatTableColumn(opacityConfig.from, tables)}
      </div>
    );
  }

  if (
    activeSource === "groupBy" &&
    isGroupByConfig(opacityConfig) &&
    tableId !== null
  ) {
    return (
      <div className={className}>
        {formatTableColumn(opacityConfig.groupBy, tables)}
      </div>
    );
  }

  return (
    <div className={className}>{percentFormat.format(defaultOpacity)}</div>
  );
}
