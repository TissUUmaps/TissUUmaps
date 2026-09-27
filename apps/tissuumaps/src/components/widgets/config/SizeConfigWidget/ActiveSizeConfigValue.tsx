import {
  isConstantConfig,
  isFromConfig,
  isGroupByConfig,
} from "@tissuumaps/core";

import { formatTableColumn } from "@/components/widgets/TableColumnField/formatTableColumn";
import { useProjectStore } from "@/stores/project";

import type { SizeConfigWidgetAdapter } from "./adapter";

export type ActiveSizeConfigValueProps = {
  adapter: SizeConfigWidgetAdapter;
  className?: string;
};

export function ActiveSizeConfigValue({
  adapter,
  className,
}: ActiveSizeConfigValueProps) {
  const { activeSource, sizeConfig, defaultSize, tableId } = adapter;

  const tables = useProjectStore((state) => state.tables);

  if (activeSource === "constant" && isConstantConfig(sizeConfig)) {
    return <div className={className}>{sizeConfig.constant.value}</div>;
  }

  if (activeSource === "from" && isFromConfig(sizeConfig) && tableId !== null) {
    return (
      <div className={className}>
        {formatTableColumn(sizeConfig.from, tables)}
      </div>
    );
  }

  if (
    activeSource === "groupBy" &&
    isGroupByConfig(sizeConfig) &&
    tableId !== null
  ) {
    return (
      <div className={className}>
        {formatTableColumn(sizeConfig.groupBy, tables)}
      </div>
    );
  }

  return <div className={className}>{defaultSize}</div>;
}
