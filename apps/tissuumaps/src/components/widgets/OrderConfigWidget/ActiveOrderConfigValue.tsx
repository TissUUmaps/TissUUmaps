import { formatTableColumn } from "@/components/controls/TableColumnField/formatTableColumn";
import { useProjectStore } from "@/stores/project";

import type { OrderConfigWidgetAdapter } from "./adapter";

export type ActiveOrderConfigValueProps = {
  adapter: OrderConfigWidgetAdapter;
  className?: string;
};

export function ActiveOrderConfigValue({
  adapter,
  className,
}: ActiveOrderConfigValueProps) {
  const { orderConfig, tableId } = adapter;

  const tables = useProjectStore((state) => state.tables);

  if (orderConfig !== undefined && tableId !== null) {
    return (
      <div className={className}>
        {formatTableColumn(orderConfig.from, tables)}
      </div>
    );
  }

  return <div className={className}>None</div>;
}
