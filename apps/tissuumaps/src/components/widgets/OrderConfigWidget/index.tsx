import { TableColumnField } from "@/components/controls/TableColumnField";

import type { OrderConfigWidgetAdapter } from "./adapter";

export { ActiveOrderConfigValue } from "./ActiveOrderConfigValue";
export { OrderConfigSourceToggleGroup } from "./OrderConfigSourceToggleGroup";

export type OrderConfigWidgetProps = {
  adapter: OrderConfigWidgetAdapter;
  className?: string;
};

export function OrderConfigWidget({
  adapter,
  className,
}: OrderConfigWidgetProps) {
  switch (adapter.currentSource) {
    case "none":
      return null;
    case "from":
      return <FromOrderConfigWidget adapter={adapter} className={className} />;
  }
}

type FromOrderConfigWidgetProps = {
  adapter: OrderConfigWidgetAdapter;
  className?: string;
};

function FromOrderConfigWidget({
  adapter,
  className,
}: FromOrderConfigWidgetProps) {
  const {
    tableId,
    currentFromTableColumn: from,
    setCurrentFromTableColumn: setFrom,
  } = adapter;

  return (
    <div className={className}>
      <TableColumnField
        label="Column"
        tableId={tableId}
        value={from}
        onValueChange={setFrom}
      />
    </div>
  );
}
