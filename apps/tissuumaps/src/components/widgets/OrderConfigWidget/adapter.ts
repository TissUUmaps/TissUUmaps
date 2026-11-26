import type { OrderConfig, TableColumnRef } from "@tissuumaps/core";

export type OrderConfigSource =
  "none" | Exclude<OrderConfig["source"], undefined>;

export type OrderConfigWidgetAdapter = {
  orderConfig: OrderConfig | undefined;
  tableId: string | null;
  activeSource: OrderConfigSource;
  currentSource: OrderConfigSource;
  currentFromTableColumn: TableColumnRef | null;
  setCurrentSource: (newCurrentSource: OrderConfigSource) => void;
  setCurrentFromTableColumn: (
    newCurrentFromTableColumn: TableColumnRef | null,
  ) => void;
};
