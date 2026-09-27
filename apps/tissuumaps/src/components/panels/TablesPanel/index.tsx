import { Trash2Icon } from "lucide-react";

import { type Table, createTable } from "@tissuumaps/core";

import {
  Accordion,
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  AccordionTrigger,
  AccordionTriggerDownUpIcon,
} from "@/components/common/accordion";
import { useConfirmDialog } from "@/components/dialogs/ConfirmDialog/hooks";
import { Button } from "@/components/ui/button";
import { AddDataObjectDialog } from "@/components/widgets/AddDataObjectDialog";
import { DataSourceWidget } from "@/components/widgets/DataSourceWidget";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

import { TableSettingsWidget } from "./TableSettingsWidget";

export type TablesPanelProps = {
  className?: string;
};

export function TablesPanel({ className }: TablesPanelProps) {
  const tableDataProviders = useAppStore((state) => state.tableDataProviders);

  const tables = useProjectStore((state) => state.tables);
  const addTable = useProjectStore((state) => state.addTable);

  return (
    <div className={cn("flex flex-col gap-y-2", className)}>
      <Accordion multiple className="gap-y-2">
        {tables.map((table) => (
          <TableAccordionItem key={table.id} table={table} />
        ))}
      </Accordion>
      <AddDataObjectDialog
        title="Add table"
        dataProviders={tableDataProviders}
        onAdd={(name, _layerId, dataSource) => {
          const table = createTable({
            id: crypto.randomUUID(),
            name,
            dataSource,
          });
          addTable(table);
        }}
      />
    </div>
  );
}

type TableAccordionItemProps = {
  table: Table;
};

function TableAccordionItem({ table }: TableAccordionItemProps) {
  const tableDataProviders = useAppStore((state) => state.tableDataProviders);

  const updateTable = useProjectStore((state) => state.updateTable);
  const deleteTable = useProjectStore((state) => state.deleteTable);
  const confirm = useConfirmDialog();

  return (
    <AccordionItem className="border rounded-md bg-sidebar p-2">
      <AccordionHeader>
        <div className="flex-1 w-full">
          <AccordionTrigger className="w-full cursor-pointer">
            {table.name}
          </AccordionTrigger>
        </div>
        <div className="ml-auto flex flex-row items-center gap-x-2">
          <Button
            variant="ghost"
            aria-label="Delete table"
            onClick={() => {
              void confirm({
                title: "Delete table",
                body: "Are you sure you want to delete this table? This action cannot be undone.",
              }).then((confirmed) => {
                if (confirmed) {
                  deleteTable(table.id);
                }
              });
            }}
            title="Delete table"
          >
            <Trash2Icon />
          </Button>
        </div>
        <AccordionTriggerDownUpIcon />
      </AccordionHeader>
      <AccordionPanel className="pt-2 flex flex-col gap-y-2">
        <DataSourceWidget
          dataSource={table.dataSource}
          dataProviders={tableDataProviders}
          onDataSourceChange={(newDataSource) => {
            updateTable(table.id, { dataSource: newDataSource });
          }}
          className="bg-card"
        />
        <TableSettingsWidget table={table} className="bg-card" />
      </AccordionPanel>
    </AccordionItem>
  );
}
