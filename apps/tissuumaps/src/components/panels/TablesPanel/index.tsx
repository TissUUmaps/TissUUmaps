import type { Table } from "@tissuumaps/core";

import { AddDataObjectButton } from "@/components/controls/AddDataObjectButton";
import { ObjectList, ObjectListItem } from "@/components/controls/ObjectList";
import { DataSourceWidget } from "@/components/widgets/DataSourceWidget";
import { addTableDataObject } from "@/data/io/dataObject";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

export type TablesPanelProps = {
  className?: string;
};

export function TablesPanel({ className }: TablesPanelProps) {
  const expandedTableIds = useAppStore((state) => state.expandedTableIds);
  const setExpandedTableIds = useAppStore((state) => state.setExpandedTableIds);
  const tableDataProviders = useAppStore((state) => state.tableDataProviders);

  const tables = useProjectStore((state) => state.tables);

  return (
    <div className={cn("flex flex-col gap-y-2", className)}>
      <ObjectList
        expandedIds={expandedTableIds}
        onExpandedIdsChange={setExpandedTableIds}
      >
        {tables.map((table) => (
          <TableAccordionItem key={table.id} table={table} />
        ))}
      </ObjectList>
      <AddDataObjectButton
        title="Add table"
        dataProviders={tableDataProviders}
        onAdd={(name, _layerId, origSource, preparedDataSource) =>
          addTableDataObject(name, origSource, preparedDataSource)
        }
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

  return (
    <ObjectListItem
      id={table.id}
      name={table.name}
      objectLabel="table"
      onRename={(name) => updateTable(table.id, { name })}
      onDelete={() => deleteTable(table.id)}
    >
      <DataSourceWidget
        dataSource={table.dataSource}
        dataProviders={tableDataProviders}
        onDataSourceChange={(newDataSource) => {
          updateTable(table.id, { dataSource: newDataSource });
        }}
        className="bg-card"
      />
    </ObjectListItem>
  );
}
