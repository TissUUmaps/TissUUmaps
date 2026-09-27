import { Columns3Icon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

export type GroupColumnPickerProps = {
  columns: { id: string; header: string; isShown: boolean }[];
  onShownChange: (id: string, isShown: boolean) => void;
};

export function GroupColumnPicker({
  columns,
  onShownChange,
}: GroupColumnPickerProps) {
  return (
    <Popover>
      <PopoverTrigger
        render={
          <Button
            variant="ghost"
            size="icon-sm"
            title="Show columns"
            aria-label="Show columns"
          />
        }
      >
        <Columns3Icon />
      </PopoverTrigger>
      <PopoverContent align="end" className="w-48 gap-2 p-2">
        {columns.map((column) => (
          <Label key={column.id} className="font-normal">
            <Checkbox
              checked={column.isShown}
              onCheckedChange={(checked) => {
                onShownChange(column.id, checked);
              }}
            />
            {column.header}
          </Label>
        ))}
      </PopoverContent>
    </Popover>
  );
}
