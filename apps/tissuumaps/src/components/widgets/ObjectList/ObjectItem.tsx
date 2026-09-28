import type { ReactNode, Ref } from "react";

import {
  AccordionHeader,
  AccordionItem,
  AccordionPanel,
  AccordionTrigger,
  AccordionTriggerRightDownIcon,
} from "@/components/common/accordion";
import { cn } from "@/lib/utils";

import { ObjectMenu } from "./ObjectMenu";

export type ObjectItemProps = {
  id: string;
  name: string;
  objectLabel: string;
  dimmed?: boolean;
  handle?: ReactNode;
  leadingControls?: ReactNode;
  trailingControls?: ReactNode;
  onRename: (name: string) => void;
  deleteDisabledReason?: string;
  onDelete: () => void;
  children: ReactNode;
  ref?: Ref<HTMLDivElement>;
  className?: string;
};

export function ObjectItem({
  id,
  name,
  objectLabel,
  dimmed = false,
  handle,
  leadingControls,
  trailingControls,
  onRename,
  deleteDisabledReason,
  onDelete,
  children,
  ref,
  className,
}: ObjectItemProps) {
  return (
    <div ref={ref}>
      <AccordionItem
        value={id}
        className={cn(
          "data-open:bg-muted/40 data-open:border-border rounded-lg border border-transparent",
          className,
        )}
      >
        <AccordionHeader className="hover:bg-muted/60 h-9 gap-0.5 rounded-lg px-0.5">
          {handle}
          {leadingControls}
          <AccordionTrigger
            className={cn(
              "min-w-0 flex-1 self-stretch pl-1 text-left text-sm",
              dimmed && "text-muted-foreground",
            )}
          >
            <span className="truncate">{name || "Untitled"}</span>
          </AccordionTrigger>
          {trailingControls}
          <ObjectMenu
            name={name}
            objectLabel={objectLabel}
            onRename={onRename}
            deleteDisabledReason={deleteDisabledReason}
            onDelete={onDelete}
          />
          <AccordionTriggerRightDownIcon className="text-muted-foreground hover:bg-muted hover:text-foreground flex size-8 shrink-0 items-center justify-center rounded-md" />
        </AccordionHeader>
        <AccordionPanel className="flex flex-col gap-y-2 px-2 pt-1 pb-2 text-sm">
          {children}
        </AccordionPanel>
      </AccordionItem>
    </div>
  );
}
