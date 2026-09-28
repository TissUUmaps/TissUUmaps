import type { ReactNode } from "react";

import {
  Collapsible,
  CollapsiblePanel,
  CollapsibleTrigger,
  CollapsibleTriggerRightDownIcon,
} from "@/components/common/collapsible";
import { cn } from "@/lib/utils";

export type SectionProps = {
  title: string;
  summary?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
  className?: string;
};

export function Section({
  title,
  summary,
  defaultOpen = false,
  children,
  className,
}: SectionProps) {
  return (
    <Collapsible
      defaultOpen={defaultOpen}
      className={cn("border-t", className)}
    >
      <h3 className="text-muted-foreground flex h-10 items-center gap-1 px-1">
        <CollapsibleTriggerRightDownIcon />
        <CollapsibleTrigger className="group/section-trigger hover:text-foreground min-w-0 flex-1 gap-2 self-stretch">
          <span className="text-xs font-semibold tracking-wider uppercase">
            {title}
          </span>
          {summary !== undefined && (
            <span className="ml-auto truncate pr-1 text-xs group-aria-expanded/section-trigger:hidden">
              {summary}
            </span>
          )}
        </CollapsibleTrigger>
      </h3>
      <CollapsiblePanel className="px-1 pb-3">{children}</CollapsiblePanel>
    </Collapsible>
  );
}
