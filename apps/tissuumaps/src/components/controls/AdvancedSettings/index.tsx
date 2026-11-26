import type { ReactNode } from "react";

import {
  Collapsible,
  CollapsiblePanel,
  CollapsibleTrigger,
  CollapsibleTriggerRightDownIcon,
} from "@/components/common/collapsible";
import { cn } from "@/lib/utils";

export type AdvancedSettingsProps = {
  /** Called when the section is opened or closed */
  onOpenChange?: (open: boolean) => void;
  children: ReactNode;
  className?: string;
};

/** The settings of an object beyond its style, collapsed by default */
export function AdvancedSettings({
  onOpenChange,
  children,
  className,
}: AdvancedSettingsProps) {
  return (
    <Collapsible
      onOpenChange={onOpenChange}
      className={cn("flex flex-col gap-y-2", className)}
    >
      <div className="flex flex-row items-center gap-x-1 text-sm">
        <CollapsibleTriggerRightDownIcon />
        <CollapsibleTrigger className="font-medium">
          Advanced
        </CollapsibleTrigger>
      </div>
      <CollapsiblePanel className="flex flex-col gap-y-2">
        {children}
      </CollapsiblePanel>
    </Collapsible>
  );
}
