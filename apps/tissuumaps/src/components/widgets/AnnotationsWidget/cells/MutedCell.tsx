import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

export type MutedCellProps = {
  /** Whether the cell is grayed out, as editing it changes the property source */
  isMuted: boolean;
  children: ReactNode;
};

export function MutedCell({ isMuted, children }: MutedCellProps) {
  // the wrapper stays when the cell turns active, so that the input or picker
  // that turned it active is not remounted
  return (
    <div
      className={cn("flex w-full", isMuted && "opacity-50")}
      title={isMuted ? "Edit to group by this column" : undefined}
    >
      {children}
    </div>
  );
}
