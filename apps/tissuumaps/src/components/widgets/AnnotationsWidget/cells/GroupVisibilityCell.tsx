import { EyeIcon, EyeOffIcon } from "lucide-react";
import { useEffect, useState } from "react";

import type { HighlightedItemGroup } from "@tissuumaps/core";

import { Button } from "@/components/ui/button";
import { useAppStore } from "@/stores/app";

export type GroupVisibilityCellProps = {
  visible: boolean;
  onVisibleChange: (visible: boolean) => void;

  /** The group to show alone in the viewer while the button is hovered */
  itemGroup?: HighlightedItemGroup;
};

export function GroupVisibilityCell({
  visible,
  onVisibleChange,
  itemGroup,
}: GroupVisibilityCellProps) {
  const setHighlightedItemGroup = useAppStore(
    (state) => state.setHighlightedItemGroup,
  );

  const [hovered, setHovered] = useState(false);

  const annotatedObject = itemGroup?.annotatedObject;
  const groupByTable = itemGroup?.groupBy.table;
  const groupByColumn = itemGroup?.groupBy.column;
  const group = itemGroup?.group;

  // a row that is scrolled out or filtered away unmounts without a pointer leave
  useEffect(() => {
    if (
      !hovered ||
      annotatedObject === undefined ||
      groupByColumn === undefined ||
      group === undefined
    ) {
      return;
    }
    setHighlightedItemGroup({
      annotatedObject,
      groupBy: { table: groupByTable, column: groupByColumn },
      group,
    });
    return () => {
      setHighlightedItemGroup(null);
    };
  }, [
    hovered,
    annotatedObject,
    groupByTable,
    groupByColumn,
    group,
    setHighlightedItemGroup,
  ]);

  return (
    <Button
      variant="ghost"
      size="icon-xs"
      title="Toggle group visibility"
      aria-label={visible ? "Hide group" : "Show group"}
      onClick={() => {
        setHovered(false);
        onVisibleChange(!visible);
      }}
      onPointerEnter={() => {
        setHovered(true);
      }}
      onPointerLeave={() => {
        setHovered(false);
      }}
    >
      {visible ? <EyeIcon /> : <EyeOffIcon />}
    </Button>
  );
}
