import { OpacityControl } from "@/components/common/opacity-control";

export type GroupOpacityCellProps = {
  opacity: number;
  onOpacityChange: (opacity: number) => void;
};

export function GroupOpacityCell({
  opacity,
  onOpacityChange,
}: GroupOpacityCellProps) {
  return <OpacityControl opacity={opacity} onOpacityCommit={onOpacityChange} />;
}
