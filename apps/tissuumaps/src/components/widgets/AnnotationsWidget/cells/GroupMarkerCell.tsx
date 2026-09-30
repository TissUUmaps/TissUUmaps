import type { Marker } from "@tissuumaps/core";

import { SimpleMarkerPicker } from "@/components/common/simple-marker-picker";
import { markers } from "@/components/markers";

export type GroupMarkerCellProps = {
  marker: Marker;
  onMarkerChange: (marker: Marker) => void;
};

export function GroupMarkerCell({
  marker,
  onMarkerChange,
}: GroupMarkerCellProps) {
  return (
    <SimpleMarkerPicker
      marker={marker}
      onMarkerChange={onMarkerChange}
      className="size-6"
    >
      {markers.find((m) => m.value === marker)?.icon}
    </SimpleMarkerPicker>
  );
}
