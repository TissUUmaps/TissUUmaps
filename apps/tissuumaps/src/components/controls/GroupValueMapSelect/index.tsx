import type { GroupValueMap } from "@tissuumaps/core";

import { SimpleSelect } from "@/components/common/simple-select";
import { useConfirmDialog } from "@/components/dialogs/ConfirmDialog/hooks";

export type GroupValueMapSelectProps = {
  maps: GroupValueMap<unknown>[];
  isMapDeletable?: (map: GroupValueMap<unknown>) => boolean;
  value: string | null;
  onValueChange: (mapId: string | null) => void;
  onMapDelete: (mapId: string) => void;
};

export function GroupValueMapSelect({
  maps,
  isMapDeletable,
  value,
  onValueChange,
  onMapDelete,
}: GroupValueMapSelectProps) {
  const confirm = useConfirmDialog();

  return (
    <SimpleSelect
      items={maps}
      itemLabel={(map) => map.name}
      itemValue={(map) => map.id}
      value={value}
      onValueChange={onValueChange}
      nullable
      isItemDeletable={isMapDeletable}
      onItemDelete={(map) => {
        void confirm({
          title: "Delete map",
          body: `Are you sure you want to delete the map "${map.name}"? This action cannot be undone.`,
        }).then((confirmed) => {
          if (confirmed) {
            if (map.id === value) {
              onValueChange(null);
            }
            onMapDelete(map.id);
          }
        });
      }}
    />
  );
}
