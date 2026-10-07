import { PlusIcon } from "lucide-react";

import type { DataSource } from "@tissuumaps/core";

import type { AddDataObjectDialogParams } from "@/components/dialogs/AddDataObjectDialog";
import { useAddDataObjectDialog } from "@/components/dialogs/AddDataObjectDialog/hooks";
import { Button } from "@/components/ui/button";
import { useProjectStore } from "@/stores/project";

export type AddDataObjectButtonProps<TDataSource extends DataSource> =
  AddDataObjectDialogParams<TDataSource>;

/**
 * A button that opens a dialog for adding a data object (see
 * `useAddDataObjectDialog`)
 *
 * Disabled while a data object that is added to a layer has no layer to go to.
 */
export function AddDataObjectButton<TDataSource extends DataSource>(
  props: AddDataObjectButtonProps<TDataSource>,
) {
  const hasLayers = useProjectStore((state) => state.layers.length > 0);
  const addDataObject = useAddDataObjectDialog();

  if (props.dataProviders.size === 0) {
    return null;
  }

  const disabled = props.withLayer && !hasLayers;
  return (
    <span title={disabled ? "Add a layer first" : undefined}>
      <Button
        variant="outline"
        className="w-full"
        disabled={disabled}
        onClick={() => addDataObject(props)}
      >
        <PlusIcon className="size-4" />
        Add
      </Button>
    </span>
  );
}
