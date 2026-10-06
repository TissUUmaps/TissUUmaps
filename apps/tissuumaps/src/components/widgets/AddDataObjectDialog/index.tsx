import { JsonForms } from "@jsonforms/react";
import { PlusIcon } from "lucide-react";
import { useCallback, useState } from "react";

import type { Data, DataProvider, DataSource, Layer } from "@tissuumaps/core";

import { Field, FieldLabel } from "@/components/common/field";
import { Fieldset } from "@/components/common/fieldset";
import { SimpleSelect } from "@/components/common/simple-select";
import { cells } from "@/components/jsonforms/cells";
import { renderers } from "@/components/jsonforms/renderers";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";

export type AddDataObjectDialogProps<
  TDataSource extends DataSource = DataSource,
> = {
  title: string;
  layers?: Layer[];
  dataProviders: Map<string, DataProvider<TDataSource, Data>>;
  onAdd: (
    name: string,
    layerId: string | undefined,
    dataSource: TDataSource,
  ) => void;
};

export function AddDataObjectDialog<TDataSource extends DataSource>({
  title,
  layers,
  dataProviders,
  onAdd,
}: AddDataObjectDialogProps<TDataSource>) {
  const providerEntries = Array.from(dataProviders.entries());
  const requiresLayer = layers !== undefined;

  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [selectedLayerId, setSelectedLayerId] = useState(layers?.[0]?.id ?? "");
  const [selectedType, setSelectedType] = useState(
    providerEntries[0]?.[0] ?? "",
  );
  const [dataSourceDraft, setDataSourceDraft] = useState<TDataSource>({
    type: selectedType,
  } as TDataSource);

  const resetForm = useCallback(() => {
    const defaultType = providerEntries[0]?.[0] ?? "";
    setName("");
    setSelectedLayerId(layers?.[0]?.id ?? "");
    setSelectedType(defaultType);
    setDataSourceDraft({ type: defaultType } as TDataSource);
  }, [providerEntries, layers]);

  const handleTypeChange = useCallback((value: string | null) => {
    if (value == null) return;
    setSelectedType(value);
    setDataSourceDraft({ type: value } as TDataSource);
  }, []);

  const handleAdd = useCallback(() => {
    if (requiresLayer && !selectedLayerId) return;
    onAdd(
      name.trim() || "Untitled",
      selectedLayerId || undefined,
      dataSourceDraft,
    );
    setOpen(false);
    resetForm();
  }, [name, requiresLayer, selectedLayerId, dataSourceDraft, onAdd, resetForm]);

  const selectedProvider = dataProviders.get(selectedType);

  if (providerEntries.length === 0) {
    return null;
  }

  const addDisabled = requiresLayer && !selectedLayerId;
  const triggerDisabled = requiresLayer && layers.length === 0;

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <span title={triggerDisabled ? "Add a layer first" : undefined}>
        <DialogTrigger
          render={
            <Button
              variant="outline"
              className="w-full"
              disabled={triggerDisabled}
            />
          }
          onClick={() => resetForm()}
        >
          <PlusIcon className="size-4" />
          Add
        </DialogTrigger>
      </span>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>

        <Fieldset className="flex flex-col gap-4">
          <Field className="flex flex-col gap-2">
            <FieldLabel>Name</FieldLabel>
            <Input
              type="text"
              placeholder="Enter a name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </Field>

          {layers !== undefined && layers.length > 0 && (
            <Field className="flex flex-col gap-2">
              <FieldLabel>Layer</FieldLabel>
              <SimpleSelect
                items={layers}
                itemLabel={(layer) => layer.name}
                itemValue={(layer) => layer.id}
                value={selectedLayerId}
                onValueChange={(value) => {
                  if (value !== null) {
                    setSelectedLayerId(value);
                  }
                }}
              />
            </Field>
          )}

          {providerEntries.length >= 1 && (
            <Field className="flex flex-col gap-2">
              <FieldLabel>Source type</FieldLabel>
              <SimpleSelect
                items={providerEntries}
                itemLabel={([, provider]) => provider.name}
                itemValue={([type]) => type}
                value={selectedType}
                onValueChange={handleTypeChange}
              />
            </Field>
          )}

          {selectedProvider && (
            <Field className="flex flex-col gap-2">
              <FieldLabel>Configuration</FieldLabel>
              <JsonForms
                data={dataSourceDraft}
                onChange={({ data }) => setDataSourceDraft(data as TDataSource)}
                schema={selectedProvider.schema}
                uischema={selectedProvider.uischema}
                renderers={renderers}
                cells={cells}
              />
            </Field>
          )}
        </Fieldset>

        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>
            Cancel
          </DialogClose>
          <Button onClick={handleAdd} disabled={addDisabled}>
            Add
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
