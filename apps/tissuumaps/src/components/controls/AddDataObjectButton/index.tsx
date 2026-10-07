import { JsonForms } from "@jsonforms/react";
import { PlusIcon } from "lucide-react";
import { useState } from "react";

import type { Data, DataProvider, DataSource, Layer } from "@tissuumaps/core";

import { Field, FieldLabel } from "@/components/common/field";
import { Fieldset } from "@/components/common/fieldset";
import { SimpleSelect } from "@/components/common/simple-select";
import { SourceField } from "@/components/controls/SourceField";
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
import { usePrepareDataSource } from "@/data/hooks/usePrepareDataSource";

export type AddDataObjectButtonProps<
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

/**
 * A button that opens a dialog for adding a data object
 *
 * The data source is prepared before the data object is added (see
 * `DataProvider.prepareDataSource`). Closing the dialog cancels a pending add.
 */
export function AddDataObjectButton<TDataSource extends DataSource>({
  title,
  layers,
  dataProviders,
  onAdd,
}: AddDataObjectButtonProps<TDataSource>) {
  const providerEntries = Array.from(dataProviders.entries());
  const defaultType = providerEntries[0]?.[0] ?? "";
  const defaultLayerId = layers?.[0]?.id ?? "";

  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [layerId, setLayerId] = useState(defaultLayerId);
  const [draft, setDraft] = useState({ type: defaultType } as TDataSource);

  const { isPreparing, prepare, cancel } = usePrepareDataSource();

  const dataProvider = dataProviders.get(draft.type);
  const requiresLayer = layers !== undefined;
  const triggerDisabled = requiresLayer && layers.length === 0;

  const resetForm = () => {
    setName("");
    setLayerId(defaultLayerId);
    setDraft({ type: defaultType } as TDataSource);
  };

  const add = async () => {
    const preparedDataSource = await prepare(draft, dataProvider);
    if (preparedDataSource !== undefined) {
      onAdd(
        name.trim() || "Untitled",
        layerId || undefined,
        preparedDataSource,
      );
      setOpen(false);
    }
  };

  if (providerEntries.length === 0) {
    return null;
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(newOpen) => {
        if (!newOpen) {
          cancel();
        }
        setOpen(newOpen);
      }}
    >
      <span title={triggerDisabled ? "Add a layer first" : undefined}>
        <DialogTrigger
          render={
            <Button
              variant="outline"
              className="w-full"
              disabled={triggerDisabled}
            />
          }
          onClick={resetForm}
        >
          <PlusIcon className="size-4" />
          Add
        </DialogTrigger>
      </span>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>

        <Fieldset disabled={isPreparing} className="flex flex-col gap-4">
          <Field className="flex flex-col gap-2">
            <FieldLabel>Name</FieldLabel>
            <Input
              type="text"
              placeholder="Enter a name"
              disabled={isPreparing}
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </Field>

          {layers !== undefined && layers.length > 0 && (
            <Field className="flex flex-col gap-2">
              <FieldLabel>Layer</FieldLabel>
              <SimpleSelect
                items={layers}
                itemLabel={(layer) => layer.name}
                itemValue={(layer) => layer.id}
                value={layerId}
                disabled={isPreparing}
                onValueChange={(value) => {
                  if (value !== null) {
                    setLayerId(value);
                  }
                }}
              />
            </Field>
          )}

          <Field className="flex flex-col gap-2">
            <FieldLabel>Source type</FieldLabel>
            <SimpleSelect
              items={providerEntries}
              itemLabel={([, provider]) => provider.name}
              itemValue={([type]) => type}
              value={draft.type}
              disabled={isPreparing}
              onValueChange={(type) => {
                if (type !== null) {
                  const dataProvider = dataProviders.get(type);
                  setDraft((draft) =>
                    draft.source !== undefined &&
                    dataProvider?.schema.properties?.source !== undefined
                      ? ({ type, source: draft.source } as TDataSource)
                      : ({ type } as TDataSource),
                  );
                }
              }}
            />
          </Field>

          {dataProvider !== undefined && (
            <SourceField
              schema={dataProvider.schema}
              value={draft.source}
              onValueChange={(source) =>
                setDraft((draft) => {
                  const newDraft = { ...draft, source };
                  if (source === undefined) {
                    delete newDraft.source;
                  }
                  return newDraft;
                })
              }
              disabled={isPreparing}
            />
          )}

          {dataProvider !== undefined &&
            (!("elements" in dataProvider.uischema) ||
              dataProvider.uischema.elements.length > 0) && (
              <Field className="flex flex-col gap-2">
                <FieldLabel>Configuration</FieldLabel>
                <JsonForms
                  data={draft}
                  onChange={({ data }) => setDraft(data as TDataSource)}
                  schema={dataProvider.schema}
                  uischema={dataProvider.uischema}
                  renderers={renderers}
                  cells={cells}
                  readonly={isPreparing}
                />
              </Field>
            )}
        </Fieldset>

        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>
            Cancel
          </DialogClose>
          <Button
            onClick={() => void add()}
            disabled={(requiresLayer && !layerId) || isPreparing}
          >
            Add
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
