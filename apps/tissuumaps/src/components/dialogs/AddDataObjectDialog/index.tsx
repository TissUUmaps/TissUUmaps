import { computeLabel } from "@jsonforms/core";
import { JsonForms } from "@jsonforms/react";
import { useEffect, useRef, useState } from "react";

import type { Data, DataProvider, DataSource } from "@tissuumaps/core";

import { Field, FieldError, FieldLabel } from "@/components/common/field";
import { Fieldset } from "@/components/common/fieldset";
import { SimpleSelect } from "@/components/common/simple-select";
import { SourceField } from "@/components/controls/SourceField";
import { useAlertDialog } from "@/components/dialogs/AlertDialog/hooks";
import { ajv } from "@/components/jsonforms/ajv";
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
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { usePrepareDataSource } from "@/data/hooks/usePrepareDataSource";
import { useLatestCallback } from "@/hooks/useLatestCallback";
import { useProjectStore } from "@/stores/project";

import { useSourceInspection } from "./hooks";

// The caller-facing fields: everything except the open state, the source of
// a single dialog and the callbacks, which are owned by the DialogProvider.
export type AddDataObjectDialogParams<
  TDataSource extends DataSource = DataSource,
> = Omit<
  AddDataObjectDialogProps<TDataSource>,
  "open" | "initialSource" | "onClose" | "onClosed"
>;

export type AddDataObjectDialogProps<
  TDataSource extends DataSource = DataSource,
> = {
  /** Whether the dialog is shown */
  open: boolean;
  /** The dialog title, e.g. "Add images" */
  title: string;
  /** The registered data providers of the kind, by data source type */
  dataProviders: Map<string, DataProvider<TDataSource, Data>>;
  /** Whether the data object is added to a layer, chosen in the dialog */
  withLayer?: boolean;
  /** The source the dialog opens with, e.g. a dropped file */
  initialSource?: string;
  /**
   * Adds the data object; called with its name, its layer (if `withLayer`),
   * the source of its data source before it was prepared, and its prepared
   * data source
   */
  onAdd: (
    name: string,
    layerId: string | undefined,
    origSource: string | undefined,
    preparedDataSource: TDataSource,
  ) => void;
  /** Called when the dialog is cancelled, and after the data object was added */
  onClose: () => void;
  /** Called once the dialog has finished closing */
  onClosed: () => void;
};

/**
 * Dialog for adding a data object, built around its source
 *
 * Once a source is committed (see `SourceField`), or right away for an
 * initial one, the data providers are asked whether they support the source
 * (see `DataProvider.supports`): the first one that does is selected, unless
 * the user chose one, and those that do not are muted. The name is then read
 * from the source (see `DataProvider.readName`), or else is its file name,
 * unless the user edited it. Adding is disabled while this inspection runs.
 *
 * Errors are shown once the source was committed or adding was attempted.
 * The data source is prepared before the data object is added (see
 * `usePrepareDataSource`); closing the dialog cancels it.
 */
export function AddDataObjectDialog<TDataSource extends DataSource>({
  open,
  title,
  dataProviders,
  withLayer = false,
  initialSource,
  onAdd,
  onClose,
  onClosed,
}: AddDataObjectDialogProps<TDataSource>) {
  const providerEntries = Array.from(dataProviders.entries());
  const layers = useProjectStore((state) => state.layers);
  const alert = useAlertDialog();
  const {
    supportByType,
    isInspecting,
    inspect,
    cancel: cancelInspection,
  } = useSourceInspection(dataProviders);
  const {
    isPreparing,
    prepare,
    cancel: cancelPreparation,
  } = usePrepareDataSource();

  const [name, setName] = useState("");
  const [draft, setDraft] = useState(
    () =>
      ({
        type: providerEntries[0]?.[0] ?? "",
        ...(initialSource !== undefined && { source: initialSource }),
      }) as TDataSource,
  );
  const [layerId, setLayerId] = useState(layers[0]?.id);
  const [showErrors, setShowErrors] = useState(false);

  const isNameEditedRef = useRef(false);
  const chosenTypeRef = useRef<string | null>(null);

  const dataProvider = dataProviders.get(draft.type);
  const validate =
    dataProvider !== undefined ? ajv.compile(dataProvider.schema) : null;
  const isValid = validate !== null && validate(draft);
  // errors at the root, such as an unmet `anyOf`, belong to no field
  const hasRootErrors =
    !isValid &&
    (validate?.errors ?? []).some(
      (error) => error.instancePath === "" && error.keyword !== "required",
    );
  const isLayerMissing = withLayer && layerId === undefined;

  // a new type resets the draft, but keeps its source if the type declares one
  const withType = (draft: TDataSource, type: string) => {
    let result = draft;
    if (draft.type !== type) {
      result = { type } as TDataSource;
      const dataProvider = dataProviders.get(type);
      if (
        dataProvider?.schema.properties?.source !== undefined &&
        draft.source !== undefined
      ) {
        result.source = draft.source;
      }
    }
    return result;
  };

  // an inspection selects the type and gives the name, unless the user chose
  // the type or edited the name
  const inspectSource = async (source: string | undefined) => {
    const inspection = await inspect(
      source,
      (firstSupportedType) =>
        chosenTypeRef.current ?? firstSupportedType ?? draft.type,
    );
    if (inspection !== undefined) {
      if (chosenTypeRef.current === null) {
        setDraft((draft) => withType(draft, inspection.type));
      }
      if (!isNameEditedRef.current) {
        setName(inspection.name);
      }
    }
  };

  // an initial source is inspected once, when the dialog opens
  const inspectInitialSource = useLatestCallback(() => {
    if (initialSource !== undefined) {
      void inspectSource(initialSource);
    }
  });
  useEffect(() => inspectInitialSource(), [inspectInitialSource]);

  const add = async () => {
    if (dataProvider === undefined || !isValid || isLayerMissing) {
      setShowErrors(true);
      return;
    }
    const preparedDataSource = await prepare(draft, dataProvider);
    if (preparedDataSource === undefined) {
      return;
    }
    try {
      onAdd(
        name.trim() || "Untitled",
        withLayer ? layerId : undefined,
        draft.source,
        preparedDataSource,
      );
    } catch (error) {
      console.error("Failed to add the data object", error);
      void alert({
        title: "Cannot add the data object",
        body: error instanceof Error ? error.message : String(error),
      });
      return;
    }
    onClose();
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(newOpen) => {
        if (!newOpen) {
          cancelInspection();
          cancelPreparation();
          onClose();
        }
      }}
      onOpenChangeComplete={(newOpen) => {
        if (!newOpen) {
          onClosed();
        }
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>

        <Fieldset disabled={isPreparing} className="flex flex-col gap-4">
          {dataProvider !== undefined && (
            <SourceField
              schema={dataProvider.schema}
              value={draft.source}
              onValueChange={(source) => {
                setDraft((draft) => {
                  const newDraft = { ...draft, source };
                  if (source === undefined) {
                    delete newDraft.source;
                  }
                  return newDraft;
                });
              }}
              onValueCommit={(source) => {
                if (!open) {
                  return;
                }
                setShowErrors(true);
                void inspectSource(source);
              }}
              showErrors={showErrors}
              disabled={isPreparing}
            />
          )}

          <Field className="flex flex-col gap-2">
            <FieldLabel>Type</FieldLabel>
            <SimpleSelect
              items={providerEntries}
              itemLabel={([type, provider]) =>
                supportByType?.get(type) === false ? (
                  <span className="text-muted-foreground">{provider.name}</span>
                ) : (
                  provider.name
                )
              }
              itemValue={([type]) => type}
              value={draft.type}
              disabled={isPreparing}
              onValueChange={(type) => {
                if (type !== null) {
                  chosenTypeRef.current = type;
                  setDraft((draft) => withType(draft, type));
                }
              }}
            />
          </Field>

          <Field className="flex flex-col gap-2">
            <FieldLabel>Name</FieldLabel>
            <Input
              type="text"
              placeholder="Enter a name"
              disabled={isPreparing}
              value={name}
              onChange={(event) => {
                isNameEditedRef.current = event.target.value !== "";
                setName(event.target.value);
              }}
            />
          </Field>

          {withLayer && (
            <Field
              invalid={showErrors && isLayerMissing}
              className="flex flex-col gap-2"
            >
              <FieldLabel>{computeLabel("Layer", true, false)}</FieldLabel>
              <SimpleSelect
                items={layers}
                itemLabel={(layer) => layer.name}
                itemValue={(layer) => layer.id}
                value={layerId ?? null}
                disabled={isPreparing}
                onValueChange={(value) => {
                  if (value !== null) {
                    setLayerId(value);
                  }
                }}
              />
              {showErrors && isLayerMissing && (
                <FieldError match>Required</FieldError>
              )}
            </Field>
          )}

          {dataProvider !== undefined &&
            (!("elements" in dataProvider.uischema) ||
              dataProvider.uischema.elements.length > 0) && (
              <Field className="flex flex-col gap-2">
                <FieldLabel>Configuration</FieldLabel>
                <JsonForms
                  ajv={ajv}
                  data={draft}
                  onChange={({ data }) => setDraft(data as TDataSource)}
                  schema={dataProvider.schema}
                  uischema={dataProvider.uischema}
                  renderers={renderers}
                  cells={cells}
                  readonly={isPreparing}
                  validationMode={
                    showErrors ? "ValidateAndShow" : "ValidateAndHide"
                  }
                />
              </Field>
            )}

          {showErrors && hasRootErrors && (
            <span className="text-xs text-destructive">
              Invalid data source
            </span>
          )}
        </Fieldset>

        <DialogFooter>
          <DialogClose render={<Button variant="outline" />}>
            Cancel
          </DialogClose>
          <Button
            onClick={() => void add()}
            disabled={isInspecting || isPreparing || !open}
          >
            Add
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
