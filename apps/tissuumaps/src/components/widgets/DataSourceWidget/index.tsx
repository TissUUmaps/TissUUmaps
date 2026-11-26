import { JsonForms } from "@jsonforms/react";
import { EditIcon, RotateCcwIcon, SaveIcon, XIcon } from "lucide-react";
import { useMemo, useState } from "react";

import type { Data, DataProvider, DataSource } from "@tissuumaps/core";

import {
  Collapsible,
  CollapsiblePanel,
  CollapsibleTrigger,
  CollapsibleTriggerRightDownIcon,
} from "@/components/common/collapsible";
import { Field, FieldLabel } from "@/components/common/field";
import { Fieldset, FieldsetLegend } from "@/components/common/fieldset";
import { IconButton } from "@/components/common/icon-button";
import { SimpleSelect } from "@/components/common/simple-select";
import { SourceField } from "@/components/controls/SourceField";
import { useAlertDialog } from "@/components/dialogs/AlertDialog/hooks";
import { ajv } from "@/components/jsonforms/ajv";
import { cells } from "@/components/jsonforms/cells";
import { listRenderers, renderers } from "@/components/jsonforms/renderers";
import { usePrepareDataSource } from "@/data/hooks/usePrepareDataSource";
import { cn } from "@/lib/utils";

export type DataSourceWidgetProps<TDataSource extends DataSource> = {
  dataSource: TDataSource;
  dataProviders: Map<string, DataProvider<DataSource, Data>>;
  onDataSourceChange: (newDataSource: TDataSource) => void;
  className?: string;
};

/**
 * Shows the type and source of a data source, the full source and the rest of
 * it when expanded, and edits it on demand
 *
 * The source is not part of any data provider's UI schema, but rendered here.
 */
export function DataSourceWidget<TDataSource extends DataSource>({
  dataSource,
  dataProviders,
  onDataSourceChange,
  className,
}: DataSourceWidgetProps<TDataSource>) {
  const [isEditing, setEditing] = useState(false);
  return (
    <Fieldset
      className={cn("flex flex-col gap-y-2 border rounded-md p-2", className)}
    >
      {isEditing ? (
        <DataSourceEditor
          dataSource={dataSource}
          dataProviders={dataProviders}
          onSave={(newDataSource) => {
            onDataSourceChange(newDataSource);
            setEditing(false);
          }}
          onCancel={() => setEditing(false)}
        />
      ) : (
        <DataSourceView
          dataSource={dataSource}
          dataProviders={dataProviders}
          onEdit={() => setEditing(true)}
        />
      )}
    </Fieldset>
  );
}

type DataSourceViewProps = {
  dataSource: DataSource;
  dataProviders: Map<string, DataProvider<DataSource, Data>>;
  onEdit: () => void;
};

function DataSourceView({
  dataSource,
  dataProviders,
  onEdit,
}: DataSourceViewProps) {
  const dataProvider = dataProviders.get(dataSource.type);
  // without a data provider, the source is the only hint to the data
  const showSource =
    dataProvider === undefined ||
    dataProvider.schema.properties?.source !== undefined;
  const hasSource = showSource && dataSource.source !== undefined;
  const hasFields =
    dataProvider !== undefined &&
    (!("elements" in dataProvider.uischema) ||
      dataProvider.uischema.elements.length > 0);
  const canExpand = hasSource || hasFields;
  const summary = (
    <>
      <span className="shrink-0 text-muted-foreground">Type</span>
      <span className="shrink-0">{dataProvider?.name ?? dataSource.type}</span>
      {hasSource && (
        <span className="truncate" title={dataSource.source}>
          {dataSource.source}
        </span>
      )}
    </>
  );
  return (
    <>
      <FieldsetLegend className="sr-only">Data source</FieldsetLegend>
      <Collapsible className="flex flex-col gap-y-1">
        <div className="flex flex-row items-center gap-x-1 text-xs">
          {canExpand ? (
            <>
              <CollapsibleTriggerRightDownIcon />
              <CollapsibleTrigger className="flex min-w-0 flex-1 flex-row gap-x-3 text-left">
                {summary}
              </CollapsibleTrigger>
            </>
          ) : (
            <span className="flex min-w-0 flex-1 flex-row gap-x-3 pl-5">
              {summary}
            </span>
          )}
          <IconButton label="Edit" size="icon-xs" onClick={onEdit}>
            <EditIcon className="size-4" />
          </IconButton>
        </div>
        {canExpand && (
          <CollapsiblePanel>
            <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 pl-5 text-xs">
              {hasSource && (
                <div className="contents">
                  <dt className="text-muted-foreground">Source</dt>
                  <dd className="wrap-anywhere">{dataSource.source}</dd>
                </div>
              )}
              {hasFields && (
                <JsonForms
                  ajv={ajv}
                  data={dataSource}
                  schema={dataProvider.schema}
                  uischema={dataProvider.uischema}
                  renderers={listRenderers}
                  cells={cells}
                  readonly
                />
              )}
            </dl>
          </CollapsiblePanel>
        )}
      </Collapsible>
      {dataProvider === undefined && <MissingDataProviderHint />}
    </>
  );
}

type DataSourceEditorProps<TDataSource extends DataSource> = {
  dataSource: TDataSource;
  dataProviders: Map<string, DataProvider<DataSource, Data>>;
  onSave: (newDataSource: TDataSource) => void;
  onCancel: () => void;
};

function DataSourceEditor<TDataSource extends DataSource>({
  dataSource,
  dataProviders,
  onSave,
  onCancel,
}: DataSourceEditorProps<TDataSource>) {
  const [draft, setDraft] = useState(() => structuredClone(dataSource));
  const { isPreparing, prepare, cancel } = usePrepareDataSource();
  const alert = useAlertDialog();

  const dataProvider = dataProviders.get(draft.type);
  const providerEntries = useMemo(
    () => Array.from(dataProviders.entries()),
    [dataProviders],
  );

  const validate = useMemo(
    () =>
      dataProvider !== undefined ? ajv.compile(dataProvider.schema) : null,
    [dataProvider],
  );
  const isValid = validate !== null && validate(draft);
  // errors at the root, such as an unmet `anyOf`, belong to no field
  const hasRootErrors =
    !isValid &&
    (validate?.errors ?? []).some(
      (error) => error.instancePath === "" && error.keyword !== "required",
    );

  const save = async () => {
    // without keys its data provider's schema does not declare, e.g. ones a
    // project file carries
    const knownKeys = new Set([
      "type",
      ...Object.keys(dataProvider?.schema.properties ?? {}),
    ]);
    const newDataSource = Object.fromEntries(
      Object.entries(draft).filter(([key]) => knownKeys.has(key)),
    ) as TDataSource;
    // prepared only if its source or type changed, as preparing may create
    // data objects for the source (see DataProvider.prepareDataSource)
    let preparedDataSource: TDataSource | undefined = newDataSource;
    if (
      newDataSource.type !== dataSource.type ||
      newDataSource.source !== dataSource.source
    ) {
      try {
        preparedDataSource = await prepare(newDataSource, dataProvider);
      } catch (error) {
        console.error("Failed to prepare the data source", error);
        void alert({
          title: "Cannot prepare the data source",
          body: error instanceof Error ? error.message : String(error),
        });
        return;
      }
    }
    if (preparedDataSource !== undefined) {
      onSave(preparedDataSource);
    }
  };

  return (
    <>
      <FieldsetLegend className="flex flex-row items-center gap-x-1 font-medium text-foreground">
        <Field className="flex flex-row items-center gap-x-1">
          <FieldLabel>Type</FieldLabel>
          <SimpleSelect
            items={providerEntries}
            itemLabel={([, provider]) => provider.name}
            itemValue={([type]) => type}
            value={draft.type}
            disabled={isPreparing}
            onValueChange={(type) => {
              if (type !== null) {
                setDraft((draft) => ({ ...draft, type }));
              }
            }}
          />
        </Field>
        <span className="ml-auto flex flex-row">
          <IconButton label="Cancel" onClick={onCancel}>
            <XIcon className="size-4" />
          </IconButton>
          <IconButton
            label="Reset"
            onClick={() => {
              cancel(); // a pending save, which may hang on remote data
              setDraft(structuredClone(dataSource));
            }}
          >
            <RotateCcwIcon className="size-4" />
          </IconButton>
          <IconButton
            label="Save"
            disabled={!isValid || isPreparing}
            onClick={() => void save()}
          >
            <SaveIcon className="size-4" />
          </IconButton>
        </span>
      </FieldsetLegend>
      {dataProvider === undefined ? (
        <>
          <SourceRow source={draft.source} />
          <MissingDataProviderHint isEditing />
        </>
      ) : (
        <>
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
            showErrors
            disabled={isPreparing}
          />
          {(!("elements" in dataProvider.uischema) ||
            dataProvider.uischema.elements.length > 0) && (
            <JsonForms
              ajv={ajv}
              data={draft}
              onChange={({ data }) => setDraft(data as TDataSource)}
              schema={dataProvider.schema}
              uischema={dataProvider.uischema}
              renderers={renderers}
              cells={cells}
              readonly={isPreparing}
            />
          )}
        </>
      )}
      {hasRootErrors && (
        <span className="text-xs text-destructive">Invalid data source</span>
      )}
    </>
  );
}

function SourceRow({ source }: { source: string | undefined }) {
  if (source === undefined) {
    return null;
  }
  return (
    <Field className="grid grid-cols-[auto_1fr] gap-x-2 items-baseline">
      <FieldLabel>Source:</FieldLabel>
      <span className="truncate">{source}</span>
    </Field>
  );
}

function MissingDataProviderHint({ isEditing = false }) {
  return (
    <span className="text-xs text-muted-foreground">
      No data provider is registered for this data source type.
      {isEditing && " Select another type above."}
    </span>
  );
}
