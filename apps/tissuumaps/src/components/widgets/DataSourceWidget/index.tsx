import { createAjv } from "@jsonforms/core";
import { JsonForms } from "@jsonforms/react";
import { EditIcon, RotateCcwIcon, SaveIcon } from "lucide-react";
import { useMemo, useState } from "react";

import type { Data, DataProvider, DataSource } from "@tissuumaps/core";

import { Field, FieldLabel } from "@/components/common/field";
import { Fieldset, FieldsetLegend } from "@/components/common/fieldset";
import { IconButton } from "@/components/common/icon-button";
import { SimpleSelect } from "@/components/common/simple-select";
import { cn } from "@/lib/utils";

import { cells } from "./cells";
import { renderers } from "./renderers";

export type DataSourceWidgetProps<TDataSource extends DataSource> = {
  dataSource: TDataSource;
  dataProviders: Map<string, DataProvider<DataSource, Data>>;
  onDataSourceChange: (newDataSource: TDataSource) => void;
  className?: string;
};

export function DataSourceWidget<TDataSource extends DataSource>({
  dataSource,
  dataProviders,
  onDataSourceChange,
  className,
}: DataSourceWidgetProps<TDataSource>) {
  const [dataSourceDraft, setDataSourceDraft] = useState<TDataSource | null>(
    null,
  );
  const currentDataSource = dataSourceDraft ?? dataSource;
  const isEditing = dataSourceDraft !== null;

  const providerEntries = useMemo(
    () => Array.from(dataProviders.entries()),
    [dataProviders],
  );

  const dataProvider = dataProviders.get(currentDataSource.type);

  // Validate the current draft against the active provider's schema using the
  // same Ajv config JsonForms uses, so shared fields (e.g. URL) carry over and
  // validity is re-checked on every edit or type switch instead of being reset.
  const ajv = useMemo(() => createAjv(), []);
  const validate = useMemo(
    () =>
      dataProvider !== undefined ? ajv.compile(dataProvider.schema) : null,
    [ajv, dataProvider],
  );
  const hasErrors =
    isEditing && (validate === null || !validate(currentDataSource));

  return (
    <Fieldset
      className={cn("flex flex-col gap-y-2 border rounded-md p-2", className)}
    >
      <FieldsetLegend className="flex flex-row items-center gap-x-1 font-medium text-foreground">
        {isEditing ? (
          <Field className="flex flex-row items-center gap-x-1">
            <FieldLabel>Source</FieldLabel>
            <SimpleSelect
              items={providerEntries}
              itemLabel={([, provider]) => provider.name}
              itemValue={([type]) => type}
              value={currentDataSource.type}
              onValueChange={(value) => {
                if (value !== null) {
                  setDataSourceDraft({
                    ...dataSourceDraft,
                    type: value,
                  });
                }
              }}
            />
          </Field>
        ) : (
          <>Source: {dataProvider?.name ?? `type=${currentDataSource.type}`}</>
        )}
        {isEditing ? (
          <span className="ml-auto flex flex-row">
            <IconButton
              label="Reset"
              onClick={() => setDataSourceDraft(structuredClone(dataSource))}
            >
              <RotateCcwIcon className="size-4" />
            </IconButton>
            <IconButton
              label="Save"
              disabled={hasErrors}
              onClick={() => {
                const knownKeys = new Set([
                  "type",
                  ...Object.keys(dataProvider?.schema.properties ?? {}),
                ]);
                const cleaned = Object.fromEntries(
                  Object.entries(dataSourceDraft).filter(([k]) =>
                    knownKeys.has(k),
                  ),
                ) as TDataSource;
                onDataSourceChange(cleaned);
                setDataSourceDraft(null);
              }}
            >
              <SaveIcon className="size-4" />
            </IconButton>
          </span>
        ) : (
          <IconButton
            label="Edit"
            className="ml-auto"
            onClick={() => {
              setDataSourceDraft(structuredClone(dataSource));
            }}
          >
            <EditIcon className="size-4" />
          </IconButton>
        )}
      </FieldsetLegend>
      {dataProvider !== undefined ? (
        <JsonForms
          data={currentDataSource}
          onChange={({ data }) => {
            if (isEditing) {
              setDataSourceDraft(data as TDataSource);
            }
          }}
          schema={dataProvider.schema}
          uischema={dataProvider.uischema}
          renderers={renderers}
          cells={cells}
          readonly={!isEditing}
        />
      ) : (
        <span className="text-xs text-muted-foreground">
          No data provider is registered for this data source type.
          {isEditing && " Select another source above."}
        </span>
      )}
    </Fieldset>
  );
}
