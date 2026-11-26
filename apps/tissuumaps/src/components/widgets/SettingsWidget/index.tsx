import { Fragment, type ReactNode, useId } from "react";

import type { TableColumnType } from "@tissuumaps/core";

import { Fieldset, FieldsetLegend } from "@/components/common/fieldset";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

import { ColumnTypeMenu } from "./ColumnTypeMenu";
import { StyleByPicker } from "./StyleByPicker";
import { StyleRow } from "./StyleRow";
import {
  type StyleBy,
  type StyleProperty,
  getFollowingNames,
  getPartToggledNames,
  getToggledNames,
  isStyleItemOn,
  isStyleItemSupported,
} from "./styleBy";
import { useStyleBy } from "./useStyleBy";

/**
 * A property of an object that can follow the column of the settings widget,
 * with its toggle and options
 */
export type SettingsWidgetProperty = StyleProperty & {
  /** The name shown to the user, e.g. "Color" */
  label: string;

  /** The icon of the property's toggle */
  icon: ReactNode;

  /**
   * What following a column does, if not obvious, shown in the toggle's
   * tooltip after the label
   */
  hint?: string;

  /**
   * Rows of the property's options while it follows a categorical column,
   * e.g. the palette of a color
   */
  categoricalOptions?: ReactNode;

  /**
   * Rows of the property's options while it follows a numerical column, e.g.
   * the palette and range of a color
   */
  numericalOptions?: ReactNode;
};

/**
 * The properties of a part of an object, e.g. the color, opacity and
 * visibility of the fill of shapes, toggled together under the part's name
 * (see {@link StylePart})
 *
 * Its own toggles are shown for numerical columns only; for categorical
 * columns, the part stands for its first property.
 */
export type SettingsWidgetPart = {
  /** Tells the part apart from the properties of the object */
  name: string;

  /** The name shown to the user, e.g. "Shape fill" */
  label: string;

  /** The icon of the part's toggle */
  icon: ReactNode;

  /** The properties of the part; the first one follows a newly toggled part */
  properties: [SettingsWidgetProperty, ...SettingsWidgetProperty[]];
};

type SettingsWidgetItem = SettingsWidgetProperty | SettingsWidgetPart;

export type SettingsWidgetProps = {
  /** The ID of the object's own table, if it has one */
  tableId: string | null;

  /** The column the object is styled by, if any (see `findStyleBy`) */
  styleBy: StyleBy | null;

  /**
   * The properties and parts that can follow the column, in the order of
   * their toggles; the first one follows a newly picked column if none
   * follows the column
   */
  items: [SettingsWidgetItem, ...SettingsWidgetItem[]];

  /**
   * Writes what the object is styled by and which properties follow it, to
   * the object as it is when called (the call may follow a load)
   *
   * @param styleBy - The column the object is styled by, if any
   * @param propertyNames - The names of the properties that follow the
   * column, empty for none; parts are named by their properties
   */
  onStyleChange: (styleBy: StyleBy | null, propertyNames: string[]) => void;

  /**
   * What the object looks like while styled by no column, e.g. "Random
   * colors" (see `describeColorWithoutColumn`)
   */
  placeholder: string;

  /** The settings below the style, e.g. the advanced settings */
  children?: ReactNode;

  className?: string;
};

/**
 * The always available settings of an object annotated by a table: the
 * column to style it by, its type, and which properties follow it, in compact
 * labelled rows
 *
 * The column type follows the picked column and determines which properties
 * can follow it. The values of the categories are edited in the group table
 * below the widget; the other settings in the advanced settings.
 */
export function SettingsWidget({
  tableId,
  styleBy,
  items,
  onStyleChange,
  placeholder,
  children,
  className,
}: SettingsWidgetProps) {
  const { isCategorical, isPending, pick, setColumnType } = useStyleBy(
    tableId,
    styleBy,
    (newStyleBy) => {
      onStyleChange(
        newStyleBy,
        newStyleBy !== null
          ? getFollowingNames(items, newStyleBy.columnType)
          : [],
      );
    },
  );

  return (
    <Fieldset
      className={cn("flex flex-col gap-y-2 border rounded-md p-2", className)}
    >
      <FieldsetLegend className="font-medium text-foreground">
        Settings
      </FieldsetLegend>
      <Fieldset
        disabled={tableId === null}
        className="grid grid-cols-[5rem_1fr] items-center gap-x-2 gap-y-1.5"
      >
        <FieldsetLegend className="sr-only">Settings</FieldsetLegend>
        <StyleByPicker
          tableId={tableId}
          column={styleBy?.column ?? null}
          onColumnChange={pick}
          placeholder={tableId === null ? "No table" : placeholder}
          columnControls={
            styleBy !== null ? (
              <ColumnTypeMenu
                value={styleBy.columnType}
                onValueChange={setColumnType}
                isCategoricalOnly={isCategorical === true}
                // the column types a column allows are known with its type
                disabled={isPending || isCategorical === null}
              />
            ) : undefined
          }
        />
        {styleBy !== null && (
          <>
            <StyleRow label="Applies to">
              <ItemToggleGroup
                items={items}
                columnType={styleBy.columnType}
                disabled={isPending}
                onValueChange={(value) => {
                  const names = getToggledNames(
                    items,
                    value,
                    styleBy.columnType,
                  );
                  // the column always styles something; None clears it
                  if (names !== null) {
                    onStyleChange(styleBy, names);
                  }
                }}
              />
            </StyleRow>
            {styleBy.columnType === "categorical" &&
              items.filter(isStyleItemOn).map((item) => {
                // a part follows a categorical column by its first property
                const { categoricalOptions } =
                  "properties" in item ? item.properties[0] : item;
                return (
                  categoricalOptions !== undefined && (
                    <StyleGroup key={item.name} label={item.label}>
                      {categoricalOptions}
                    </StyleGroup>
                  )
                );
              })}
            {styleBy.columnType === "numerical" &&
              items.filter(isStyleItemOn).map((item) =>
                "properties" in item ? (
                  <StyleGroup
                    key={item.name}
                    label={item.label}
                    actions={
                      <ItemToggleGroup
                        items={item.properties}
                        columnType={styleBy.columnType}
                        disabled={isPending}
                        onValueChange={(value) => {
                          const names = getPartToggledNames(items, item, value);
                          if (names !== null) {
                            onStyleChange(styleBy, names);
                          }
                        }}
                      />
                    }
                  >
                    {item.properties
                      .filter(({ isOn }) => isOn)
                      .map((property) => (
                        <Fragment key={property.name}>
                          {property.numericalOptions}
                        </Fragment>
                      ))}
                  </StyleGroup>
                ) : (
                  item.numericalOptions !== undefined && (
                    <StyleGroup key={item.name} label={item.label}>
                      {item.numericalOptions}
                    </StyleGroup>
                  )
                ),
              )}
          </>
        )}
        {isPending && (
          <StyleRow>
            <span className="text-xs text-muted-foreground">
              Loading column…
            </span>
          </StyleRow>
        )}
      </Fieldset>
      {children}
    </Fieldset>
  );
}

type ItemToggleGroupProps = {
  items: SettingsWidgetItem[];
  columnType: TableColumnType;
  disabled: boolean;
  /** Called with the names of the items toggled on */
  onValueChange: (value: string[]) => void;
};

/**
 * The icon toggles of properties or parts, named by a tooltip; one that
 * cannot follow the column type is disabled
 */
function ItemToggleGroup({
  items,
  columnType,
  disabled,
  onValueChange,
}: ItemToggleGroupProps) {
  return (
    <ToggleGroup
      multiple
      value={items.filter(isStyleItemOn).map(({ name }) => name)}
      onValueChange={onValueChange}
      size="sm"
      className="border rounded"
      disabled={disabled}
    >
      {items.map((item) => {
        const hint = "properties" in item ? undefined : item.hint;
        return (
          <Tooltip key={item.name}>
            <TooltipTrigger
              render={
                <ToggleGroupItem
                  value={item.name}
                  aria-label={item.label}
                  disabled={!isStyleItemSupported(item, columnType)}
                  className="px-2"
                />
              }
            >
              {item.icon}
            </TooltipTrigger>
            <TooltipContent>
              {hint !== undefined ? `${item.label}: ${hint}` : item.label}
            </TooltipContent>
          </Tooltip>
        );
      })}
    </ToggleGroup>
  );
}

type StyleGroupProps = {
  label: string;
  /** Controls shown at the right of the name, e.g. the toggles of a part */
  actions?: ReactNode;
  children?: ReactNode;
};

/** The options of one property or part, under its name, as rows of the grid */
function StyleGroup({ label, actions, children }: StyleGroupProps) {
  const labelId = useId();
  return (
    <div
      role="group"
      aria-labelledby={labelId}
      className="col-span-2 grid grid-cols-subgrid items-center gap-y-1.5 border-t pt-1.5"
    >
      <div className="col-span-2 flex flex-row items-center justify-between gap-x-2">
        <span id={labelId} className="text-xs font-medium">
          {label}
        </span>
        {actions}
      </div>
      {children}
    </div>
  );
}
