import { Fragment, type ReactNode, useId } from "react";

import type { TableColumnType } from "@tissuumaps/core";

import {
  Collapsible,
  CollapsiblePanel,
  CollapsibleTrigger,
  CollapsibleTriggerRightDownIcon,
} from "@/components/common/collapsible";
import { Fieldset, FieldsetLegend } from "@/components/common/fieldset";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useTableData } from "@/data/hooks/useData";
import { cn } from "@/lib/utils";

import { ColumnNameBox } from "./ColumnNameBox";
import { ColumnRows } from "./ColumnRows";
import { ColumnTypeMenu } from "./ColumnTypeMenu";
import { SettingsRow } from "./SettingsRow";
import {
  type FollowedColumn,
  getFollowingNames,
  getPartToggledNames,
  getToggledNames,
  isItemOn,
  isItemSupported,
} from "./follow";
import { useColumnLocation } from "./useColumnLocation";
import { useFollowedColumn } from "./useFollowedColumn";

/**
 * A property of an object that can follow the column of the settings widget,
 * with its toggle and options
 */
export type SettingsWidgetProperty = {
  /** Tells the property apart from the other properties of the object */
  name: string;

  /**
   * The column types the property can follow, e.g. only categorical columns
   * for markers
   */
  columnTypes: TableColumnType[];

  /** Whether the property follows the column of the settings widget */
  isOn: boolean;

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
 *
 * The part follows the column while any of its properties does. Its own
 * toggles are shown for numerical columns only; for categorical columns, the
 * part stands for its first property.
 */
export type SettingsWidgetPart = {
  /** Tells the part apart from the properties of the object */
  name: string;

  /** The name shown to the user, e.g. "Shape fill" */
  label: string;

  /** The icon of the part's toggle */
  icon: ReactNode;

  /** The properties of the part; the first one starts when the part is toggled on */
  properties: [SettingsWidgetProperty, ...SettingsWidgetProperty[]];
};

/** A property or a part of the settings widget */
export type SettingsWidgetItem = SettingsWidgetProperty | SettingsWidgetPart;

export type SettingsWidgetProps = {
  /** The ID of the object's own table, if it has one */
  tableId: string | null;

  /** The column the properties follow, if any (see `findFollowedColumn`) */
  followedColumn: FollowedColumn | null;

  /**
   * The properties and parts that can follow the column, in the order of
   * their toggles; the first one follows a newly picked column if none
   * follows the column
   */
  items: [SettingsWidgetItem, ...SettingsWidgetItem[]];

  /**
   * Writes the column to follow and which properties follow it, to the
   * object as it is when called (the call may come after a load)
   *
   * @param followedColumn - The column to follow, if any
   * @param propertyNames - The names of the properties that follow the
   * column, empty for none; parts are named by their properties
   */
  onFollowedColumnChange: (
    followedColumn: FollowedColumn | null,
    propertyNames: string[],
  ) => void;

  /**
   * What the object looks like while no property follows a column, e.g. "Random
   * colors" (see `describeColorWithoutColumn`)
   */
  placeholder: string;

  /** The advanced settings, collapsed under "Advanced" at the end */
  advancedSettings: ReactNode;

  className?: string;
};

/**
 * The always available settings of an object annotated by a table: the
 * column to follow, its type, and which properties follow it, in compact
 * labelled rows
 *
 * The picked column sets the column type, which determines which properties
 * can follow it. The values of the categories are edited in the group table
 * below the widget; the other settings in the advanced settings. For tables
 * with sections, the title line shows the query of the column and picks any
 * column by query (see {@link ColumnNameBox}); a column in no section, or not
 * in the table, shows only there, with its type.
 */
export function SettingsWidget({
  tableId,
  followedColumn,
  items,
  onFollowedColumnChange,
  placeholder,
  advancedSettings,
  className,
}: SettingsWidgetProps) {
  const tableData = useTableData(tableId);
  const column = followedColumn?.column ?? null;
  const location = useColumnLocation(tableData, column);
  const { isCategorical, isPending, pick, setColumnType } = useFollowedColumn(
    tableId,
    followedColumn,
    (newFollowedColumn) => {
      onFollowedColumnChange(
        newFollowedColumn,
        newFollowedColumn !== null
          ? getFollowingNames(items, newFollowedColumn.columnType)
          : [],
      );
    },
  );

  const columnTypeMenu = followedColumn !== null && (
    <ColumnTypeMenu
      value={followedColumn.columnType}
      onValueChange={setColumnType}
      isCategoricalOnly={isCategorical === true}
      // the column types a column allows are known with its type
      disabled={isPending || isCategorical === null}
    />
  );
  // tables without sections pick any column in their "Column" row
  const hasSections = (tableData?.getColumnSections?.() ?? []).length > 0;
  // the location of a previous column is kept until the new one resolves
  const currentLocation = location?.column === column ? location : null;
  // the rows hold only columns of a section; the header shows any other
  const isOutsideRows =
    currentLocation !== null && currentLocation.kind !== "section";

  return (
    <Fieldset
      className={cn("flex flex-col gap-y-2 border rounded-md p-2", className)}
    >
      <div className="flex min-w-0 flex-row items-center gap-x-1">
        <FieldsetLegend className="font-medium text-foreground">
          Settings
        </FieldsetLegend>
        {tableId !== null && hasSections && (
          <div className="flex min-w-0 flex-1 flex-row items-center justify-end gap-x-1">
            {isOutsideRows && columnTypeMenu}
            <ColumnNameBox
              tableId={tableId}
              column={column}
              isMissing={currentLocation?.kind === "missing"}
              onColumnChange={pick}
            />
          </div>
        )}
      </div>
      <Fieldset
        disabled={tableId === null}
        className="grid grid-cols-[5rem_1fr] items-center gap-x-2 gap-y-1.5"
      >
        <FieldsetLegend className="sr-only">Settings</FieldsetLegend>
        <ColumnRows
          // the leaves, messages and searches of the column rows are per table
          key={tableId ?? ""}
          tableId={tableId}
          column={column}
          location={location}
          onColumnChange={pick}
          placeholder={tableId === null ? "No table" : placeholder}
          columnControls={columnTypeMenu}
        />
        {followedColumn !== null && (
          <>
            <SettingsRow label="Applies to">
              <ItemToggleGroup
                items={items}
                columnType={followedColumn.columnType}
                disabled={isPending}
                onValueChange={(value) => {
                  const names = getToggledNames(
                    items,
                    value,
                    followedColumn.columnType,
                  );
                  // some property always follows the column; None clears it
                  if (names !== null) {
                    onFollowedColumnChange(followedColumn, names);
                  }
                }}
              />
            </SettingsRow>
            {followedColumn.columnType === "categorical" &&
              items.filter(isItemOn).map((item) => {
                // a part follows a categorical column by its first property
                const { categoricalOptions } =
                  "properties" in item ? item.properties[0] : item;
                return (
                  categoricalOptions !== undefined && (
                    <ItemOptions key={item.name} label={item.label}>
                      {categoricalOptions}
                    </ItemOptions>
                  )
                );
              })}
            {followedColumn.columnType === "numerical" &&
              items.filter(isItemOn).map((item) =>
                "properties" in item ? (
                  <ItemOptions
                    key={item.name}
                    label={item.label}
                    actions={
                      <ItemToggleGroup
                        items={item.properties}
                        columnType={followedColumn.columnType}
                        disabled={isPending}
                        onValueChange={(value) => {
                          const names = getPartToggledNames(items, item, value);
                          if (names !== null) {
                            onFollowedColumnChange(followedColumn, names);
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
                  </ItemOptions>
                ) : (
                  item.numericalOptions !== undefined && (
                    <ItemOptions key={item.name} label={item.label}>
                      {item.numericalOptions}
                    </ItemOptions>
                  )
                ),
              )}
          </>
        )}
        {isPending && (
          <SettingsRow>
            <span className="text-xs text-muted-foreground">
              Loading column…
            </span>
          </SettingsRow>
        )}
      </Fieldset>
      <Collapsible className="flex flex-col gap-y-2">
        <div className="flex flex-row items-center gap-x-1 text-sm">
          <CollapsibleTriggerRightDownIcon />
          <CollapsibleTrigger className="font-medium">
            Advanced
          </CollapsibleTrigger>
        </div>
        <CollapsiblePanel className="flex flex-col gap-y-2">
          {advancedSettings}
        </CollapsiblePanel>
      </Collapsible>
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
      value={items.filter(isItemOn).map(({ name }) => name)}
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
                  disabled={!isItemSupported(item, columnType)}
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

type ItemOptionsProps = {
  label: string;
  /** Controls shown at the right of the name, e.g. the toggles of a part */
  actions?: ReactNode;
  children?: ReactNode;
};

/** The options of one property or part, under its name, as rows of the grid */
function ItemOptions({ label, actions, children }: ItemOptionsProps) {
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
