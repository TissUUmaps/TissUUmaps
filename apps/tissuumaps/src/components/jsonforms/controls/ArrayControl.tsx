import {
  type ArrayControlProps,
  type ControlElement,
  composePaths,
  createDefaultValue,
  createLabelDescriptionFrom,
  findUISchema,
} from "@jsonforms/core";
import {
  DispatchCell,
  JsonFormsDispatch,
  withJsonFormsArrayControlProps,
} from "@jsonforms/react";
import { ArrowDownIcon, ArrowUpIcon, XIcon } from "lucide-react";
import { Fragment, useMemo } from "react";

import {
  Field,
  FieldError,
  FieldItem,
  FieldLabel,
} from "@/components/common/field";
import { IconButton } from "@/components/common/icon-button";
import { Button } from "@/components/ui/button";

/** The control of a primitive array item, for rendering it as a cell */
const itemControl: ControlElement = { type: "Control", scope: "#" };

export const ArrayControl = withJsonFormsArrayControlProps(
  (props: ArrayControlProps) => {
    const childUISchema = useMemo(
      () =>
        findUISchema(
          props.uischemas ?? [],
          props.schema,
          props.uischema.scope,
          props.path,
          undefined,
          props.uischema,
          props.rootSchema,
        ),
      [
        props.uischemas,
        props.schema,
        props.path,
        props.uischema,
        props.rootSchema,
      ],
    );

    const description = createLabelDescriptionFrom(
      props.uischema,
      props.schema,
    );

    // readonly mode
    if (!props.enabled) {
      const length = Array.isArray(props.data) ? props.data.length : 0;
      if (length === 0) {
        return null;
      }
      return (
        <div className="contents">
          <dt className="text-muted-foreground">{description.text}</dt>
          <dd className="wrap-anywhere">
            {Array.from({ length }, (_, index) => (
              <Fragment key={index}>
                {index > 0 && ", "}
                <DispatchCell
                  schema={props.schema}
                  uischema={itemControl}
                  path={composePaths(props.path, `${index}`)}
                  enabled={false}
                />
              </Fragment>
            ))}
          </dd>
        </div>
      );
    }

    return (
      <Field>
        {description.show && <FieldLabel>{description.text}</FieldLabel>}
        <div className="grid grid-cols-[1fr_auto_auto_auto]">
          {Array.from(
            { length: (props.data as unknown[]).length },
            (_, index) => {
              const childPath = composePaths(props.path, `${index}`);
              return (
                <FieldItem key={index} className="contents">
                  <JsonFormsDispatch
                    schema={props.schema}
                    uischema={childUISchema || props.uischema}
                    path={childPath}
                    key={childPath}
                    renderers={props.renderers}
                  />
                  <IconButton
                    label="Move up"
                    size="icon"
                    onClick={() => props.moveUp?.(props.path, index)()}
                  >
                    <ArrowUpIcon />
                  </IconButton>
                  <IconButton
                    label="Move down"
                    size="icon"
                    onClick={() => props.moveDown?.(props.path, index)()}
                  >
                    <ArrowDownIcon />
                  </IconButton>
                  <IconButton
                    label="Remove"
                    size="icon"
                    onClick={() => props.removeItems?.(props.path, [index])()}
                  >
                    <XIcon />
                  </IconButton>
                </FieldItem>
              );
            },
          )}
        </div>
        <Button
          className="w-full"
          onClick={() =>
            props.addItem?.(
              props.path,
              createDefaultValue(props.schema, props.rootSchema),
            )()
          }
        >
          Add item
        </Button>
        {props.errors && <FieldError>{props.errors}</FieldError>}
      </Field>
    );
  },
);
