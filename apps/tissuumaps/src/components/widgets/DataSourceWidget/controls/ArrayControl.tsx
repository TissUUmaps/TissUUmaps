import {
  type ArrayControlProps,
  composePaths,
  createDefaultValue,
  createLabelDescriptionFrom,
  findUISchema,
} from "@jsonforms/core";
import {
  JsonFormsDispatch,
  withJsonFormsArrayControlProps,
} from "@jsonforms/react";
import { ArrowDownIcon, ArrowUpIcon, XIcon } from "lucide-react";
import { useMemo } from "react";

import {
  Field,
  FieldError,
  FieldItem,
  FieldLabel,
} from "@/components/common/field";
import { IconButton } from "@/components/common/icon-button";
import { Button } from "@/components/ui/button";

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
                    variant="default"
                    size="icon"
                    disabled={!props.enabled}
                    onClick={() => props.moveUp?.(props.path, index)()}
                  >
                    <ArrowUpIcon />
                  </IconButton>
                  <IconButton
                    label="Move down"
                    variant="default"
                    size="icon"
                    disabled={!props.enabled}
                    onClick={() => props.moveDown?.(props.path, index)()}
                  >
                    <ArrowDownIcon />
                  </IconButton>
                  <IconButton
                    label="Remove"
                    variant="default"
                    size="icon"
                    disabled={!props.enabled}
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
          disabled={!props.enabled}
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
