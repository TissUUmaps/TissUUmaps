import {
  type ArrayControlProps,
  type ControlElement,
  composePaths,
} from "@jsonforms/core";
import { DispatchCell, withJsonFormsArrayControlProps } from "@jsonforms/react";
import { Fragment } from "react";

/** The control of a primitive array item, for rendering it as a cell */
const itemControl: ControlElement = { type: "Control", scope: "#" };

export const ListArrayControl = withJsonFormsArrayControlProps(
  (props: ArrayControlProps) => {
    const length = Array.isArray(props.data) ? props.data.length : 0;
    if (!props.visible || length === 0) {
      return null;
    }
    return (
      <div className="contents">
        <dt className="text-muted-foreground">{props.label}</dt>
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
  },
);
