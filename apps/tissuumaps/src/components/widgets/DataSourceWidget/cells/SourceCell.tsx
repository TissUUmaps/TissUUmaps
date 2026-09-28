import type { CellProps } from "@jsonforms/core";
import { withJsonFormsCellProps } from "@jsonforms/react";
import { FolderOpenIcon } from "lucide-react";

import { useAlertDialog } from "@/components/dialogs/AlertDialog/hooks";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { pickWorkspaceFile } from "@/data/io/workspace";
import { useAppStore } from "@/stores/app";

export const SourceCell = withJsonFormsCellProps((props: CellProps) => {
  const workspace = useAppStore((state) => state.workspace);
  const alert = useAlertDialog();
  const value = (props.data as string | undefined | null) ?? "";
  if (!props.enabled) {
    return value;
  }
  const options = {
    ...(props.config as { [key: string]: unknown }),
    ...props.uischema.options,
  };
  return (
    <InputGroup>
      <InputGroupInput
        type="text"
        id={props.id}
        value={value}
        onChange={(event) =>
          props.handleChange(
            props.path,
            event.target.value !== "" ? event.target.value : undefined,
          )
        }
        autoFocus={options.focus as boolean | undefined}
        placeholder={options.placeholder as string | undefined}
        maxLength={props.schema.maxLength}
      />
      {workspace !== null && (
        <InputGroupAddon align="inline-end">
          <Tooltip>
            <TooltipTrigger
              render={
                <InputGroupButton
                  size="icon-xs"
                  aria-label="Choose a file in the connected folder"
                  onClick={() => {
                    pickWorkspaceFile(workspace)
                      .then((source) => {
                        if (source !== null) {
                          props.handleChange(props.path, source);
                        }
                      })
                      .catch((error: unknown) => {
                        console.error("Failed to pick a data source", error);
                        void alert({
                          title: "Cannot use this file",
                          body:
                            error instanceof Error
                              ? error.message
                              : String(error),
                        });
                      });
                  }}
                />
              }
            >
              <FolderOpenIcon />
            </TooltipTrigger>
            <TooltipContent>
              Choose a file in the connected folder
            </TooltipContent>
          </Tooltip>
        </InputGroupAddon>
      )}
    </InputGroup>
  );
});
