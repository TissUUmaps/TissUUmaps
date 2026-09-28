import type { CellProps } from "@jsonforms/core";
import { withJsonFormsCellProps } from "@jsonforms/react";
import { FolderOpenIcon } from "lucide-react";

import { IconButton } from "@/components/common/icon-button";
import { useAlertDialog } from "@/components/dialogs/AlertDialog/hooks";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
import {
  pickWorkspaceDirectoryPath,
  pickWorkspaceFilePath,
} from "@/data/io/workspace";
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
          <IconButton
            label="Choose from the connected folder"
            render={<InputGroupButton size="icon-xs" />}
            onClick={() => {
              (options.directory === true
                ? pickWorkspaceDirectoryPath(workspace)
                : pickWorkspaceFilePath(workspace)
              )
                .then((source) => {
                  if (source !== null) {
                    props.handleChange(props.path, source);
                  }
                })
                .catch((error: unknown) => {
                  console.error("Failed to pick a data source", error);
                  void alert({
                    title: "Cannot use this source",
                    body:
                      error instanceof Error ? error.message : String(error),
                  });
                });
            }}
          >
            <FolderOpenIcon />
          </IconButton>
        </InputGroupAddon>
      )}
    </InputGroup>
  );
});
