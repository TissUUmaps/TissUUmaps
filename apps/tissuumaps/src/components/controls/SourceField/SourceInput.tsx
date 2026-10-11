import { FileIcon, FolderOpenIcon } from "lucide-react";
import { useRef } from "react";

import { SourceUtils } from "@tissuumaps/core";

import { IconButton } from "@/components/common/icon-button";
import { useAlertDialog } from "@/components/dialogs/AlertDialog/hooks";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from "@/components/ui/input-group";
import { pickWorkspacePath } from "@/data/io/workspace";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

export type SourceInputProps = {
  value: string | undefined;
  onValueChange: (value: string | undefined) => void;
  onValueCommit?: (value: string | undefined) => void;
  disabled?: boolean;
  invalid?: boolean;
  className?: string;
};

/**
 * A text input for the source of a data source
 *
 * While a workspace is open, a file or folder in it can be picked as well. The
 * picked path is relative to the project file if the project was loaded from
 * the workspace, and workspace-relative otherwise (see
 * `SourceUtils.makeProjectPath`). A value is committed when a file or folder
 * is picked, and when the input loses focus after the value changed.
 */
export function SourceInput({
  value,
  onValueChange,
  onValueCommit,
  disabled = false,
  invalid = false,
  className,
}: SourceInputProps) {
  const workspace = useAppStore((state) => state.workspace);
  const projectSource = useProjectStore((state) => state.source);
  const alert = useAlertDialog();

  const committedValueRef = useRef(value);
  const commit = (value: string | undefined) => {
    committedValueRef.current = value;
    onValueCommit?.(value);
  };

  const pick = (
    workspace: FileSystemDirectoryHandle,
    kind: FileSystemHandleKind,
  ) => {
    pickWorkspacePath(workspace, kind)
      .then((workspacePath) => {
        if (workspacePath !== null) {
          const source = SourceUtils.makeProjectPath(
            workspacePath,
            projectSource,
          );
          onValueChange(source);
          commit(source);
        }
      })
      .catch((error: unknown) => {
        console.error("Failed to pick a data source", error);
        void alert({
          title: "Cannot choose a source",
          body: error instanceof Error ? error.message : String(error),
        });
      });
  };

  return (
    <InputGroup data-disabled={disabled} className={className}>
      <InputGroupInput
        type="text"
        disabled={disabled}
        aria-invalid={invalid}
        value={value ?? ""}
        onChange={(event) =>
          onValueChange(
            event.target.value !== "" ? event.target.value : undefined,
          )
        }
        onBlur={() => {
          if (value !== committedValueRef.current) {
            commit(value);
          }
        }}
      />
      {workspace !== null && (
        <InputGroupAddon align="inline-end">
          <IconButton
            label="Choose a file in the connected folder"
            render={<InputGroupButton size="icon-xs" />}
            disabled={disabled}
            onClick={() => pick(workspace, "file")}
          >
            <FileIcon />
          </IconButton>
          <IconButton
            label="Choose a folder in the connected folder"
            render={<InputGroupButton size="icon-xs" />}
            disabled={disabled}
            onClick={() => pick(workspace, "directory")}
          >
            <FolderOpenIcon />
          </IconButton>
        </InputGroupAddon>
      )}
    </InputGroup>
  );
}
