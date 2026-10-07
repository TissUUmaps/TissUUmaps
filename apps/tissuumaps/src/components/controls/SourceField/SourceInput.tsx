import { FileIcon, FolderOpenIcon } from "lucide-react";

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
 * `SourceUtils.makeProjectPath`).
 */
export function SourceInput({
  value,
  onValueChange,
  disabled = false,
  invalid = false,
  className,
}: SourceInputProps) {
  const workspace = useAppStore((state) => state.workspace);
  const projectSource = useProjectStore((state) => state.source);
  const alert = useAlertDialog();

  const pick = (
    workspace: FileSystemDirectoryHandle,
    kind: FileSystemHandleKind,
  ) => {
    pickWorkspacePath(workspace, kind)
      .then((workspacePath) => {
        if (workspacePath !== null) {
          onValueChange(
            SourceUtils.makeProjectPath(workspacePath, projectSource),
          );
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
