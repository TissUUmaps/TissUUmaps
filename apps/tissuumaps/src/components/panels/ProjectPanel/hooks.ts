import { useCallback } from "react";

import { SourceUtils } from "@tissuumaps/core";

import { useAlertDialog } from "@/components/dialogs/AlertDialog/hooks";
import type { ConfirmDialogParams } from "@/components/dialogs/ConfirmDialog";
import { useConfirmDialog } from "@/components/dialogs/ConfirmDialog/hooks";
import { usePromptDialog } from "@/components/dialogs/PromptDialog/hooks";
import {
  clearProjectURLParam,
  hasUnsavedChanges,
  loadProjectFromFile,
  loadProjectFromFileHandle,
  loadProjectFromURL,
  makeProjectFileName,
  makeProjectLink,
  resolveProjectSource,
  saveProjectAs,
  saveProjectToSourceFile,
  setProjectURLParam,
} from "@/data/io/project";
import {
  pickProjectFile,
  pickWorkspace,
  pickWorkspaceSaveFile,
} from "@/data/io/workspace";
import { useAppStore } from "@/stores/app";
import { projectStore, useProjectStore } from "@/stores/project";

import { pickProjectFileFromInput } from "./pickProjectFileFromInput";

/** Why a folder cannot be opened, for browsers without the folder picker */
export const workspaceUnsupportedMessage =
  "Opening a folder needs a Chromium-based browser, such as Chrome or Edge.";

/**
 * Returns a callback that lets the user pick the workspace
 *
 * Failures are logged, as the callback is called from event handlers.
 *
 * @returns The callback
 */
export function useOpenWorkspace(): () => void {
  const setWorkspace = useAppStore((state) => state.setWorkspace);

  return useCallback(() => {
    void pickWorkspace()
      .then((directory) => {
        if (directory !== null) {
          setWorkspace(directory);
        }
      })
      .catch((error) => {
        console.error("Failed to open workspace", error);
      });
  }, [setWorkspace]);
}

/**
 * Returns a callback that closes the workspace, after confirmation
 *
 * @returns The callback
 */
export function useCloseWorkspace(): () => void {
  const setWorkspace = useAppStore((state) => state.setWorkspace);
  const confirm = useConfirmDialog();

  return useCallback(() => {
    void confirm({
      title: "Disconnect folder",
      body: "Are you sure you want to disconnect the folder? Data loaded from it will no longer be available until you connect it again.",
    }).then((confirmed) => {
      if (confirmed) {
        setWorkspace(null);
      }
    });
  }, [confirm, setWorkspace]);
}

/**
 * Returns a callback that asks the user to confirm discarding the open
 * project's unsaved changes
 *
 * @returns The callback, which resolves to `true` right away when there is
 * nothing unsaved to lose
 */
function useConfirmDiscard(): (
  params: ConfirmDialogParams,
) => Promise<boolean> {
  const confirm = useConfirmDialog();

  return useCallback(
    (params: ConfirmDialogParams) =>
      hasUnsavedChanges(projectStore.getState())
        ? confirm(params)
        : Promise.resolve(true),
    [confirm],
  );
}

/**
 * Returns a callback that logs a failed save and reports it in an alert dialog
 *
 * @returns The callback
 */
function useReportSaveError(): (error: unknown) => void {
  const alert = useAlertDialog();

  return useCallback(
    (error: unknown) => {
      console.error("Failed to save project", error);
      void alert({
        title: "Cannot save the project",
        body: error instanceof Error ? error.message : String(error),
      });
    },
    [alert],
  );
}

/**
 * Returns a callback that saves the open project back to its file in the
 * workspace
 *
 * Failures are logged and reported in an alert dialog.
 *
 * @returns The callback, or `null` if the open project was not loaded from a
 * file in the workspace
 */
export function useSaveProjectToFolder(): (() => void) | null {
  const canSave = useProjectStore((state) => state.sourceFile !== null);
  const reportSaveError = useReportSaveError();

  const saveProjectToFolder = useCallback(() => {
    saveProjectToSourceFile().catch(reportSaveError);
  }, [reportSaveError]);
  return canSave ? saveProjectToFolder : null;
}

/**
 * Returns a callback that lets the user choose a new file within the workspace
 * and saves the open project to it
 *
 * Failures are logged and reported in an alert dialog.
 *
 * @returns The callback, or `null` if no workspace is open
 */
export function useSaveProjectToFolderAs(): (() => void) | null {
  const workspace = useAppStore((state) => state.workspace);
  const reportSaveError = useReportSaveError();

  const saveProjectToFolderAs = useCallback(() => {
    if (workspace === null) {
      return;
    }
    const { name } = projectStore.getState();
    pickWorkspaceSaveFile(workspace, makeProjectFileName(name))
      .then(async (picked) => {
        if (picked !== null) {
          await saveProjectAs(picked.file, picked.source, workspace);
          clearProjectURLParam();
        }
      })
      .catch(reportSaveError);
  }, [workspace, reportSaveError]);
  return workspace !== null ? saveProjectToFolderAs : null;
}

/**
 * Returns a callback that copies a link that opens the project, and shows it
 *
 * @returns The callback, and why no link to the open project can be copied, or
 * `null` if it can
 */
export function useCopyProjectLink(): {
  copyProjectLink: () => void;
  unavailableReason: string | null;
} {
  const source = useProjectStore((state) => state.source);
  const alert = useAlertDialog();

  // Only a project loaded from a URL can be opened by a link
  const shareableProjectUrl =
    source !== null && !SourceUtils.isWorkspacePath(source) ? source : null;

  const copyProjectLink = useCallback(() => {
    if (shareableProjectUrl === null) {
      return;
    }
    const link = makeProjectLink(shareableProjectUrl);
    navigator.clipboard
      .writeText(link)
      .then(() => alert({ title: "Link copied", body: link }))
      .catch((error) => {
        console.error("Failed to copy project link", error);
        void alert({ title: "Cannot copy the link", body: link });
      });
  }, [shareableProjectUrl, alert]);

  const unavailableReason =
    shareableProjectUrl === null
      ? "Only for projects opened from a URL"
      : !window.isSecureContext
        ? "Copying needs a secure (https) connection"
        : null;
  return { copyProjectLink, unavailableReason };
}

/**
 * Returns a callback that replaces the open project with an empty one, without
 * confirmation
 *
 * @returns The callback
 */
export function useOpenEmptyProject(): () => void {
  const clearProject = useProjectStore((state) => state.clear);
  const setProjectOpen = useAppStore((state) => state.setProjectOpen);

  return useCallback(() => {
    clearProject();
    clearProjectURLParam();
    setProjectOpen(true);
  }, [clearProject, setProjectOpen]);
}

/**
 * Returns a callback that asks the user for a project URL, and loads the
 * project from it, after confirmation if the open project has unsaved changes
 *
 * Failures are logged, as the callback is called from event handlers.
 *
 * @returns The callback
 */
export function useOpenProjectFromURL(): () => void {
  const prompt = usePromptDialog();
  const confirmDiscard = useConfirmDiscard();

  return useCallback(() => {
    void prompt({ title: "Enter project URL to load" })
      .then(async (value) => {
        const projectUrl = value?.trim();
        if (
          projectUrl &&
          (await confirmDiscard({
            title: "Open project",
            body: `Open the project "${projectUrl}"? It replaces the open project, and unsaved changes are lost.`,
            actionLabel: "Open",
          }))
        ) {
          await loadProjectFromURL(projectUrl);
          setProjectURLParam(projectUrl);
        }
      })
      .catch((error) => {
        console.error("Failed to load project from URL", error);
      });
  }, [prompt, confirmDiscard]);
}

/**
 * Returns a callback that lets the user pick a project file, within the
 * workspace if one is open, and loads it, after confirmation if needed (see
 * {@link useConfirmOpenProjectFile})
 *
 * Failures are logged, as the callback is called from event handlers.
 *
 * @returns The callback
 */
export function useOpenProjectFromFile(): () => void {
  const workspace = useAppStore((state) => state.workspace);
  const confirmOpenProjectFile = useConfirmOpenProjectFile();
  const loadProjectFile = useLoadProjectFile();

  return useCallback(() => {
    const pickedProjectFile =
      workspace === null
        ? pickProjectFileFromInput()
        : pickProjectFile({ startIn: workspace });
    void pickedProjectFile
      .then(async (projectFile) => {
        if (
          projectFile !== null &&
          (await confirmOpenProjectFile(projectFile))
        ) {
          await loadProjectFile(projectFile);
        }
      })
      .catch((error) => {
        console.error("Failed to load project from file", error);
      });
  }, [workspace, confirmOpenProjectFile, loadProjectFile]);
}

/**
 * Returns a callback that asks the user to confirm opening a project file
 *
 * It asks only if the open project has unsaved changes, or if the file lies
 * outside the open workspace, as its project-relative paths are then resolved
 * against the workspace instead of the file's directory (see
 * `loadProjectFromFileHandle`).
 *
 * @returns The callback, which resolves to whether to open the file
 */
export function useConfirmOpenProjectFile(): (
  projectFile: FileSystemFileHandle | File,
) => Promise<boolean> {
  const workspace = useAppStore((state) => state.workspace);
  const confirm = useConfirmDialog();

  return useCallback(
    async (projectFile) => {
      const unsavedChanges = hasUnsavedChanges(projectStore.getState());
      const isOutsideWorkspace =
        workspace !== null &&
        (projectFile instanceof File ||
          (await resolveProjectSource(projectFile, workspace)) === null);
      if (!unsavedChanges && !isOutsideWorkspace) {
        return true;
      }
      const question = unsavedChanges
        ? `Open the project "${projectFile.name}"? It replaces the open project, and unsaved changes are lost.`
        : `Open the project "${projectFile.name}"?`;
      return await confirm({
        title: "Open project",
        body: isOutsideWorkspace
          ? `"${projectFile.name}" is not in the connected folder "${workspace.name}". Its relative paths will be looked up in "${workspace.name}" instead of next to the project file, so some of its data may fail to load. ${question}`
          : question,
        actionLabel: "Open",
      });
    },
    [workspace, confirm],
  );
}

/**
 * Returns a function that loads a project from a file and marks it open
 *
 * A file handle is loaded with its source if it lies within the workspace
 * (see `loadProjectFromFileHandle`), a file without one. The project URL
 * parameter is cleared, as it names another project.
 *
 * @returns The function, which rejects if the project cannot be read or
 * parsed
 */
export function useLoadProjectFile(): (
  projectFile: FileSystemFileHandle | File,
) => Promise<void> {
  const workspace = useAppStore((state) => state.workspace);
  const setProjectOpen = useAppStore((state) => state.setProjectOpen);

  return useCallback(
    async (projectFile) => {
      if (projectFile instanceof File) {
        await loadProjectFromFile(projectFile);
      } else {
        await loadProjectFromFileHandle(projectFile, workspace);
      }
      clearProjectURLParam();
      setProjectOpen(true);
    },
    [workspace, setProjectOpen],
  );
}

/**
 * Returns a callback that closes the open project, after confirmation if it
 * has unsaved changes
 *
 * @returns The callback
 */
export function useCloseProject(): () => void {
  const clearProject = useProjectStore((state) => state.clear);
  const setProjectOpen = useAppStore((state) => state.setProjectOpen);
  const confirmDiscard = useConfirmDiscard();

  return useCallback(() => {
    void confirmDiscard({
      title: "Close project",
      body: "Are you sure you want to close the project? All unsaved changes will be lost.",
    }).then((confirmed) => {
      if (confirmed) {
        clearProject();
        clearProjectURLParam();
        setProjectOpen(false);
      }
    });
  }, [clearProject, confirmDiscard, setProjectOpen]);
}
