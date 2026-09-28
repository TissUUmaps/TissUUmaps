import { useCallback } from "react";

import { SourceUtils } from "@tissuumaps/core";

import { useAlertDialog } from "@/components/dialogs/AlertDialog/hooks";
import { useConfirmDialog } from "@/components/dialogs/ConfirmDialog/hooks";
import { usePromptDialog } from "@/components/dialogs/PromptDialog/hooks";
import {
  clearProjectURLParam,
  forgetSourceFile,
  hasUnsavedChanges,
  loadProjectFromFile,
  loadProjectFromFileHandle,
  loadProjectFromURL,
  makeProjectFileName,
  makeProjectLink,
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
 * Returns a callback that asks the user to confirm discarding the open
 * project's unsaved changes
 *
 * @returns The callback, which resolves to `true` right away when there is
 * nothing unsaved to lose
 */
function useConfirmDiscard(): (
  title: string,
  body: string,
) => Promise<boolean> {
  const confirm = useConfirmDialog();

  return useCallback(
    (title: string, body: string) =>
      hasUnsavedChanges(projectStore.getState())
        ? confirm({ title, body })
        : Promise.resolve(true),
    [confirm],
  );
}

/**
 * Returns a callback that asks the user to confirm opening another project
 * over the open project's unsaved changes
 *
 * @returns The callback, which resolves to `true` right away when there is
 * nothing unsaved to lose
 */
function useConfirmOpen(): () => Promise<boolean> {
  const confirmDiscard = useConfirmDiscard();

  return useCallback(
    () =>
      confirmDiscard(
        "Open project",
        "Are you sure you want to open another project? All unsaved changes will be lost.",
      ),
    [confirmDiscard],
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
 * Returns a callback that lets the user pick a project file, within the
 * workspace if one is open, and loads it, after confirmation if the open
 * project has unsaved changes
 *
 * Failures are logged, as the callback is called from event handlers.
 *
 * @returns The callback
 */
export function useOpenProjectFile(): () => void {
  const workspace = useAppStore((state) => state.workspace);
  const setProjectOpen = useAppStore((state) => state.setProjectOpen);
  const confirmOpen = useConfirmOpen();

  // Only a file handle can be located within the workspace
  return useCallback(() => {
    const onLoaded = () => {
      clearProjectURLParam();
      setProjectOpen(true);
    };
    if (workspace === null) {
      void pickProjectFileFromInput()
        .then(async (file) => {
          if (file !== null && (await confirmOpen())) {
            await loadProjectFromFile(file);
            onLoaded();
          }
        })
        .catch((error) => {
          console.error("Failed to load project from file", error);
        });
      return;
    }
    void pickProjectFile({ startIn: workspace })
      .then(async (projectFile) => {
        if (projectFile !== null && (await confirmOpen())) {
          await loadProjectFromFileHandle(projectFile, workspace);
          onLoaded();
        }
      })
      .catch((error) => {
        console.error("Failed to load project from file", error);
      });
  }, [workspace, setProjectOpen, confirmOpen]);
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
  const confirmOpen = useConfirmOpen();

  return useCallback(() => {
    void prompt({ title: "Enter project URL to load" })
      .then(async (value) => {
        const projectUrl = value?.trim();
        if (projectUrl && (await confirmOpen())) {
          await loadProjectFromURL(projectUrl);
          setProjectURLParam(projectUrl);
        }
      })
      .catch((error) => {
        console.error("Failed to load project from URL", error);
      });
  }, [prompt, confirmOpen]);
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
export function useStartEmptyProject(): () => void {
  const clearProject = useProjectStore((state) => state.clear);
  const setProjectOpen = useAppStore((state) => state.setProjectOpen);

  return useCallback(() => {
    clearProject();
    clearProjectURLParam();
    setProjectOpen(true);
  }, [clearProject, setProjectOpen]);
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
    void confirmDiscard(
      "Close project",
      "Are you sure you want to close the project? All unsaved changes will be lost.",
    ).then((confirmed) => {
      if (confirmed) {
        clearProject();
        clearProjectURLParam();
        setProjectOpen(false);
      }
    });
  }, [clearProject, confirmDiscard, setProjectOpen]);
}

/**
 * Returns a callback that lets the user pick the workspace, forgetting the file
 * the open project was loaded from within the previous one
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
          forgetSourceFile();
        }
      })
      .catch((error) => {
        console.error("Failed to open workspace", error);
      });
  }, [setWorkspace]);
}

/**
 * Returns a callback that closes the workspace, after confirmation, and
 * forgets the file the open project was loaded from within it
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
        forgetSourceFile();
      }
    });
  }, [confirm, setWorkspace]);
}
