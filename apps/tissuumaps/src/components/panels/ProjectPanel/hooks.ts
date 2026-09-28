import { useCallback } from "react";

import { useConfirmDialog } from "@/components/dialogs/ConfirmDialog/hooks";
import { usePromptDialog } from "@/components/dialogs/PromptDialog/hooks";
import {
  clearProjectURLParam,
  loadProjectFromFile,
  loadProjectFromFileHandle,
  loadProjectFromURL,
  setProjectURLParam,
} from "@/data/io/project";
import { pickProjectFile, pickWorkspace } from "@/data/io/workspace";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

import { pickProjectFileFromInput } from "./pickProjectFileFromInput";

/** Why a folder cannot be opened, for browsers without the folder picker */
export const workspaceUnsupportedMessage =
  "Opening a folder needs a Chromium-based browser, such as Chrome or Edge.";

/**
 * Returns a callback that lets the user pick a project file, within the
 * workspace if one is open, and loads it
 *
 * Failures are logged, as the callback is called from event handlers.
 *
 * @returns The callback
 */
export function useOpenProjectFile(): () => void {
  const workspace = useAppStore((state) => state.workspace);
  const setProjectOpen = useAppStore((state) => state.setProjectOpen);

  // Only a file handle can be located within the workspace
  return useCallback(() => {
    const onLoaded = () => {
      clearProjectURLParam();
      setProjectOpen(true);
    };
    if (workspace === null) {
      void pickProjectFileFromInput()
        .then(async (file) => {
          if (file !== null) {
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
        if (projectFile !== null) {
          await loadProjectFromFileHandle(projectFile, workspace);
          onLoaded();
        }
      })
      .catch((error) => {
        console.error("Failed to load project from file", error);
      });
  }, [workspace, setProjectOpen]);
}

/**
 * Returns a callback that asks the user for a project URL, and loads the
 * project from it
 *
 * Failures are logged, as the callback is called from event handlers.
 *
 * @returns The callback
 */
export function useOpenProjectFromURL(): () => void {
  const prompt = usePromptDialog();

  return useCallback(() => {
    void prompt({ title: "Enter project URL to load" })
      .then(async (value) => {
        const projectUrl = value?.trim();
        if (projectUrl) {
          await loadProjectFromURL(projectUrl);
          setProjectURLParam(projectUrl);
        }
      })
      .catch((error) => {
        console.error("Failed to load project from URL", error);
      });
  }, [prompt]);
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
 * Returns a callback that closes the open project, after confirmation
 *
 * @returns The callback
 */
export function useCloseProject(): () => void {
  const clearProject = useProjectStore((state) => state.clear);
  const setProjectOpen = useAppStore((state) => state.setProjectOpen);
  const confirm = useConfirmDialog();

  return useCallback(() => {
    void confirm({
      title: "Close project",
      body: "Are you sure you want to close the project? All unsaved changes will be lost.",
    }).then((confirmed) => {
      if (confirmed) {
        clearProject();
        clearProjectURLParam();
        setProjectOpen(false);
      }
    });
  }, [clearProject, confirm, setProjectOpen]);
}

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
