import { useCallback } from "react";

import { useConfirmDialog } from "@/components/dialogs/ConfirmDialog/hooks";
import { usePromptDialog } from "@/components/dialogs/PromptDialog/hooks";
import {
  clearProjectURLParam,
  loadProjectFromFile,
  loadProjectFromFileHandle,
  loadProjectFromURL,
  saveAndDownloadProjectToJSON,
  setProjectURLParam,
} from "@/data/io/project";
import {
  pickProjectFile,
  pickProjectFileFromInput,
  pickWorkspace,
} from "@/data/io/workspace";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

/** Why a folder cannot be opened, for browsers without the folder picker */
export const workspaceUnsupportedMessage =
  "Opening a folder needs a Chromium-based browser, such as Chrome or Edge.";

/** The actions of the Project panel, each safe to call from an event handler */
export type ProjectActions = {
  /**
   * Lets the user pick a project file, within the workspace if one is open,
   * and loads it
   */
  openProjectFile: () => void;
  /** Asks the user for a project URL, and loads the project from it */
  openProjectFromURL: () => void;
  /** Downloads the open project as a `.tmap` file */
  downloadProject: () => void;
  /** Closes the open project and shows the start page, after confirmation */
  closeProject: () => void;
  /** Replaces the open project with an empty one, without confirmation */
  startEmptyProject: () => void;
  /** Lets the user pick the workspace */
  openWorkspace: () => void;
  /** Closes the workspace, after confirmation */
  closeWorkspace: () => void;
};

/**
 * Provides the actions of the Project panel
 *
 * Failures are logged, as the actions are called from event handlers.
 *
 * @returns The actions, bound to the open workspace and the dialogs
 */
export function useProjectActions(): ProjectActions {
  const clearProject = useProjectStore((state) => state.clear);
  const workspace = useAppStore((state) => state.workspace);
  const setWorkspace = useAppStore((state) => state.setWorkspace);
  const setStartPageDismissed = useAppStore(
    (state) => state.setStartPageDismissed,
  );
  const confirm = useConfirmDialog();
  const prompt = usePromptDialog();

  // Only a file handle can be located within the workspace
  const openProjectFile = useCallback(() => {
    const onLoaded = () => {
      clearProjectURLParam();
      setStartPageDismissed(true);
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
  }, [workspace, setStartPageDismissed]);

  const openProjectFromURL = useCallback(() => {
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

  const downloadProject = useCallback(() => {
    saveAndDownloadProjectToJSON();
  }, []);

  const startEmptyProject = useCallback(() => {
    clearProject();
    clearProjectURLParam();
    setStartPageDismissed(true);
  }, [clearProject, setStartPageDismissed]);

  const closeProject = useCallback(() => {
    void confirm({
      title: "Close project",
      body: "Are you sure you want to close the project? All unsaved changes will be lost.",
    }).then((confirmed) => {
      if (confirmed) {
        clearProject();
        clearProjectURLParam();
        setStartPageDismissed(false);
      }
    });
  }, [clearProject, confirm, setStartPageDismissed]);

  const openWorkspace = useCallback(() => {
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

  const closeWorkspace = useCallback(() => {
    void confirm({
      title: "Disconnect folder",
      body: "Are you sure you want to disconnect the folder? Data loaded from it will no longer be available until you connect it again.",
    }).then((confirmed) => {
      if (confirmed) {
        setWorkspace(null);
      }
    });
  }, [confirm, setWorkspace]);

  return {
    openProjectFile,
    openProjectFromURL,
    downloadProject,
    closeProject,
    startEmptyProject,
    openWorkspace,
    closeWorkspace,
  };
}
