import { useCallback } from "react";

import { type ProjectProvider, ProjectUtils } from "@tissuumaps/core";
import { tissuumapsProjectType } from "@tissuumaps/storage";

import { useAlertDialog } from "@/components/dialogs/AlertDialog/hooks";
import { useChooseDialog } from "@/components/dialogs/ChooseDialog/hooks";
import { useConfirmDialog } from "@/components/dialogs/ConfirmDialog/hooks";
import { usePromptDialog } from "@/components/dialogs/PromptDialog/hooks";
import {
  appendProjectURLParam,
  clearProjectURLParams,
  downloadProject,
  findLocalProjectProviders,
  findProjectProviders,
  getProjectFileExtensions,
  getProjectFileTypes,
  getProjectProvider,
  loadLocalProject,
  loadProjectFromURL,
  mergeProject,
  openProject,
  readProjectURLParams,
  resolveProjectSource,
  setProjectURLParam,
} from "@/data/io/project";
import { pickProjectFile, pickWorkspace } from "@/data/io/workspace";
import { pickFileFromInput } from "@/lib/pickFileFromInput";
import { useAppStore } from "@/stores/app";
import { projectStore, useProjectStore } from "@/stores/project";

/** Why a workspace cannot be opened, for browsers without the folder picker */
export const workspaceUnsupportedMessage =
  "Opening a workspace needs a Chromium-based browser, such as Chrome or Edge.";

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
      title: "Close workspace",
      body: "Are you sure you want to close the workspace? Data loaded from it will no longer be available until you open it again.",
      actionLabel: "Close",
    }).then((confirmed) => {
      if (confirmed) {
        setWorkspace(null);
      }
    });
  }, [confirm, setWorkspace]);
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
    clearProjectURLParams();
    setProjectOpen(true);
  }, [clearProject, setProjectOpen]);
}

/**
 * Returns a function that settles on the project provider to load a project
 * with, asking the user unless there is exactly one to choose from
 *
 * If several were found, the user chooses one of them, the first one
 * preselected; if none was, the user chooses one of all registered project
 * providers that implement `load`.
 *
 * @returns The function, which takes the name of what is loaded, the project
 * providers found for it, and optionally a note to show before the choice and
 * the label of the action, and resolves to the project provider, or to `null`
 * if the user cancelled
 */
export function useChooseProjectProvider(): (
  name: string,
  foundProjectProviders: ProjectProvider[],
  options?: { note?: string; actionLabel?: string },
) => Promise<ProjectProvider | null> {
  const projectProviders = useAppStore((state) => state.projectProviders);

  const choose = useChooseDialog();

  return useCallback(
    async (name, foundProjectProviders, options) => {
      const choices = Array.from(projectProviders).filter(
        ([, projectProvider]) =>
          foundProjectProviders.length > 0
            ? foundProjectProviders.includes(projectProvider)
            : projectProvider.load !== undefined,
      );
      if (choices.length === 0) {
        throw new Error("No project provider can load projects.");
      }
      if (choices.length === 1) {
        const choice = choices[0]!;
        return choice[1];
      }
      const question =
        foundProjectProviders.length > 0
          ? `"${name}" matches several project formats. Which one is it?`
          : `The project format of "${name}" was not recognized. Which one is it?`;
      const type = await choose({
        title: "Choose project format",
        body:
          options?.note !== undefined
            ? `${options.note} ${question}`
            : question,
        options: choices.map(([type, projectProvider]) => ({
          value: type,
          label: projectProvider.name,
        })),
        actionLabel: options?.actionLabel,
      });
      return type !== null ? (projectProviders.get(type) ?? null) : null;
    },
    [projectProviders, choose],
  );
}

/**
 * Returns a callback that asks the user for a project URL, and loads the
 * project from it with the project provider that supports it, or the one the
 * user chooses if none or several do (see `useChooseProjectProvider`)
 *
 * Failures are logged and shown, as the callback is called from event
 * handlers.
 *
 * @returns The callback
 */
export function useOpenProjectFromURL(): () => void {
  const workspace = useAppStore((state) => state.workspace);

  const chooseProjectProvider = useChooseProjectProvider();

  const alert = useAlertDialog();
  const prompt = usePromptDialog();

  return useCallback(() => {
    void prompt({ title: "Enter project URL to load", actionLabel: "Open" })
      .then(async (value) => {
        const projectUrl = value?.trim();
        if (!projectUrl) {
          return;
        }
        const projectProvider = await chooseProjectProvider(
          projectUrl,
          await findProjectProviders(
            new URL(projectUrl, document.baseURI).href,
            workspace,
          ),
          { actionLabel: "Open" },
        );
        if (projectProvider === null) {
          return;
        }
        openProject(await loadProjectFromURL(projectProvider, projectUrl));
        setProjectURLParam(projectProvider, projectUrl);
      })
      .catch(async (error: unknown) => {
        console.error("Failed to load project from URL", error);
        await alert({
          title: "Cannot open the project",
          body: error instanceof Error ? error.message : String(error),
        });
      });
  }, [workspace, chooseProjectProvider, prompt, alert]);
}

/**
 * Returns a callback that lets the user pick a project file, within the
 * workspace if one is open, and loads it
 *
 * The file is loaded with the project provider that loads files with its
 * extension, or the one the user chooses if none or several do (see
 * `useChooseProjectProvider`). A project file picked outside the open
 * workspace is loaded only after confirmation, as its project-relative paths
 * are then resolved against the workspace instead of the project file's
 * directory (see `loadLocalProject`). Failures are logged and shown,
 * as the callback is called from event handlers.
 *
 * @returns The callback
 */
export function useOpenProjectFromFile(): () => void {
  const workspace = useAppStore((state) => state.workspace);

  const loadProjectFile = useLoadProjectFile();

  const chooseProjectProvider = useChooseProjectProvider();

  const alert = useAlertDialog();
  const confirm = useConfirmDialog();

  return useCallback(() => {
    const pickedProjectFile =
      workspace === null
        ? pickFileFromInput(getProjectFileExtensions())
        : pickProjectFile(getProjectFileTypes(), { startIn: workspace });
    void pickedProjectFile
      .then(async (projectFile) => {
        if (projectFile === null) {
          return;
        }
        if (
          workspace !== null &&
          (projectFile instanceof File ||
            (await resolveProjectSource(projectFile, workspace)) === null) &&
          !(await confirm({
            title: "Open project from outside the workspace",
            body: `"${projectFile.name}" is not in the workspace "${workspace.name}". Its relative paths will be looked up in "${workspace.name}" instead of next to the project file, so some of its data may fail to load. Open it anyway?`,
            actionLabel: "Open",
          }))
        ) {
          return;
        }
        const projectProvider = await chooseProjectProvider(
          projectFile.name,
          findLocalProjectProviders(projectFile),
          { actionLabel: "Open" },
        );
        if (projectProvider !== null) {
          await loadProjectFile(projectFile, projectProvider);
        }
      })
      .catch(async (error: unknown) => {
        console.error("Failed to load project from file", error);
        await alert({
          title: "Cannot open the project",
          body: error instanceof Error ? error.message : String(error),
        });
      });
  }, [workspace, loadProjectFile, chooseProjectProvider, confirm, alert]);
}

/**
 * Returns a function that loads a project from a local file or directory with
 * a project provider and marks it open
 *
 * A file or directory within the workspace is loaded with its source (see
 * `loadLocalProject`). The project URL parameters are cleared, as they name
 * other projects.
 *
 * @returns The function, which rejects if the project cannot be read or
 * parsed
 */
export function useLoadProjectFile(): (
  projectFile: File | FileSystemFileHandle | FileSystemDirectoryHandle,
  projectProvider: ProjectProvider,
) => Promise<void> {
  const workspace = useAppStore((state) => state.workspace);
  const setProjectOpen = useAppStore((state) => state.setProjectOpen);

  return useCallback(
    async (projectFile, projectProvider) => {
      openProject(
        await loadLocalProject(projectProvider, projectFile, workspace),
      );
      clearProjectURLParams();
      setProjectOpen(true);
    },
    [workspace, setProjectOpen],
  );
}

/**
 * Returns a callback that asks the user for a project URL, and imports the
 * project from it into the open project with the project provider that
 * supports it, or the one the user chooses if none or several do (see
 * `useChooseProjectProvider`)
 *
 * The imported project is recorded in the address bar (see
 * `appendProjectURLParam`) if the open project was loaded from URLs recorded
 * there, or has no data yet, so that a reload never opens the imported
 * project alone. Failures are logged and shown, as the callback is
 * called from event handlers.
 *
 * @returns The callback
 */
export function useImportProjectFromURL(): () => void {
  const workspace = useAppStore((state) => state.workspace);

  const chooseProjectProvider = useChooseProjectProvider();

  const alert = useAlertDialog();
  const prompt = usePromptDialog();

  return useCallback(() => {
    void prompt({ title: "Enter project URL to import", actionLabel: "Import" })
      .then(async (value) => {
        const projectUrl = value?.trim();
        if (!projectUrl) {
          return;
        }
        const projectProvider = await chooseProjectProvider(
          projectUrl,
          await findProjectProviders(
            new URL(projectUrl, document.baseURI).href,
            workspace,
          ),
          { actionLabel: "Import" },
        );
        if (projectProvider === null) {
          return;
        }
        const loadedProject = await loadProjectFromURL(
          projectProvider,
          projectUrl,
        );
        const recordProject =
          readProjectURLParams(window.location.search).length > 0 ||
          !ProjectUtils.hasData(projectStore.getState());
        mergeProject(loadedProject);
        if (recordProject) {
          appendProjectURLParam(projectProvider, projectUrl);
        }
      })
      .catch(async (error: unknown) => {
        console.error("Failed to import project from URL", error);
        await alert({
          title: "Cannot import the project",
          body: error instanceof Error ? error.message : String(error),
        });
      });
  }, [workspace, chooseProjectProvider, prompt, alert]);
}

/**
 * Returns a callback that lets the user pick a project file, within the
 * workspace if one is open, and imports it into the open project
 *
 * The file is imported with the project provider that loads files with its
 * extension, or the one the user chooses if none or several do (see
 * `useChooseProjectProvider`). A project file picked outside the open
 * workspace is imported only after confirmation, as for opening (see
 * `useOpenProjectFromFile`). The project URL parameters are left as they are,
 * as a file cannot be recorded in them. Failures are logged and shown, as the
 * callback is called from event handlers.
 *
 * @returns The callback
 */
export function useImportProjectFromFile(): () => void {
  const workspace = useAppStore((state) => state.workspace);

  const chooseProjectProvider = useChooseProjectProvider();

  const alert = useAlertDialog();
  const confirm = useConfirmDialog();

  return useCallback(() => {
    const pickedProjectFile =
      workspace === null
        ? pickFileFromInput(getProjectFileExtensions())
        : pickProjectFile(getProjectFileTypes(), { startIn: workspace });
    void pickedProjectFile
      .then(async (projectFile) => {
        if (projectFile === null) {
          return;
        }
        if (
          workspace !== null &&
          (projectFile instanceof File ||
            (await resolveProjectSource(projectFile, workspace)) === null) &&
          !(await confirm({
            title: "Import project from outside the workspace",
            body: `"${projectFile.name}" is not in the workspace "${workspace.name}". Its relative paths will be looked up in "${workspace.name}" instead of next to the project file, so some of its data may fail to load. Import it anyway?`,
            actionLabel: "Import",
          }))
        ) {
          return;
        }
        const projectProvider = await chooseProjectProvider(
          projectFile.name,
          findLocalProjectProviders(projectFile),
          { actionLabel: "Import" },
        );
        if (projectProvider === null) {
          return;
        }
        mergeProject(
          await loadLocalProject(projectProvider, projectFile, workspace),
        );
      })
      .catch(async (error: unknown) => {
        console.error("Failed to import project from file", error);
        await alert({
          title: "Cannot import the project",
          body: error instanceof Error ? error.message : String(error),
        });
      });
  }, [workspace, chooseProjectProvider, confirm, alert]);
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
      actionLabel: "Close",
    }).then((confirmed) => {
      if (confirmed) {
        clearProject();
        clearProjectURLParams();
        setProjectOpen(false);
      }
    });
  }, [clearProject, confirm, setProjectOpen]);
}

/**
 * Returns a callback that downloads the open project with a project provider,
 * by default as a TissUUmaps project file
 *
 * Failures are logged and shown, as the callback is called from event
 * handlers.
 *
 * @returns The callback, which takes the project provider to write the
 * project with
 */
export function useDownloadProject(): (
  projectProvider?: ProjectProvider,
) => void {
  const alert = useAlertDialog();

  return useCallback(
    (projectProvider) => {
      void downloadProject(
        projectProvider ?? getProjectProvider(tissuumapsProjectType),
      ).catch(async (error: unknown) => {
        console.error("Failed to download project", error);
        await alert({
          title: "Cannot download the project",
          body: error instanceof Error ? error.message : String(error),
        });
      });
    },
    [alert],
  );
}
