import { useCallback } from "react";

import {
  type ProjectProvider,
  ProjectUtils,
  SourceUtils,
} from "@tissuumaps/core";
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
  hasUnsavedChanges,
  loadLocalProject,
  loadProjectFromURL,
  mergeProject,
  openProject,
  readProjectURLParams,
  resolveProjectSource,
  saveProjectCopyToFile,
  saveProjectToFile,
  setProjectURLParam,
} from "@/data/io/project";
import {
  pickProjectFile,
  pickProjectSaveFile,
  pickWorkspace,
} from "@/data/io/workspace";
import { pickFileFromInput } from "@/lib/pickFileFromInput";
import { useAppStore } from "@/stores/app";
import { projectStore, useProjectStore } from "@/stores/project";

/** Why a workspace cannot be opened, for browsers without the folder picker */
export const workspaceUnsupportedMessage =
  "Opening a workspace needs a Chromium-based browser, such as Chrome or Edge.";

/**
 * Returns a callback that lets the user pick the workspace
 *
 * Failures are logged and shown, as the callback is called from event
 * handlers.
 *
 * @returns The callback
 */
export function useOpenWorkspace(): () => void {
  const setWorkspace = useAppStore((state) => state.setWorkspace);

  const alert = useAlertDialog();

  return useCallback(() => {
    void pickWorkspace()
      .then((directory) => {
        if (directory !== null) {
          setWorkspace(directory);
        }
      })
      .catch(async (error: unknown) => {
        console.error("Failed to open workspace", error);
        await alert({
          title: "Cannot open the workspace",
          body: error instanceof Error ? error.message : String(error),
        });
      });
  }, [setWorkspace, alert]);
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
 * Returns a callback that saves the open project back to the file it is saved
 * to (see `ProjectStoreState.saveTarget`)
 *
 * The caller offers it only if there is such a file and its project provider
 * can write projects. Failures are logged and shown, as the callback is called
 * from event handlers.
 *
 * @returns The callback, which resolves once the project is saved or the
 * failure was shown
 */
export function useSaveProject(): () => Promise<void> {
  const alert = useAlertDialog();

  return useCallback(async () => {
    const { source, saveTarget } = projectStore.getState();
    try {
      if (saveTarget === null || source === null) {
        throw new Error(
          "The open project was not opened from or saved to the workspace",
        );
      }
      await saveProjectToFile(saveTarget, source);
    } catch (error) {
      console.error("Failed to save project", error);
      await alert({
        title: "Cannot save the project",
        body: error instanceof Error ? error.message : String(error),
      });
    }
  }, [alert]);
}

/**
 * Returns a callback that lets the user pick a new project file, starting in
 * the workspace, and saves the open project to it
 *
 * The project is written with the project provider of the file it is saved
 * to if that can write projects, and as a TissUUmaps project file otherwise.
 * A file within the workspace becomes the one the project is saved to (see
 * `saveProjectToFile`). A file outside it only gets a copy, as the browser
 * has already created or emptied it (see `saveProjectCopyToFile`), and the
 * user is told so. The caller offers it only while a workspace is open.
 * Failures are logged and shown, as the callback is called from event
 * handlers.
 *
 * @returns The callback, which resolves once the project is saved, the user
 * cancelled, or the failure was shown
 */
export function useSaveProjectAs(): () => Promise<void> {
  const workspace = useAppStore((state) => state.workspace);

  const alert = useAlertDialog();

  return useCallback(async () => {
    try {
      if (workspace === null) {
        throw new Error("No workspace is open");
      }
      const { name, source, saveTarget } = projectStore.getState();
      const projectProvider =
        saveTarget?.projectProvider.toFile !== undefined
          ? saveTarget.projectProvider
          : getProjectProvider(tissuumapsProjectType);
      const fileExtension =
        Object.values(projectProvider.fileTypes ?? {}).flat()[0] ?? "";
      const stem = source !== null ? SourceUtils.getStem(source) : "";
      const picked = await pickProjectSaveFile(
        workspace,
        `${stem || name || "Untitled"}${fileExtension}`,
        projectProvider.fileTypes !== undefined
          ? [
              {
                description: `${projectProvider.name} project`,
                accept: projectProvider.fileTypes,
              },
            ]
          : [],
      );
      if (picked === null) {
        return;
      }
      const target = { handle: picked.handle, projectProvider };
      if (picked.source !== null) {
        await saveProjectToFile(target, picked.source);
        return;
      }
      await saveProjectCopyToFile(target);
      await alert({
        title: "Copy saved outside the workspace",
        body: `"${picked.handle.name}" is not in the workspace "${workspace.name}", so only a copy was saved to it, as for a download. Its paths were not rewritten, and the open project is unchanged.`,
      });
    } catch (error) {
      console.error("Failed to save project", error);
      await alert({
        title: "Cannot save the project",
        body: error instanceof Error ? error.message : String(error),
      });
    }
  }, [workspace, alert]);
}

/**
 * Returns a callback that copies a link that opens the open project, and
 * shows it
 *
 * The link is the current address, which names the projects opened and
 * imported from URLs (see `setProjectURLParam`); if it names none, the user
 * is told that there is no link. Copying needs a secure context, where the
 * clipboard is available, which the caller checks. Failures are logged and
 * shown, as the callback is called from event handlers.
 *
 * @returns The callback
 */
export function useCopyProjectLink(): () => void {
  const alert = useAlertDialog();

  return useCallback(() => {
    if (readProjectURLParams(window.location.search).length === 0) {
      void alert({
        title: "Cannot copy the link",
        body: "Only projects opened from a URL have a link to share.",
      });
      return;
    }
    const link = window.location.href;
    navigator.clipboard
      .writeText(link)
      .then(() => alert({ title: "Link copied", body: link }))
      .catch(async (error: unknown) => {
        console.error("Failed to copy project link", error);
        await alert({ title: "Cannot copy the link", body: link });
      });
  }, [alert]);
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
        throw new Error("No project provider can load projects");
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
 * The project is opened after confirmation if it would replace unsaved
 * changes. Failures are logged and shown, as the callback is called from event
 * handlers.
 *
 * @returns The callback
 */
export function useOpenProjectFromURL(): () => void {
  const workspace = useAppStore((state) => state.workspace);

  const chooseProjectProvider = useChooseProjectProvider();

  const alert = useAlertDialog();
  const confirm = useConfirmDialog();
  const prompt = usePromptDialog();

  return useCallback(() => {
    void prompt({ title: "Open project from URL", actionLabel: "Open" })
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
        if (
          projectProvider === null ||
          (hasUnsavedChanges(projectStore.getState()) &&
            !(await confirm({
              title: "Open project",
              body: `Are you sure you want to open "${projectUrl}"? The open project's unsaved changes will be lost.`,
              actionLabel: "Open",
            })))
        ) {
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
  }, [workspace, chooseProjectProvider, prompt, confirm, alert]);
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
 * directory (see `loadLocalProject`), as is any project that would replace
 * unsaved changes. Failures are logged and shown, as the callback is called
 * from event handlers.
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
        const outsideWorkspaceNote =
          workspace !== null &&
          (projectFile instanceof File ||
            (await resolveProjectSource(projectFile, workspace)) === null)
            ? `"${projectFile.name}" is not in the workspace "${workspace.name}". Its relative paths will be looked up in "${workspace.name}" instead of next to the project file, so some of its data may fail to load.`
            : null;
        const question = hasUnsavedChanges(projectStore.getState())
          ? `Are you sure you want to open "${projectFile.name}"? The open project's unsaved changes will be lost.`
          : outsideWorkspaceNote !== null
            ? "Open it anyway?"
            : null;
        if (
          question !== null &&
          !(await confirm({
            title:
              outsideWorkspaceNote !== null
                ? "Open project from outside the workspace"
                : "Open project",
            body:
              outsideWorkspaceNote !== null
                ? `${outsideWorkspaceNote} ${question}`
                : question,
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
 * A file or directory within the workspace is loaded with its source, and a
 * file within it becomes the one the project is saved to (see
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
    void prompt({ title: "Import project from URL", actionLabel: "Import" })
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
 * Returns a callback that closes the open project, after confirmation if it
 * has unsaved changes
 *
 * @returns The callback
 */
export function useCloseProject(): () => void {
  const clearProject = useProjectStore((state) => state.clear);
  const setProjectOpen = useAppStore((state) => state.setProjectOpen);
  const confirm = useConfirmDialog();

  return useCallback(() => {
    void (
      hasUnsavedChanges(projectStore.getState())
        ? confirm({
            title: "Close project",
            body: "Are you sure you want to close the project? Its unsaved changes will be lost.",
            actionLabel: "Close",
          })
        : Promise.resolve(true)
    ).then((confirmed) => {
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
