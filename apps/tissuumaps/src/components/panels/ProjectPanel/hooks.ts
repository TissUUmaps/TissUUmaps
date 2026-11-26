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
  saveProjectAs,
  saveProjectToSourceFile,
  setProjectURLParam,
} from "@/data/io/project";
import {
  pickProjectFile,
  pickWorkspace,
  pickWorkspaceSaveFile,
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
 * Returns a function that asks for confirmation before the open project is
 * replaced or closed, if it has unsaved changes or there is a note to show
 *
 * @returns The function, which takes the dialog's title, an optional note to
 * show first, the question to end with and the label of the action, and
 * resolves to whether to go ahead
 */
function useConfirmDiscard(): (params: {
  title: string;
  note?: string;
  question: string;
  actionLabel: string;
}) => Promise<boolean> {
  const confirm = useConfirmDialog();

  return useCallback(
    async ({ title, note, question, actionLabel }) => {
      const unsavedChanges = hasUnsavedChanges(projectStore.getState());
      if (note === undefined && !unsavedChanges) {
        return true;
      }
      return await confirm({
        title,
        body: [
          note,
          unsavedChanges ? "The open project has unsaved changes." : undefined,
          question,
        ]
          .filter((sentence) => sentence !== undefined)
          .join(" "),
        actionLabel,
      });
    },
    [confirm],
  );
}

/**
 * Returns a function that logs a failed save and reports it in an alert
 * dialog
 *
 * @returns The function, which takes the error
 */
function useReportSaveError(): (error: unknown) => void {
  const alert = useAlertDialog();

  return useCallback(
    (error) => {
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
 * Failures are logged and shown, as the callback is called from event
 * handlers.
 *
 * @returns The callback, or `null` if the open project has no file in the
 * workspace, or if the project provider of that file cannot write projects
 * (see `ProjectStoreState.sourceFile`)
 */
export function useSaveProjectToSourceFile(): (() => void) | null {
  const canSave = useProjectStore(
    (state) =>
      state.sourceFile !== null &&
      state.sourceFile.projectProvider.toFile !== undefined,
  );
  const reportSaveError = useReportSaveError();

  const saveToSourceFile = useCallback(() => {
    saveProjectToSourceFile().catch(reportSaveError);
  }, [reportSaveError]);
  return canSave ? saveToSourceFile : null;
}

/**
 * Returns a callback that lets the user pick a new project file within the
 * workspace and saves the open project to it
 *
 * The project is written with the project provider of its file if that can
 * write projects, and as a TissUUmaps project file otherwise. Failures are
 * logged and shown, as the callback is called from event handlers.
 *
 * @returns The callback, or `null` if no workspace is open
 */
export function useSaveProjectAs(): (() => void) | null {
  const workspace = useAppStore((state) => state.workspace);
  const reportSaveError = useReportSaveError();

  const saveAs = useCallback(() => {
    if (workspace === null) {
      return;
    }
    const { name, source, sourceFile } = projectStore.getState();
    const projectProvider =
      sourceFile?.projectProvider.toFile !== undefined
        ? sourceFile.projectProvider
        : getProjectProvider(tissuumapsProjectType);
    const fileExtension =
      Object.values(projectProvider.fileTypes ?? {}).flat()[0] ?? "";
    const stem = source !== null ? SourceUtils.getStem(source) : "";
    pickWorkspaceSaveFile(
      workspace,
      `${stem || name}${fileExtension}`,
      projectProvider.fileTypes !== undefined
        ? [
            {
              description: `${projectProvider.name} project`,
              accept: projectProvider.fileTypes,
            },
          ]
        : [],
    )
      .then(async (picked) => {
        if (picked !== null) {
          await saveProjectAs(
            { handle: picked.handle, projectProvider },
            picked.source,
          );
        }
      })
      .catch(reportSaveError);
  }, [workspace, reportSaveError]);
  return workspace !== null ? saveAs : null;
}

/**
 * Returns a callback that copies a link that opens the open project, and
 * shows it
 *
 * The link is the current address, which names the projects opened and
 * imported from URLs (see `setProjectURLParam`); it is read when the calling
 * component renders, which it does whenever the project changes. Copying needs
 * a secure context, where the clipboard is available. Failures are logged and
 * shown, as the callback is called from event handlers.
 *
 * @returns The callback, or `null` if the address names no project, so that no
 * link can open it
 */
export function useCopyProjectLink(): (() => void) | null {
  const alert = useAlertDialog();

  const copyProjectLink = useCallback(() => {
    const link = window.location.href;
    navigator.clipboard
      .writeText(link)
      .then(() => alert({ title: "Link copied", body: link }))
      .catch(async (error: unknown) => {
        console.error("Failed to copy project link", error);
        await alert({ title: "Cannot copy the link", body: link });
      });
  }, [alert]);
  return readProjectURLParams(window.location.search).length > 0
    ? copyProjectLink
    : null;
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
  const confirmDiscard = useConfirmDiscard();

  const alert = useAlertDialog();
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
          !(await confirmDiscard({
            title: "Open project",
            question: "Open the project anyway?",
            actionLabel: "Open",
          }))
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
  }, [workspace, chooseProjectProvider, confirmDiscard, prompt, alert]);
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
 * unsaved changes. Failures are logged and shown,
 * as the callback is called from event handlers.
 *
 * @returns The callback
 */
export function useOpenProjectFromFile(): () => void {
  const workspace = useAppStore((state) => state.workspace);

  const loadProjectFile = useLoadProjectFile();

  const chooseProjectProvider = useChooseProjectProvider();
  const confirmDiscard = useConfirmDiscard();

  const alert = useAlertDialog();

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
            : undefined;
        if (
          !(await confirmDiscard({
            title:
              outsideWorkspaceNote !== undefined
                ? "Open project from outside the workspace"
                : "Open project",
            note: outsideWorkspaceNote,
            question: "Open it anyway?",
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
  }, [
    workspace,
    loadProjectFile,
    chooseProjectProvider,
    confirmDiscard,
    alert,
  ]);
}

/**
 * Returns a function that loads a project from a local file or directory with
 * a project provider and marks it open
 *
 * A file or directory within the workspace is loaded with its source (see
 * `loadLocalProject`), and a file within it is kept with its project provider,
 * so that the project can be saved back to it. The project URL parameters are
 * cleared, as they name other projects.
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
      const loadedProject = await loadLocalProject(
        projectProvider,
        projectFile,
        workspace,
      );
      openProject(
        loadedProject,
        loadedProject.source !== null &&
          !(projectFile instanceof File) &&
          projectFile.kind === "file"
          ? { handle: projectFile, projectProvider }
          : null,
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
  const confirmDiscard = useConfirmDiscard();

  return useCallback(() => {
    void confirmDiscard({
      title: "Close project",
      question: "Close it anyway?",
      actionLabel: "Close",
    }).then((confirmed) => {
      if (confirmed) {
        clearProject();
        clearProjectURLParams();
        setProjectOpen(false);
      }
    });
  }, [clearProject, confirmDiscard, setProjectOpen]);
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
