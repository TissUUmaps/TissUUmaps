import { useCallback } from "react";

import type { ProjectProvider } from "@tissuumaps/core";
import { tissuumapsProjectType } from "@tissuumaps/storage";

import { useChooseDialog } from "@/components/dialogs/ChooseDialog/hooks";
import { useConfirmDialog } from "@/components/dialogs/ConfirmDialog/hooks";
import { usePromptDialog } from "@/components/dialogs/PromptDialog/hooks";
import {
  clearProjectURLParams,
  downloadProject,
  findLocalProjectProviders,
  findProjectProviders,
  getProjectFileExtensions,
  getProjectFileTypes,
  getProjectProvider,
  openProjectFromFile,
  openProjectFromFileHandle,
  openProjectFromURL,
  resolveProjectSource,
  setProjectURLParam,
} from "@/data/io/project";
import { pickProjectFile, pickWorkspace } from "@/data/io/workspace";
import { useAppStore } from "@/stores/app";
import { useProjectStore } from "@/stores/project";

import { pickProjectFileFromInput } from "./pickProjectFileFromInput";

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
 * with, asking the user unless exactly one project provider was found
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
      if (foundProjectProviders.length === 1) {
        return foundProjectProviders[0]!;
      }
      const choices = Array.from(projectProviders).filter(
        ([, projectProvider]) =>
          foundProjectProviders.length > 0
            ? foundProjectProviders.includes(projectProvider)
            : projectProvider.load !== undefined,
      );
      if (choices.length === 0) {
        throw new Error("No project provider can load projects.");
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
 * Failures are logged, as the callback is called from event handlers.
 *
 * @returns The callback
 */
export function useOpenProjectFromURL(): () => void {
  const workspace = useAppStore((state) => state.workspace);

  const chooseProjectProvider = useChooseProjectProvider();

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
        await openProjectFromURL(projectProvider, projectUrl);
        setProjectURLParam(projectProvider, projectUrl);
      })
      .catch((error) => {
        console.error("Failed to load project from URL", error);
      });
  }, [workspace, chooseProjectProvider, prompt]);
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
 * directory (see `openProjectFromFileHandle`). Failures are logged, as the
 * callback is called from event handlers.
 *
 * @returns The callback
 */
export function useOpenProjectFromFile(): () => void {
  const workspace = useAppStore((state) => state.workspace);

  const loadProjectFile = useLoadProjectFile();

  const chooseProjectProvider = useChooseProjectProvider();

  const confirm = useConfirmDialog();

  return useCallback(() => {
    const pickedProjectFile =
      workspace === null
        ? pickProjectFileFromInput(getProjectFileExtensions())
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
      .catch((error) => {
        console.error("Failed to load project from file", error);
      });
  }, [workspace, loadProjectFile, chooseProjectProvider, confirm]);
}

/**
 * Returns a function that loads a project from a file with a project provider
 * and marks it open
 *
 * A file handle is loaded with its source if it lies within the workspace
 * (see `openProjectFromFileHandle`), a file without one. The project URL
 * parameters are cleared, as they name other projects.
 *
 * @returns The function, which rejects if the project cannot be read or
 * parsed
 */
export function useLoadProjectFile(): (
  projectFile: FileSystemFileHandle | File,
  projectProvider: ProjectProvider,
) => Promise<void> {
  const workspace = useAppStore((state) => state.workspace);
  const setProjectOpen = useAppStore((state) => state.setProjectOpen);

  return useCallback(
    async (projectFile, projectProvider) => {
      if (projectFile instanceof File) {
        await openProjectFromFile(projectProvider, projectFile);
      } else {
        await openProjectFromFileHandle(
          projectProvider,
          projectFile,
          workspace,
        );
      }
      clearProjectURLParams();
      setProjectOpen(true);
    },
    [workspace, setProjectOpen],
  );
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
 * Returns a callback that downloads the open project as a TissUUmaps project
 * file
 *
 * Failures are logged, as the callback is called from event handlers.
 *
 * @returns The callback
 */
export function useDownloadProject(): () => void {
  return useCallback(() => {
    void downloadProject(getProjectProvider(tissuumapsProjectType)).catch(
      (error) => {
        console.error("Failed to download project", error);
      },
    );
  }, []);
}
