import { ProjectUtils } from "@tissuumaps/core";

import { startDataCaches } from "./data/cache";
import {
  findProjectProviders,
  loadProjectFromURL,
  mergeProject,
  openProject,
  readProjectURLParams,
} from "./data/io/project";
import {
  enableBuiltInDataProviders,
  enableBuiltInProjectProviders,
} from "./data/providers";
import { notifyTissUUmapsLoaded } from "./events";
import { enableBuiltInPlugins, startPluginRegistry } from "./plugins";
import { appStore } from "./stores/app";
import { projectStore } from "./stores/project";

/** The project opened on startup when the URL does not name one */
const fallbackProjectUrl = "project.tm4";

/**
 * Starts up the parts of the application that live outside of React
 *
 * Registers the built-in data providers and project providers, starts the data
 * caches and the plugin registry, registers the built-in plugins, marks the
 * project open once it has a source or data, collapses the expanded objects
 * whenever another project is loaded or the project is closed, forgets the
 * file the project is saved to whenever the workspace changes, starts loading
 * the initial projects, and finally announces that the application has loaded.
 * The projects are only loading, not loaded, by the time this returns.
 *
 * @returns A callback that cancels loading the initial projects, stops
 * forgetting the file the project is saved to, collapsing the expanded objects
 * and marking the project open, and stops the plugin registry and the data
 * caches, invoked on hot module replacement
 */
export function bootstrap(): () => void {
  enableBuiltInDataProviders();
  enableBuiltInProjectProviders();
  const stopDataCaches = startDataCaches();
  const stopPluginRegistry = startPluginRegistry();
  enableBuiltInPlugins();
  const stopMarkingProjectOpen = startMarkingProjectOpen();
  const stopCollapsingOnProjectSwitch = startCollapsingOnProjectSwitch();
  const stopForgettingSaveTargetOnWorkspaceSwitch =
    startForgettingSaveTargetOnWorkspaceSwitch();
  const initialProjectsLoading = new AbortController();
  void loadInitialProjects({ signal: initialProjectsLoading.signal });
  notifyTissUUmapsLoaded();
  return () => {
    initialProjectsLoading.abort();
    stopForgettingSaveTargetOnWorkspaceSwitch();
    stopCollapsingOnProjectSwitch();
    stopMarkingProjectOpen();
    stopPluginRegistry();
    stopDataCaches();
  };
}

/**
 * Marks the project open as soon as it has a source or data, however it got
 * them
 *
 * @returns A callback that stops watching the project
 */
function startMarkingProjectOpen(): () => void {
  return projectStore.subscribe((projectState) => {
    if (
      !appStore.getState().projectOpen &&
      (projectState.source !== null || ProjectUtils.hasData(projectState))
    ) {
      appStore.getState().setProjectOpen(true);
    }
  });
}

/**
 * Collapses the expanded objects of every panel whenever another project is
 * loaded or the project is closed, i.e. whenever the project's instance ID
 * changes
 *
 * Object IDs are only unique within a project, so the expanded IDs of the
 * previous project would otherwise expand objects of the new one.
 *
 * @returns A callback that stops watching the project
 */
function startCollapsingOnProjectSwitch(): () => void {
  return projectStore.subscribe((projectState, prevProjectState) => {
    if (projectState.instanceId !== prevProjectState.instanceId) {
      appStore.setState({
        expandedImageIds: [],
        expandedLabelsIds: [],
        expandedPointsIds: [],
        expandedShapesIds: [],
        expandedTableIds: [],
      });
    }
  });
}

/**
 * Forgets the file the project is saved to (see `ProjectStoreState.saveTarget`)
 * whenever the workspace changes, i.e. when another workspace is opened or it
 * is closed
 *
 * The file belongs to the previous workspace, so saving the project back to it
 * would write to a workspace that is no longer open.
 *
 * @returns A callback that stops watching the workspace
 */
function startForgettingSaveTargetOnWorkspaceSwitch(): () => void {
  return appStore.subscribe((appState, prevAppState) => {
    if (appState.workspace !== prevAppState.workspace) {
      projectStore.setState({ saveTarget: null });
    }
  });
}

/**
 * Opens the project named by the first GET parameter of a registered project
 * provider, and imports the projects named by the following ones, in order
 *
 * Without such parameters, {@link fallbackProjectUrl} is opened with the first
 * project provider that supports it, without asking. Failures are logged,
 * unless loading was cancelled; the projects that follow a failed one are
 * still loaded, into the empty project if opening failed.
 *
 * @param options - Optional abort signal
 */
async function loadInitialProjects(options?: {
  signal?: AbortSignal;
}): Promise<void> {
  const { signal } = options ?? {};
  const [baseProject, ...projectsToImport] = readProjectURLParams(
    window.location.search,
  );
  if (baseProject === undefined) {
    try {
      const [fallbackProjectProvider] = await findProjectProviders(
        new URL(fallbackProjectUrl, document.baseURI).href,
        null,
        { signal },
      );
      if (fallbackProjectProvider === undefined) {
        throw new Error(`No project provider supports '${fallbackProjectUrl}'`);
      }
      openProject(
        await loadProjectFromURL(fallbackProjectProvider, fallbackProjectUrl, {
          signal,
        }),
      );
    } catch (error) {
      if (!signal?.aborted) {
        console.error(
          `Failed to load project from '${fallbackProjectUrl}'`,
          error,
        );
      }
    }
    return;
  }
  try {
    openProject(
      await loadProjectFromURL(
        baseProject.projectProvider,
        baseProject.projectUrl,
        { signal },
      ),
    );
  } catch (error) {
    if (signal?.aborted) {
      return;
    }
    console.error(
      `Failed to load project from '${baseProject.projectUrl}'`,
      error,
    );
  }
  for (const { projectProvider, projectUrl } of projectsToImport) {
    try {
      mergeProject(
        await loadProjectFromURL(projectProvider, projectUrl, { signal }),
      );
    } catch (error) {
      if (signal?.aborted) {
        return;
      }
      console.error(`Failed to import project from '${projectUrl}'`, error);
    }
  }
}
