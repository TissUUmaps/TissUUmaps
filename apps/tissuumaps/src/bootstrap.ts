import type { ProjectStoreState } from "@tissuumaps/core";

import { startDataCaches } from "./data/cache";
import { loadProjectFromURL, projectURLParam } from "./data/io/project";
import { enableBuiltInDataProviders } from "./data/providers";
import { notifyTissUUmapsLoaded } from "./events";
import { startPluginRegistry } from "./plugins";
import { appStore } from "./stores/app";
import { projectStore } from "./stores/project";

/** The project loaded on startup when the URL does not name one */
const fallbackProjectUrl = "project.json";

/**
 * Starts up the parts of the application that live outside of React
 *
 * Registers the built-in data providers, starts the data caches and the plugin
 * registry, dismisses the start page once the project has data, starts loading
 * the initial project, and finally announces that the application has loaded.
 * The project is only loading, not loaded, by the time this returns.
 *
 * @returns A callback that cancels the initial project loading, stops watching
 * the project for the start page, and stops the plugin registry and the data
 * caches, invoked on hot module replacement
 */
export function bootstrap(): () => void {
  enableBuiltInDataProviders();
  const stopDataCaches = startDataCaches();
  const stopPluginRegistry = startPluginRegistry();
  const stopStartPageDismissal = startStartPageDismissal();
  const cancelInitialProjectLoading = loadInitialProject();
  notifyTissUUmapsLoaded();
  return () => {
    cancelInitialProjectLoading();
    stopStartPageDismissal();
    stopPluginRegistry();
    stopDataCaches();
  };
}

/**
 * Dismisses the start page the first time the project has a source or data,
 * however it got them
 *
 * @returns A callback that stops watching the project
 */
function startStartPageDismissal(): () => void {
  return projectStore.subscribe((projectState) => {
    if (
      !appStore.getState().startPageDismissed &&
      !isProjectEmpty(projectState)
    ) {
      appStore.getState().setStartPageDismissed(true);
    }
  });
}

/**
 * Tells whether a project has neither a source nor data
 *
 * @param projectState - The state of the project store
 * @returns Whether the project is empty
 */
function isProjectEmpty(projectState: ProjectStoreState): boolean {
  return (
    projectState.source === null &&
    projectState.images.length === 0 &&
    projectState.labels.length === 0 &&
    projectState.points.length === 0 &&
    projectState.shapes.length === 0 &&
    projectState.tables.length === 0
  );
}

/**
 * Starts loading the project named by the {@link projectURLParam} GET
 * parameter, falling back to {@link fallbackProjectUrl} if it is absent or empty
 *
 * Failures are logged, unless loading was cancelled.
 *
 * @returns A callback that cancels the loading
 */
function loadInitialProject(): () => void {
  const abortController = new AbortController();
  const params = new URLSearchParams(window.location.search);
  const projectUrl = params.get(projectURLParam) || fallbackProjectUrl;
  loadProjectFromURL(projectUrl, { signal: abortController.signal }).catch(
    (error) => {
      if (!abortController.signal.aborted) {
        console.error(`Failed to load project from ${projectUrl}:`, error);
      }
    },
  );
  return () => abortController.abort();
}
