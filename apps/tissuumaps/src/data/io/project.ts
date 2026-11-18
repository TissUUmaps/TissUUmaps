import { freeze } from "immer";

import {
  type Project,
  type ProjectProvider,
  ProjectUtils,
  type RawProject,
  SourceUtils,
  createProject,
} from "@tissuumaps/core";

import { appStore } from "@/stores/app";
import { projectStore } from "@/stores/project";

/**
 * Creates a deep copy of a project, keeping only the project's own properties
 *
 * This detaches the copy from the project store, and drops the store's actions
 * as well as any other state that is not part of the project itself, such as
 * where the project was loaded from.
 *
 * @param project - The project to copy
 * @returns The copied project
 */
function cleanProject(project: Project): Project {
  return {
    name: project.name,
    layers: structuredClone(project.layers),
    images: structuredClone(project.images),
    labels: structuredClone(project.labels),
    points: structuredClone(project.points),
    shapes: structuredClone(project.shapes),
    tables: structuredClone(project.tables),
    markerMaps: structuredClone(project.markerMaps),
    sizeMaps: structuredClone(project.sizeMaps),
    colorMaps: structuredClone(project.colorMaps),
    visibilityMaps: structuredClone(project.visibilityMaps),
    opacityMaps: structuredClone(project.opacityMaps),
    osOptions: structuredClone(project.osOptions),
    glOptions: structuredClone(project.glOptions),
    viewerBackgroundColor: structuredClone(project.viewerBackgroundColor),
  };
}

/**
 * Returns the registered project provider of a project type
 *
 * @param type - The project type
 * @returns The project provider
 * @throws Error if no project provider is registered for the type
 */
export function getProjectProvider(type: string): ProjectProvider {
  const projectProvider = appStore.getState().projectProviders.get(type);
  if (projectProvider === undefined) {
    throw new Error(`No project provider for type ${type}.`);
  }
  return projectProvider;
}

/**
 * Opens a project in the project store, replacing the currently open project
 *
 * The opened project is deeply frozen, so that it can only be changed through
 * the project store's actions. The project, where it was loaded from and its
 * fresh instance ID are written in a single update, so that they are never out
 * of sync for the data caches, which resolve project-relative data sources
 * against the project source, and for the viewer, which resets its viewport on
 * a new instance ID.
 *
 * @param project - The project to open
 * @param projectSource - Where the project was loaded from: its absolute URL,
 * the workspace-relative path of the project file (with `/` prefix), or `null`
 * if it was loaded from neither
 */
export function openProject(
  project: Project,
  projectSource: string | null,
): void {
  projectStore.setState(
    freeze(
      {
        ...cleanProject(project),
        source: projectSource,
        instanceId: crypto.randomUUID(),
      },
      true,
    ),
  );
}

/**
 * Loads a project from a URL with a project provider and opens it
 *
 * See {@link loadProjectFromURL} for how the URL and the project source are
 * determined.
 *
 * @param projectProvider - The project provider to load the project with
 * @param projectUrl - The URL to load the project from, absolute or relative
 * to the document base URL
 * @param options - Optional abort signal
 * @throws See {@link loadProjectFromURL}
 */
export async function openProjectFromURL(
  projectProvider: ProjectProvider,
  projectUrl: string,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { project, source } = await loadProjectFromURL(
    projectProvider,
    projectUrl,
    options,
  );
  openProject(createProject(project), source);
}

/**
 * Loads a project from a file with a project provider and opens it
 *
 * The project is opened without a source (see {@link loadProjectFromFile}),
 * so its project-relative data sources fall back to being workspace-relative,
 * and without an open workspace, to being app-relative (see `SourceUtils`).
 *
 * @param projectProvider - The project provider to load the project with
 * @param projectFile - The file to load the project from
 * @param options - Optional abort signal
 * @throws See {@link loadProjectFromFile}
 */
export async function openProjectFromFile(
  projectProvider: ProjectProvider,
  projectFile: File,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { project, source } = await loadProjectFromFile(
    projectProvider,
    projectFile,
    options,
  );
  openProject(createProject(project), source);
}

/**
 * Loads a project from a file handle with a project provider and opens it
 *
 * A file within the open workspace is opened with its workspace-relative path
 * as its source, so that its project-relative data sources are resolved within
 * the file's directory. Otherwise it is opened without a source, like an
 * uploaded file (see {@link openProjectFromFile}).
 *
 * @param projectProvider - The project provider to load the project with
 * @param projectFile - The handle of the file to load the project from
 * @param workspace - The directory handle of the open workspace, if any
 * @param options - Optional abort signal
 * @throws See {@link loadProjectFromFileHandle}
 */
export async function openProjectFromFileHandle(
  projectProvider: ProjectProvider,
  projectFile: FileSystemFileHandle,
  workspace: FileSystemDirectoryHandle | null,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { project, source } = await loadProjectFromFileHandle(
    projectProvider,
    projectFile,
    workspace,
    options,
  );
  openProject(createProject(project), source);
}

/**
 * Merges a project into the open project
 *
 * The data sources of the merged project are normalized (see
 * `SourceUtils.normalizeSource`), so relative sources become absolute URLs or
 * workspace-relative paths. The layers, data objects, tables and maps of the
 * merged project are appended to those of the open project, in a single
 * update. Its name is adopted only if the open project has no data objects;
 * its viewer options and background color are ignored. The open project keeps
 * its instance ID, so that the viewport is not reset. The merged objects are
 * expanded in their panels, and the project is marked open.
 *
 * @param rawProject - The project to merge
 * @param projectSource - Where the project to merge was loaded from: its
 * absolute URL, the workspace-relative path of the project file (with `/`
 * prefix), or `null` if it was loaded from neither
 * @throws Error if an ID of the merged project is already taken in the open
 * project, or if one of its data sources cannot be normalized; the open
 * project is left unchanged in either case
 */
export function mergeProject(
  rawProject: RawProject,
  projectSource: string | null,
): void {
  const project = createProject(rawProject);
  const { workspace } = appStore.getState();
  // resolve against where the merged project was loaded from, as its relative
  // sources would resolve against the open project's source otherwise
  for (const dataObject of [
    ...project.images,
    ...project.labels,
    ...project.points,
    ...project.shapes,
    ...project.tables,
  ]) {
    if (dataObject.dataSource.source !== undefined) {
      dataObject.dataSource.source = SourceUtils.normalizeSource(
        dataObject.dataSource.source,
        workspace,
        projectSource,
      );
    }
  }
  const baseProject = projectStore.getState();
  for (const [baseItems, mergedItems] of [
    [baseProject.layers, project.layers],
    [baseProject.images, project.images],
    [baseProject.labels, project.labels],
    [baseProject.points, project.points],
    [baseProject.shapes, project.shapes],
    [baseProject.tables, project.tables],
    [baseProject.markerMaps, project.markerMaps],
    [baseProject.sizeMaps, project.sizeMaps],
    [baseProject.colorMaps, project.colorMaps],
    [baseProject.visibilityMaps, project.visibilityMaps],
    [baseProject.opacityMaps, project.opacityMaps],
  ] as const) {
    const baseIds = new Set(baseItems.map((item) => item.id));
    const takenItem = mergedItems.find((item) => baseIds.has(item.id));
    if (takenItem !== undefined) {
      throw new Error(`Cannot merge project: ID ${takenItem.id} is taken.`);
    }
  }
  const adoptName = !ProjectUtils.hasData(baseProject);
  projectStore.setState((draft) => {
    if (adoptName) {
      draft.name = project.name;
    }
    draft.layers.push(...project.layers);
    draft.images.push(...project.images);
    draft.labels.push(...project.labels);
    draft.points.push(...project.points);
    draft.shapes.push(...project.shapes);
    draft.tables.push(...project.tables);
    draft.markerMaps.push(...project.markerMaps);
    draft.sizeMaps.push(...project.sizeMaps);
    draft.colorMaps.push(...project.colorMaps);
    draft.visibilityMaps.push(...project.visibilityMaps);
    draft.opacityMaps.push(...project.opacityMaps);
  });
  const appState = appStore.getState();
  appState.setExpandedImageIds([
    ...appState.expandedImageIds,
    ...project.images.map((image) => image.id),
  ]);
  appState.setExpandedLabelsIds([
    ...appState.expandedLabelsIds,
    ...project.labels.map((labels) => labels.id),
  ]);
  appState.setExpandedPointsIds([
    ...appState.expandedPointsIds,
    ...project.points.map((points) => points.id),
  ]);
  appState.setExpandedShapesIds([
    ...appState.expandedShapesIds,
    ...project.shapes.map((shapes) => shapes.id),
  ]);
  appState.setExpandedTableIds([
    ...appState.expandedTableIds,
    ...project.tables.map((table) => table.id),
  ]);
  appState.setProjectOpen(true);
}

/**
 * Loads a project from a URL with a project provider and merges it into the
 * open project
 *
 * See {@link loadProjectFromURL} for how the URL and the project source are
 * determined, and {@link mergeProject} for how the project is merged.
 *
 * @param projectProvider - The project provider to load the project with
 * @param projectUrl - The URL to load the project from, absolute or relative
 * to the document base URL
 * @param options - Optional abort signal
 * @throws See {@link loadProjectFromURL} and {@link mergeProject}
 */
export async function importProjectFromURL(
  projectProvider: ProjectProvider,
  projectUrl: string,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { project, source } = await loadProjectFromURL(
    projectProvider,
    projectUrl,
    options,
  );
  mergeProject(project, source);
}

/**
 * Loads a project from a file with a project provider and merges it into the
 * open project
 *
 * The project is merged without a source (see {@link loadProjectFromFile}),
 * so its project-relative data sources fall back to being workspace-relative,
 * and without an open workspace, to being app-relative (see `SourceUtils`).
 *
 * @param projectProvider - The project provider to load the project with
 * @param projectFile - The file to load the project from
 * @param options - Optional abort signal
 * @throws See {@link loadProjectFromFile} and {@link mergeProject}
 */
export async function importProjectFromFile(
  projectProvider: ProjectProvider,
  projectFile: File,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { project, source } = await loadProjectFromFile(
    projectProvider,
    projectFile,
    options,
  );
  mergeProject(project, source);
}

/**
 * Loads a project from a file handle with a project provider and merges it
 * into the open project
 *
 * A file within the open workspace is merged with its workspace-relative path
 * as its source, so that its project-relative data sources are resolved within
 * the file's directory. Otherwise it is merged without a source, like an
 * uploaded file (see {@link importProjectFromFile}).
 *
 * @param projectProvider - The project provider to load the project with
 * @param projectFile - The handle of the file to load the project from
 * @param workspace - The directory handle of the open workspace, if any
 * @param options - Optional abort signal
 * @throws See {@link loadProjectFromFileHandle} and {@link mergeProject}
 */
export async function importProjectFromFileHandle(
  projectProvider: ProjectProvider,
  projectFile: FileSystemFileHandle,
  workspace: FileSystemDirectoryHandle | null,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { project, source } = await loadProjectFromFileHandle(
    projectProvider,
    projectFile,
    workspace,
    options,
  );
  mergeProject(project, source);
}

/**
 * Determines where a project file is, for resolving its project-relative data
 * sources
 *
 * @param projectFile - The handle of the project file
 * @param workspace - The directory handle of the open workspace, if any
 * @param options - Optional abort signal
 * @returns The workspace-relative path of the file (with `/` prefix) if it
 * lies within the open workspace, or `null` if it does not, or if no workspace
 * is open
 */
export async function resolveProjectSource(
  projectFile: FileSystemFileHandle,
  workspace: FileSystemDirectoryHandle | null,
  options?: { signal?: AbortSignal },
): Promise<string | null> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  if (workspace === null) {
    return null;
  }
  const segments = await workspace.resolve(projectFile);
  signal?.throwIfAborted(); // resolve() does not throw on abort
  return segments !== null ? SourceUtils.makeWorkspacePath(segments) : null;
}

/**
 * Loads a project from a URL with a project provider
 *
 * The URL is resolved like a link, against the document base URL, never
 * against the open workspace or project. The project's source is the one the
 * project provider reports, which is the URL it was redirected to, if any - so
 * that its project-relative data sources are resolved against where the
 * project really is.
 *
 * @param projectProvider - The project provider to load the project with
 * @param projectUrl - The URL to load the project from, absolute or relative
 * to the document base URL
 * @param options - Optional abort signal
 * @returns The loaded project and its source
 * @throws Error if `projectUrl` is not a valid URL, if the project provider
 * cannot load projects, or if the project cannot be loaded
 */
function loadProjectFromURL(
  projectProvider: ProjectProvider,
  projectUrl: string,
  options?: { signal?: AbortSignal },
): Promise<{ project: RawProject; source: string | null }> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const absoluteProjectUrl = new URL(projectUrl, document.baseURI).href;
  return loadWithProjectProvider(projectProvider, absoluteProjectUrl, {
    signal,
  });
}

/**
 * Loads a project from a file with a project provider
 *
 * The project has no source: a `File` cannot be located, the object URL
 * through which it is read is revoked immediately afterwards, and `blob:` URLs
 * cannot serve as a base for project-relative data sources anyway.
 *
 * @param projectProvider - The project provider to load the project with
 * @param projectFile - The file to load the project from
 * @param options - Optional abort signal
 * @returns The loaded project, with a `null` source
 * @throws Error if the project provider cannot load projects, or if the
 * project cannot be loaded
 */
async function loadProjectFromFile(
  projectProvider: ProjectProvider,
  projectFile: File,
  options?: { signal?: AbortSignal },
): Promise<{ project: RawProject; source: string | null }> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const objectUrl = URL.createObjectURL(projectFile);
  try {
    const { project } = await loadWithProjectProvider(
      projectProvider,
      objectUrl,
      { signal },
    );
    return { project, source: null };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

/**
 * Loads a project from a file handle with a project provider
 *
 * A file within the open workspace is loaded through its workspace-relative
 * path, which becomes its source. Otherwise it is loaded like an uploaded file,
 * without a source (see {@link loadProjectFromFile}).
 *
 * @param projectProvider - The project provider to load the project with
 * @param projectFile - The handle of the file to load the project from
 * @param workspace - The directory handle of the open workspace, if any
 * @param options - Optional abort signal
 * @returns The loaded project and its source
 * @throws Error if the project provider cannot load projects, or if the
 * project cannot be loaded
 */
async function loadProjectFromFileHandle(
  projectProvider: ProjectProvider,
  projectFile: FileSystemFileHandle,
  workspace: FileSystemDirectoryHandle | null,
  options?: { signal?: AbortSignal },
): Promise<{ project: RawProject; source: string | null }> {
  const { signal } = options ?? {};
  const projectSource = await resolveProjectSource(
    projectFile,
    workspace,
    options,
  );
  if (projectSource === null) {
    const file = await projectFile.getFile();
    signal?.throwIfAborted(); // getFile() does not throw on abort
    return loadProjectFromFile(projectProvider, file, options);
  }
  return loadWithProjectProvider(projectProvider, projectSource, {
    signal,
    workspace,
  });
}

/**
 * Loads a project with a project provider
 *
 * @param projectProvider - The project provider to load the project with
 * @param normalizedSource - The normalized source to load the project from
 * @param options - Optional abort signal and the directory handle of the open
 * workspace
 * @returns The loaded project, and where it was loaded from (see
 * `ProjectProvider.load`)
 * @throws Error if the project provider cannot load projects, or if the
 * project cannot be loaded
 */
function loadWithProjectProvider(
  projectProvider: ProjectProvider,
  normalizedSource: string,
  options?: {
    signal?: AbortSignal;
    workspace?: FileSystemDirectoryHandle | null;
  },
): Promise<{ project: RawProject; source: string }> {
  if (projectProvider.load === undefined) {
    return Promise.reject(
      new Error(
        `Project provider ${projectProvider.name} cannot load projects.`,
      ),
    );
  }
  return projectProvider.load(normalizedSource, options);
}

/**
 * Writes a project to a file with a project provider and downloads it
 *
 * The project is cleaned first, which detaches it from the project store and
 * drops everything that is not part of the project itself - most notably where
 * it was loaded from, which must never be saved. Every save path goes through
 * here, so that no such state can escape into a saved project.
 *
 * @param projectProvider - The project provider to write the project with
 * @param options - Optional project to download, defaulting to the currently
 * open project, and abort signal
 * @throws Error if the project provider cannot write projects, or if the
 * project cannot be written
 */
export async function downloadProject(
  projectProvider: ProjectProvider,
  options?: { project?: Project; signal?: AbortSignal },
): Promise<void> {
  const { project = projectStore.getState(), signal } = options ?? {};
  signal?.throwIfAborted();
  if (projectProvider.toFile === undefined) {
    throw new Error(
      `Project provider ${projectProvider.name} cannot write projects.`,
    );
  }
  const projectFile = await projectProvider.toFile(cleanProject(project), {
    signal,
  });
  signal?.throwIfAborted();
  const projectUrl = URL.createObjectURL(projectFile);
  const projectLink = document.createElement("a");
  projectLink.download = projectFile.name;
  projectLink.href = projectUrl;
  projectLink.click();
  setTimeout(() => URL.revokeObjectURL(projectUrl), 60_000);
}

/**
 * Returns the registered project providers that can load a source
 *
 * Only project providers that implement `load` and `supports` are considered,
 * in registration order; a `supports` that rejects counts as `false`.
 *
 * @param normalizedSource - The normalized source to load
 * @param workspace - The directory handle of the open workspace, if any
 * @param options - Optional abort signal
 * @returns The project providers that support the source, in registration
 * order
 */
export async function findProjectProviders(
  normalizedSource: string,
  workspace: FileSystemDirectoryHandle | null,
  options?: { signal?: AbortSignal },
): Promise<ProjectProvider[]> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const projectProviders = Array.from(
    appStore.getState().projectProviders.values(),
  );
  const support = await Promise.all(
    projectProviders.map(async (projectProvider) => {
      if (
        projectProvider.load === undefined ||
        projectProvider.supports === undefined
      ) {
        return false;
      }
      try {
        return await projectProvider.supports(normalizedSource, workspace, {
          signal,
        });
      } catch {
        return false; // treated as unsupported
      }
    }),
  );
  signal?.throwIfAborted();
  return projectProviders.filter((_, i) => support[i]);
}

/**
 * Returns the registered project providers that load a local file, by its
 * extension
 *
 * Only project providers that implement `load` are considered, in
 * registration order, matching the extensions of their `fileTypes`
 * regardless of case.
 *
 * @param projectFile - The file, or its handle
 * @returns The project providers that load the file, in registration order
 */
export function findLocalProjectProviders(
  projectFile: File | FileSystemFileHandle,
): ProjectProvider[] {
  const name = projectFile.name.toLowerCase();
  return Array.from(appStore.getState().projectProviders.values()).filter(
    (projectProvider) =>
      projectProvider.load !== undefined &&
      Object.values(projectProvider.fileTypes ?? {})
        .flat()
        .some((fileExtension) => name.endsWith(fileExtension)),
  );
}

/**
 * Returns the extensions of the project files that the registered project
 * providers load, without duplicates
 *
 * @returns The file extensions, in lowercase and with leading dot
 */
export function getProjectFileExtensions(): string[] {
  const fileExtensions = new Set<string>();
  for (const projectProvider of appStore.getState().projectProviders.values()) {
    if (projectProvider.load !== undefined) {
      for (const fileExtension of Object.values(
        projectProvider.fileTypes ?? {},
      ).flat()) {
        fileExtensions.add(fileExtension);
      }
    }
  }
  return Array.from(fileExtensions);
}

/**
 * Returns the types of project files that the registered project providers
 * load, as filters for a file picker
 *
 * All project files come first, as file pickers select the first filter,
 * followed by the project files of each project provider that implements
 * `load` and has `fileTypes`, named after it.
 *
 * @returns The descriptions, and the file extensions by media type
 */
export function getProjectFileTypes(): {
  description: string;
  accept: Record<string, string[]>;
}[] {
  const allAccept: Record<string, string[]> = {};
  const fileTypes: { description: string; accept: Record<string, string[]> }[] =
    [];
  for (const projectProvider of appStore.getState().projectProviders.values()) {
    if (
      projectProvider.load === undefined ||
      projectProvider.fileTypes === undefined
    ) {
      continue;
    }
    for (const [mediaType, fileExtensions] of Object.entries(
      projectProvider.fileTypes,
    )) {
      allAccept[mediaType] = [
        ...new Set([...(allAccept[mediaType] ?? []), ...fileExtensions]),
      ];
    }
    fileTypes.push({
      description: `${projectProvider.name} project`,
      accept: projectProvider.fileTypes,
    });
  }
  return [{ description: "Project", accept: allAccept }, ...fileTypes];
}

/**
 * Returns the projects named by the GET parameters of the registered project
 * providers, in the order of the parameters
 *
 * A parameter names a project if its key is the `urlParam` of a project
 * provider that implements `load`, and its value is not empty.
 *
 * @param search - The query string, e.g. `window.location.search`
 * @returns The project providers and the URLs of the named projects
 */
export function readProjectURLParams(
  search: string,
): { projectProvider: ProjectProvider; projectUrl: string }[] {
  const projectProviders = Array.from(
    appStore.getState().projectProviders.values(),
  );
  return Array.from(new URLSearchParams(search)).flatMap(
    ([key, projectUrl]) => {
      const projectProvider = projectProviders.find(
        (projectProvider) =>
          projectProvider.load !== undefined &&
          projectProvider.urlParam === key,
      );
      return projectProvider !== undefined && projectUrl !== ""
        ? [{ projectProvider, projectUrl }]
        : [];
    },
  );
}

/**
 * Records an opened project in the address bar, so that reloading the page
 * opens the same project again
 *
 * Replaces the GET parameters of every registered project provider with the
 * one of the given project provider. Without a `urlParam`, the project cannot
 * be recorded, and the parameters are only removed.
 *
 * @param projectProvider - The project provider the project was opened with
 * @param projectUrl - The URL the project was opened from
 */
export function setProjectURLParam(
  projectProvider: ProjectProvider,
  projectUrl: string,
): void {
  clearProjectURLParams();
  if (projectProvider.urlParam !== undefined) {
    const url = new URL(window.location.href);
    url.searchParams.append(projectProvider.urlParam, projectUrl);
    window.history.replaceState({}, "", url);
  }
}

/**
 * Removes the GET parameters of every registered project provider from the
 * address bar
 */
export function clearProjectURLParams(): void {
  const url = new URL(window.location.href);
  for (const projectProvider of appStore.getState().projectProviders.values()) {
    if (projectProvider.urlParam !== undefined) {
      url.searchParams.delete(projectProvider.urlParam);
    }
  }
  window.history.replaceState({}, "", url);
}
