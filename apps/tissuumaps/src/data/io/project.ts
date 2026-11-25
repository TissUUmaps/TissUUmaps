import { freeze } from "immer";

import {
  type Project,
  type ProjectProvider,
  type ProjectStoreState,
  ProjectUtils,
  type RawProject,
  SourceUtils,
  createProject,
} from "@tissuumaps/core";

import { appStore } from "@/stores/app";
import { projectStore } from "@/stores/project";

/**
 * Picks the project's own properties, without copying them
 *
 * This drops the store's actions as well as any other state that is not part
 * of the project itself, such as where the project was loaded from.
 *
 * @param project - The project to pick from
 * @returns The project's own properties, shared with `project`
 */
function pickProject(project: Project): Project {
  return {
    name: project.name,
    layers: project.layers,
    images: project.images,
    labels: project.labels,
    points: project.points,
    shapes: project.shapes,
    tables: project.tables,
    markerMaps: project.markerMaps,
    sizeMaps: project.sizeMaps,
    colorMaps: project.colorMaps,
    visibilityMaps: project.visibilityMaps,
    opacityMaps: project.opacityMaps,
    osOptions: project.osOptions,
    glOptions: project.glOptions,
    viewerBackgroundColor: project.viewerBackgroundColor,
    physicalUnitSize: project.physicalUnitSize,
    physicalUnit: project.physicalUnit,
    scaleBarVisibility: project.scaleBarVisibility,
  };
}

/**
 * Creates a deep copy of a project, keeping only the project's own properties
 *
 * This detaches the copy from the project store (see {@link pickProject}).
 *
 * @param project - The project to copy
 * @returns The copied project
 */
function cleanProject(project: Project): Project {
  return structuredClone(pickProject(project));
}

/**
 * Returns whether any part of a project differs by reference from another
 *
 * @param project - The project to compare
 * @param otherProject - The project to compare it with
 * @returns `true` if any part of `project` is not the one of `otherProject`
 */
function hasChangedParts(project: Project, otherProject: Project): boolean {
  const parts = pickProject(project);
  return (Object.keys(parts) as (keyof Project)[]).some(
    (key) => parts[key] !== otherProject[key],
  );
}

/**
 * Returns whether the open project has changed since it was last loaded or
 * saved
 *
 * Every change goes through the project store's actions, which replace the
 * changed parts of the project. A replaced part counts as changed, even if it
 * is equal to the saved one.
 *
 * @param state - The project store state
 * @returns `true` if any part of the project differs from the saved project
 */
export function hasUnsavedChanges(state: ProjectStoreState): boolean {
  return hasChangedParts(state, state.savedProject);
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
 * Opens a loaded project in the project store, replacing the currently open
 * project
 *
 * The opened project is deeply frozen, so that it can only be changed through
 * the project store's actions. The project, where it was loaded from and its
 * fresh instance ID are written in a single update, so that they are never out
 * of sync for the data caches, which resolve project-relative data sources
 * against the project source, and for the viewer, which resets its viewport on
 * a new instance ID. The opened project is also the saved project, so that it
 * has no unsaved changes (see {@link hasUnsavedChanges}).
 *
 * @param loadedProject - The project, and where it was loaded from: its
 * absolute URL, its workspace-relative path (with `/` prefix), or `null` if it
 * was loaded from neither (see {@link loadProjectFromURL} and
 * {@link loadLocalProject})
 * @param sourceFile - The project file within the workspace that the project
 * was loaded from, with the project provider that loaded it, if any (see
 * `ProjectStoreState.sourceFile`)
 */
export function openProject(
  loadedProject: { project: RawProject; source: string | null },
  sourceFile: ProjectStoreState["sourceFile"] = null,
): void {
  const project = cleanProject(createProject(loadedProject.project));
  projectStore.setState(
    freeze(
      {
        ...project,
        source: loadedProject.source,
        sourceFile,
        instanceId: crypto.randomUUID(),
        savedProject: project,
      },
      true,
    ),
  );
}

/**
 * Merges a loaded project into the open project
 *
 * The project-relative data sources of the merged project are resolved
 * against where it was loaded from (see `SourceUtils.rebaseSource`), so they
 * become absolute URLs or workspace-relative paths; other data sources are
 * kept as is, so that their fallbacks apply when they are loaded. The layers, data objects, tables and maps of the
 * merged project are appended to those of the open project, in a single
 * update. Its name is adopted only if the open project has no data objects;
 * its viewer options and background color are ignored. The open project keeps
 * its instance ID, so that the viewport is not reset. The merged objects are
 * expanded in their panels, and the project is marked open.
 *
 * @param loadedProject - The project to merge, and where it was loaded from
 * (see {@link openProject})
 * @throws Error if an ID of the merged project is already taken in the open
 * project, or if one of its data sources cannot be resolved; the open
 * project is left unchanged in either case
 */
export function mergeProject(loadedProject: {
  project: RawProject;
  source: string | null;
}): void {
  const project = createProject(loadedProject.project);
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
      dataObject.dataSource.source = SourceUtils.rebaseSource(
        dataObject.dataSource.source,
        loadedProject.source,
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
export function loadProjectFromURL(
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
 * Loads a project from a local file or directory with a project provider
 *
 * A file or directory within the workspace, or the workspace itself, is
 * loaded through its workspace-relative path, which becomes its source, so
 * that its project-relative data sources are resolved within it. Any other
 * file is loaded through an object URL and has no source: it cannot be
 * located, and `blob:` URLs cannot serve as a base for project-relative data
 * sources, so they fall back to being workspace-relative, and without a
 * workspace, to being app-relative (see `SourceUtils`). A directory outside
 * the workspace cannot be read at all.
 *
 * @param projectProvider - The project provider to load the project with
 * @param projectFile - The file, or the handle of the file or directory
 * @param workspace - The directory handle of the workspace, if any
 * @param options - Optional abort signal
 * @returns The loaded project and its source
 * @throws Error if a directory does not lie within the workspace, if the
 * project provider cannot load projects, or if the project cannot be loaded
 */
export async function loadLocalProject(
  projectProvider: ProjectProvider,
  projectFile: File | FileSystemFileHandle | FileSystemDirectoryHandle,
  workspace: FileSystemDirectoryHandle | null,
  options?: { signal?: AbortSignal },
): Promise<{ project: RawProject; source: string | null }> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  if (projectFile instanceof File) {
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
  const projectSource = await resolveProjectSource(
    projectFile,
    workspace,
    options,
  );
  if (projectSource !== null) {
    return loadWithProjectProvider(projectProvider, projectSource, {
      signal,
      workspace,
    });
  }
  if (projectFile.kind === "directory") {
    throw new Error(
      `The folder "${projectFile.name}" is not in the workspace.`,
    );
  }
  const file = await projectFile.getFile();
  signal?.throwIfAborted(); // getFile() does not throw on abort
  return loadLocalProject(projectProvider, file, workspace, options);
}

/**
 * Determines where a project file or directory is, for resolving its
 * project-relative data sources
 *
 * @param projectFile - The handle of the project file or directory
 * @param workspace - The directory handle of the open workspace, if any
 * @param options - Optional abort signal
 * @returns The workspace-relative path of the file or directory (with `/`
 * prefix, and `/` for the workspace itself) if it lies within the open
 * workspace, or `null` if it does not, or if no workspace is open
 */
export async function resolveProjectSource(
  projectFile: FileSystemFileHandle | FileSystemDirectoryHandle,
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
 * it was loaded from, which must never be saved; saving to the workspace
 * cleans it the same way (see {@link writeProjectFile}). The currently
 * open project is downloaded under the name of where it was opened from, if
 * that has one (e.g. `my_experiment` for `…/my_experiment.tm4`), so that it
 * keeps its file name; otherwise the project provider names the file.
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
  const { project, signal } = options ?? {};
  signal?.throwIfAborted();
  if (projectProvider.toFile === undefined) {
    throw new Error(
      `Project provider ${projectProvider.name} cannot write projects.`,
    );
  }
  const projectState = projectStore.getState();
  const name =
    project === undefined && projectState.source !== null
      ? SourceUtils.getStem(projectState.source) || undefined
      : undefined;
  const projectFile = await projectProvider.toFile(
    cleanProject(project ?? projectState),
    { name, signal },
  );
  signal?.throwIfAborted();
  const projectUrl = URL.createObjectURL(projectFile);
  const projectLink = document.createElement("a");
  projectLink.download = projectFile.name;
  projectLink.href = projectUrl;
  projectLink.click();
  setTimeout(() => URL.revokeObjectURL(projectUrl), 60_000);
}

/**
 * Saves the open project back to its file in the workspace, and marks it as
 * saved (see {@link hasUnsavedChanges})
 *
 * The project is written with the project provider that loaded the file, to
 * the file handle it was loaded from or last saved as (see
 * `ProjectStoreState.sourceFile`), so that opening another workspace in the
 * meantime cannot redirect the save. The browser asks for permission to write
 * the file, as the workspace is opened for reading only. The project is only
 * marked as saved if it is still open and still belongs to that file once
 * written, and changes made while it is being written are not.
 *
 * @param options - Optional abort signal
 * @throws Error if the open project has no file within the workspace, or if
 * its project provider cannot write projects
 * @throws DOMException if the file cannot be written, e.g. because the
 * permission to write it was denied (`NotAllowedError`)
 */
export async function saveProjectToSourceFile(options?: {
  signal?: AbortSignal;
}): Promise<void> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const state = projectStore.getState();
  const { sourceFile, instanceId } = state;
  if (sourceFile === null) {
    throw new Error(
      "The open project was not opened from or saved to the workspace.",
    );
  }
  const project = pickProject(state);
  await writeProjectFile(sourceFile, project, { signal });
  const currentState = projectStore.getState();
  if (
    currentState.instanceId === instanceId &&
    currentState.sourceFile === sourceFile
  ) {
    projectStore.setState({ savedProject: project });
  }
}

/**
 * Saves the open project to a new file within the workspace, and makes that
 * file the one the project was loaded from
 *
 * The sources of the project's data objects are rewritten for the new file
 * (see {@link rebaseProjectSources}), and the open project takes them over, so
 * that saving it again writes to the new file, and the project URLs are
 * removed from the address bar. The project is only switched over if it is
 * still open and the workspace is unchanged once written, and changes made
 * while it is being written are kept, but not marked as saved.
 *
 * @param sourceFile - The new file, with the project provider to write it
 * with
 * @param projectSource - The workspace-relative path of the new file (with `/`
 * prefix)
 * @param options - Optional abort signal
 * @throws Error if a source of the project is invalid, or if the project
 * provider cannot write projects
 * @throws DOMException if the file cannot be written, e.g. because the
 * permission to write it was denied (`NotAllowedError`)
 */
export async function saveProjectAs(
  sourceFile: NonNullable<ProjectStoreState["sourceFile"]>,
  projectSource: string,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const state = projectStore.getState();
  const { instanceId } = state;
  const { workspace } = appStore.getState();
  const project = rebaseProjectSources(
    pickProject(state),
    state.source,
    projectSource,
  );
  await writeProjectFile(sourceFile, project, { signal });
  const currentState = projectStore.getState();
  if (
    currentState.instanceId !== instanceId ||
    appStore.getState().workspace !== workspace
  ) {
    return;
  }
  const currentProject = hasChangedParts(currentState, state)
    ? rebaseProjectSources(
        pickProject(currentState),
        currentState.source,
        projectSource,
      )
    : project;
  projectStore.setState(
    freeze(
      {
        ...currentProject,
        source: projectSource,
        sourceFile,
        savedProject: project,
      },
      true,
    ),
  );
  clearProjectURLParams();
}

/**
 * Rewrites the sources of a project's data objects for a new project file
 * within the workspace
 *
 * Each source is first made independent of the project it was loaded with
 * (see `SourceUtils.rebaseSource`), then relative to the new project file
 * where possible (see `SourceUtils.makeProjectPath`): workspace-relative paths
 * become relative to the new file, URLs and app-relative paths stay as they
 * are, and paths relative to a project loaded from a URL become absolute URLs.
 * Only `source` is rewritten, as no other field of a data source is resolved
 * as a path (see `RawDataSource.source`).
 *
 * @param project - The project whose sources to rewrite
 * @param fromProjectSource - Where the project was loaded from (see
 * `ProjectStoreState.source`)
 * @param toProjectSource - The workspace-relative path of the new project file
 * (with `/` prefix)
 * @returns The project with rewritten sources, sharing the objects whose source
 * is unchanged
 * @throws Error if a source is invalid (see `SourceUtils.rebaseSource`)
 */
function rebaseProjectSources(
  project: Project,
  fromProjectSource: string | null,
  toProjectSource: string,
): Project {
  const rebase = <TObject extends { dataSource: { source?: string } }>(
    object: TObject,
  ): TObject => {
    const { source } = object.dataSource;
    if (source === undefined) {
      return object;
    }
    const rebasedSource = SourceUtils.makeProjectPath(
      SourceUtils.rebaseSource(source, fromProjectSource),
      toProjectSource,
    );
    return rebasedSource !== source
      ? {
          ...object,
          dataSource: { ...object.dataSource, source: rebasedSource },
        }
      : object;
  };
  return {
    ...project,
    images: project.images.map(rebase),
    labels: project.labels.map(rebase),
    points: project.points.map(rebase),
    shapes: project.shapes.map(rebase),
    tables: project.tables.map(rebase),
  };
}

/**
 * Writes a project to a file with a project provider, replacing its contents
 *
 * The project is cleaned first, as for downloading (see
 * {@link downloadProject}). A failed write discards the partly written
 * contents, leaving the file as it was.
 *
 * @param sourceFile - The file, with the project provider to write it with
 * @param project - The project to write
 * @param options - Optional abort signal
 * @throws Error if the project provider cannot write projects
 * @throws DOMException if the file cannot be written, e.g. because the
 * permission to write it was denied (`NotAllowedError`)
 */
async function writeProjectFile(
  sourceFile: NonNullable<ProjectStoreState["sourceFile"]>,
  project: Project,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const { handle, projectProvider } = sourceFile;
  if (projectProvider.toFile === undefined) {
    throw new Error(
      `Project provider ${projectProvider.name} cannot write projects.`,
    );
  }
  const projectFile = await projectProvider.toFile(cleanProject(project), {
    signal,
  });
  signal?.throwIfAborted();
  const writable = await handle.createWritable();
  try {
    signal?.throwIfAborted(); // createWritable() does not throw on abort
    await writable.write(projectFile);
    await writable.close();
  } catch (error) {
    // The write error is the one to report, not a failure to discard the
    // partly written file
    await writable.abort().catch(() => undefined);
    throw error;
  }
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
  appendProjectURLParam(projectProvider, projectUrl);
}

/**
 * Records an imported project in the address bar, after the projects recorded
 * already, so that reloading the page imports it again
 *
 * Does nothing for a project provider without a `urlParam`.
 *
 * @param projectProvider - The project provider the project was imported with
 * @param projectUrl - The URL the project was imported from
 */
export function appendProjectURLParam(
  projectProvider: ProjectProvider,
  projectUrl: string,
): void {
  if (projectProvider.urlParam === undefined) {
    return;
  }
  const url = new URL(window.location.href);
  url.searchParams.append(projectProvider.urlParam, projectUrl);
  window.history.replaceState({}, "", url);
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
