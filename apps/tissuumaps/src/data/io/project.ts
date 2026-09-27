import { freeze } from "immer";

import {
  JSONUtils,
  type Project,
  type ProjectStoreState,
  type RawProject,
  SourceUtils,
  createProject,
} from "@tissuumaps/core";

import { projectStore } from "@/stores/project";

/** The GET parameter naming the project to load */
export const projectURLParam = "project";

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
 * Returns whether the open project has changed since it was last loaded or
 * saved
 *
 * Every change goes through the project store's actions, which replace the
 * changed parts of the project, so comparing the parts by reference suffices.
 *
 * @param state - The project store state
 * @returns `true` if any part of the project differs from the saved project
 */
export function hasUnsavedChanges(state: ProjectStoreState): boolean {
  const project = pickProject(state);
  const savedProject = pickProject(state.savedProject);
  return (Object.keys(project) as (keyof Project)[]).some(
    (key) => project[key] !== savedProject[key],
  );
}

/**
 * Loads a project into the project store, replacing the currently open project
 *
 * The loaded project is deeply frozen, so that it can only be changed through
 * the project store's actions. The project, where it was loaded from and its
 * fresh instance ID are written in a single update, so that they are never out
 * of sync for the data caches, which resolve project-relative data sources
 * against the project source, and for the viewer, which resets its viewport on
 * a new instance ID.
 *
 * @param project - The project to load
 * @param projectSource - Where the project was loaded from: its absolute URL,
 * the workspace-relative path of the project file (with `/` prefix), or `null`
 * if it was loaded from neither
 * @param projectFile - The project file within the workspace that the project
 * was loaded from, for saving it back, if any
 */
export function loadProject(
  project: Project,
  projectSource: string | null,
  projectFile: FileSystemFileHandle | null = null,
): void {
  const cleanedProject = cleanProject(project);
  projectStore.setState(
    freeze(
      {
        ...cleanedProject,
        source: projectSource,
        sourceFile: projectFile,
        instanceId: crypto.randomUUID(),
        savedProject: cleanedProject,
      },
      true,
    ),
  );
}

/**
 * Fetches a project from a URL and loads it into the project store
 *
 * The project is loaded with the URL it was actually fetched from, which is the
 * one it was redirected to, if any - so that its project-relative data sources
 * are resolved against where the project file really is.
 *
 * @param projectUrl - The URL to fetch the project from
 * @param options - Optional abort signal
 * @throws Error if the project cannot be fetched or parsed
 */
export async function loadProjectFromURL(
  projectUrl: string,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { project, resolvedProjectUrl } = await fetchProject(
    projectUrl,
    options,
  );
  loadProject(project, resolvedProjectUrl);
}

/**
 * Reads a project from a file and loads it into the project store
 *
 * The project is loaded without a source: a `File` cannot be located, the
 * object URL through which it is read is revoked immediately afterwards, and
 * `blob:` URLs cannot serve as a base for project-relative data sources anyway.
 * Its project-relative data sources hence fall back to being
 * workspace-relative, and without an open workspace, to being app-relative
 * (see `SourceUtils`).
 *
 * @param projectFile - The file to read the project from
 * @param options - Optional abort signal
 * @throws Error if the project cannot be read or parsed
 */
export async function loadProjectFromFile(
  projectFile: File,
  options?: { signal?: AbortSignal },
): Promise<void> {
  loadProject(await readProjectFile(projectFile, options), null);
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
 * Reads a project from a file handle and loads it into the project store
 *
 * The project is loaded with the workspace-relative path of the file as its
 * source if the file lies within the open workspace, so that its
 * project-relative data sources are resolved within the file's directory.
 * Otherwise it is loaded without a source, like an uploaded file, and its
 * project-relative data sources fall back to being workspace-relative and then
 * app-relative (see `SourceUtils`). Only a file within the workspace is kept
 * for saving the project back to it.
 *
 * @param projectFile - The handle of the file to read the project from
 * @param workspace - The directory handle of the open workspace, if any
 * @param options - Optional abort signal
 * @throws Error if the project cannot be read or parsed
 */
export async function loadProjectFromFileHandle(
  projectFile: FileSystemFileHandle,
  workspace: FileSystemDirectoryHandle | null,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { signal } = options ?? {};
  const projectSource = await resolveProjectSource(
    projectFile,
    workspace,
    options,
  );
  const file = await projectFile.getFile();
  signal?.throwIfAborted(); // getFile() does not throw on abort
  loadProject(
    await readProjectFile(file, options),
    projectSource,
    projectSource !== null ? projectFile : null,
  );
}

/**
 * Reads a project from a file, without loading it into the project store
 *
 * The file is read through an object URL, which is revoked afterwards.
 *
 * @param file - The file to read the project from
 * @param options - Optional abort signal
 * @returns The project read from the file
 * @throws Error if the project cannot be read or parsed
 */
async function readProjectFile(
  file: File,
  options?: { signal?: AbortSignal },
): Promise<Project> {
  const objectUrl = URL.createObjectURL(file);
  try {
    const { project } = await fetchProject(objectUrl, options);
    return project;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

/**
 * Fetches a project from a URL, without loading it into the project store
 *
 * @param projectUrl - The URL to fetch the project from, absolute or relative
 * to the document base URL
 * @param options - Optional abort signal
 * @returns The fetched project, and the absolute URL it was fetched from after
 * following any redirects
 * @throws Error if `projectUrl` is not a valid URL, or if the project cannot be
 * fetched or parsed
 */
async function fetchProject(
  projectUrl: string,
  options?: { signal?: AbortSignal },
): Promise<{ project: Project; resolvedProjectUrl: string }> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const absoluteProjectUrl = new URL(projectUrl, document.baseURI).href;
  const response = await fetch(absoluteProjectUrl, { signal });
  if (!response.ok) {
    throw new Error(
      `Failed to load project from ${projectUrl}: ${response.status} ${response.statusText}`,
    );
  }
  const rawProjectJSON = await response.text(); // throws on abort
  const rawProject = JSONUtils.parse(rawProjectJSON) as RawProject; // TODO validate raw project data
  // response.url is empty for responses that are not the result of a request
  return {
    project: createProject(rawProject),
    resolvedProjectUrl: response.url || absoluteProjectUrl,
  };
}

/**
 * Saves a project
 *
 * The project is cleaned, which detaches it from the project store and drops
 * everything that is not part of the project itself - most notably where it
 * was loaded from, which must never be saved. Every save path goes through
 * here, so that no such state can escape into a saved project.
 *
 * @param project - The project to save, defaulting to the currently open
 * project
 * @returns The project, detached from the project store
 */
export function saveProject(project?: Project): Project {
  return cleanProject(project ?? projectStore.getState());
}

/**
 * Serializes a project to JSON
 *
 * @param project - The project to serialize, defaulting to the currently open
 * project
 * @returns The serialized project
 */
export function saveProjectToJSON(project?: Project): string {
  return JSONUtils.stringify(saveProject(project));
}

/**
 * Makes the name of a project file from the name of the project
 *
 * @param projectName - The name of the project
 * @returns The project name with whitespace replaced by hyphens and any other
 * non-alphanumeric characters removed, falling back to `Untitled`, with the
 * `.tm4` extension
 */
export function makeProjectFileName(projectName: string): string {
  const sanitizedProjectName = projectName
    .trim()
    .replace(/\s+/g, "-")
    .replace(/[^a-zA-Z0-9_-]+/g, "")
    .replace(/^[-_]+|[-_]+$/g, "");
  return `${sanitizedProjectName || "Untitled"}.tm4`;
}

/**
 * Serializes a project to JSON and downloads it as a `.tm4` file
 *
 * The file is named after the project (see {@link makeProjectFileName}).
 *
 * @param project - The project to download, defaulting to the currently open
 * project
 */
export function saveAndDownloadProjectToJSON(project?: Project): void {
  const savedProject = saveProject(project);
  const projectJSON = JSONUtils.stringify(savedProject);
  const projectBlob = new Blob([projectJSON], { type: "application/json" });
  const projectUrl = URL.createObjectURL(projectBlob);
  const projectLink = document.createElement("a");
  projectLink.download = makeProjectFileName(savedProject.name);
  projectLink.href = projectUrl;
  projectLink.click();
  setTimeout(() => URL.revokeObjectURL(projectUrl), 60_000);
}

/**
 * Saves the currently open project back to the workspace file it was loaded
 * from, and marks it as saved (see {@link hasUnsavedChanges})
 *
 * The project is written to the file handle it was loaded from, so connecting
 * another workspace in the meantime cannot redirect the save. The browser asks
 * for permission to write the file, as the workspace is opened for reading
 * only. The project is only marked as saved if it is still open once written,
 * and changes made while it is being written are not.
 *
 * @param options - Optional abort signal
 * @throws Error if the open project was not loaded from a file within the
 * workspace
 * @throws DOMException if the file cannot be written, e.g. because the
 * permission to write it was denied (`NotAllowedError`)
 */
export async function saveProjectToSourceFile(options?: {
  signal?: AbortSignal;
}): Promise<void> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const state = projectStore.getState();
  if (state.sourceFile === null) {
    throw new Error("The open project was not loaded from the workspace");
  }
  const { instanceId } = state;
  const project = pickProject(state);
  await writeProjectFile(state.sourceFile, project, options);
  if (projectStore.getState().instanceId === instanceId) {
    projectStore.setState({ savedProject: project });
  }
}

/**
 * Saves the currently open project to a new file within the workspace, and
 * makes that file the one the project was loaded from
 *
 * The sources of the project's data objects are rewritten for the new file
 * (see {@link rebaseProjectSources}), and the open project takes them over, so
 * that saving it again writes to the new file. The project is only switched
 * over if it is still open once written.
 *
 * @param projectFile - The handle of the new file
 * @param projectSource - The workspace-relative path of the new file (with `/`
 * prefix)
 * @param workspace - The directory handle of the open workspace
 * @param options - Optional abort signal
 * @throws Error if a source of the project is invalid
 * @throws DOMException if the file cannot be written, e.g. because the
 * permission to write it was denied (`NotAllowedError`)
 */
export async function saveProjectAs(
  projectFile: FileSystemFileHandle,
  projectSource: string,
  workspace: FileSystemDirectoryHandle,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const state = projectStore.getState();
  const { instanceId } = state;
  const project = rebaseProjectSources(
    pickProject(state),
    workspace,
    state.source,
    projectSource,
  );
  await writeProjectFile(projectFile, project, options);
  if (projectStore.getState().instanceId === instanceId) {
    projectStore.setState(
      freeze(
        {
          ...project,
          source: projectSource,
          sourceFile: projectFile,
          savedProject: project,
        },
        true,
      ),
    );
  }
}

/**
 * Rewrites the sources of a project's data objects for a new project file
 * within the workspace
 *
 * Sources within the workspace, whether workspace-relative or relative to the
 * current project file, become relative to the new project file. URLs and
 * app-relative paths stay as they are, and paths relative to a project loaded
 * from a URL become absolute URLs, as the new file cannot reach them relatively.
 *
 * @param project - The project whose sources to rewrite
 * @param workspace - The directory handle of the open workspace
 * @param fromProjectSource - Where the project was loaded from (see
 * `ProjectStoreState.source`)
 * @param toProjectSource - The workspace-relative path of the new project file
 * (with `/` prefix)
 * @returns The project with rewritten sources, sharing the unchanged parts
 * @throws Error if a source is invalid (see `SourceUtils.normalizeSource`)
 */
export function rebaseProjectSources(
  project: Project,
  workspace: FileSystemDirectoryHandle | null,
  fromProjectSource: string | null,
  toProjectSource: string,
): Project {
  const directory =
    SourceUtils.getParentSource(toProjectSource)?.parentSource ?? "/";
  const rebaseSource = (source: string | undefined): string | undefined => {
    if (source === undefined || source === "") {
      return source;
    }
    const isAppPath =
      source.startsWith("/") && !SourceUtils.isWorkspacePath(source);
    if (isAppPath) {
      return source;
    }
    const normalizedSource = SourceUtils.normalizeSource(
      source,
      workspace,
      fromProjectSource,
    );
    return SourceUtils.isWorkspacePath(normalizedSource)
      ? SourceUtils.makeRelativePath(normalizedSource, directory)
      : normalizedSource;
  };
  const rebase = <TObject extends { dataSource: { source?: string } }>(
    object: TObject,
  ): TObject => ({
    ...object,
    dataSource: {
      ...object.dataSource,
      source: rebaseSource(object.dataSource.source),
    },
  });
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
 * Writes a project to a file, replacing its contents
 *
 * A failed write discards the partly written contents, leaving the file as it
 * was.
 *
 * @param projectFile - The handle of the file to write
 * @param project - The project to write
 * @param options - Optional abort signal
 * @throws DOMException if the file cannot be written, e.g. because the
 * permission to write it was denied (`NotAllowedError`)
 */
async function writeProjectFile(
  projectFile: FileSystemFileHandle,
  project: Project,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const writable = await projectFile.createWritable();
  try {
    signal?.throwIfAborted(); // createWritable() does not throw on abort
    await writable.write(saveProjectToJSON(project));
    await writable.close();
  } catch (error) {
    // The write error is the one to report, not a failure to discard the
    // partly written file
    await writable.abort().catch(() => undefined);
    throw error;
  }
}

/**
 * Records the project URL in the address bar, so that reloading the page
 * restores the same project
 *
 * @param projectUrl - The URL the project was loaded from
 */
export function setProjectURLParam(projectUrl: string): void {
  const url = new URL(window.location.href);
  url.searchParams.set(projectURLParam, projectUrl);
  window.history.replaceState({}, "", url);
}

/**
 * Removes the project URL from the address bar
 */
export function clearProjectURLParam(): void {
  const url = new URL(window.location.href);
  url.searchParams.delete(projectURLParam);
  window.history.replaceState({}, "", url);
}
