import { freeze } from "immer";

import {
  JSONUtils,
  type Project,
  type ProjectProvider,
  type RawProject,
  SourceUtils,
  createProject,
} from "@tissuumaps/core";

import { appStore } from "@/stores/app";
import { projectStore } from "@/stores/project";

/** The GET parameter naming the project to load */
export const projectURLParam = "project";

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
 * The URL is resolved like a link, against the document base URL, never
 * against the open workspace or project. The project is opened with the source
 * the project provider reports, which is the URL it was redirected to, if any -
 * so that its project-relative data sources are resolved against where the
 * project really is.
 *
 * @param projectProvider - The project provider to load the project with
 * @param projectUrl - The URL to load the project from, absolute or relative
 * to the document base URL
 * @param options - Optional abort signal
 * @throws Error if `projectUrl` is not a valid URL, if the project provider
 * cannot load projects, or if the project cannot be loaded
 */
export async function openProjectFromURL(
  projectProvider: ProjectProvider,
  projectUrl: string,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const absoluteProjectUrl = new URL(projectUrl, document.baseURI).href;
  const { project, source } = await loadWithProjectProvider(
    projectProvider,
    absoluteProjectUrl,
    { signal },
  );
  openProject(createProject(project), source);
}

/**
 * Loads a project from a file with a project provider and opens it
 *
 * The project is opened without a source: a `File` cannot be located, the
 * object URL through which it is read is revoked immediately afterwards, and
 * `blob:` URLs cannot serve as a base for project-relative data sources anyway.
 * Its project-relative data sources hence fall back to being
 * workspace-relative, and without an open workspace, to being app-relative
 * (see `SourceUtils`).
 *
 * @param projectProvider - The project provider to load the project with
 * @param projectFile - The file to load the project from
 * @param options - Optional abort signal
 * @throws Error if the project provider cannot load projects, or if the
 * project cannot be loaded
 */
export async function openProjectFromFile(
  projectProvider: ProjectProvider,
  projectFile: File,
  options?: { signal?: AbortSignal },
): Promise<void> {
  const { signal } = options ?? {};
  signal?.throwIfAborted();
  const objectUrl = URL.createObjectURL(projectFile);
  try {
    const { project } = await loadWithProjectProvider(
      projectProvider,
      objectUrl,
      { signal },
    );
    openProject(createProject(project), null);
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
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
 * Loads a project from a file handle with a project provider and opens it
 *
 * A file within the open workspace is loaded through its workspace-relative
 * path, which the project is opened with as its source, so that its
 * project-relative data sources are resolved within the file's directory.
 * Otherwise it is loaded like an uploaded file, without a source (see
 * {@link openProjectFromFile}), and its project-relative data sources fall
 * back to being workspace-relative and then app-relative (see `SourceUtils`).
 *
 * @param projectProvider - The project provider to load the project with
 * @param projectFile - The handle of the file to load the project from
 * @param workspace - The directory handle of the open workspace, if any
 * @param options - Optional abort signal
 * @throws Error if the project provider cannot load projects, or if the
 * project cannot be loaded
 */
export async function openProjectFromFileHandle(
  projectProvider: ProjectProvider,
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
  if (projectSource === null) {
    const file = await projectFile.getFile();
    signal?.throwIfAborted(); // getFile() does not throw on abort
    await openProjectFromFile(projectProvider, file, options);
    return;
  }
  const { project, source } = await loadWithProjectProvider(
    projectProvider,
    projectSource,
    { signal, workspace },
  );
  openProject(createProject(project), source);
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
 * Serializes a project to JSON and downloads it as a `.tm4` file
 *
 * The file is named after the project, with whitespace replaced by hyphens and
 * any other non-alphanumeric characters removed, falling back to `Untitled`.
 *
 * @param project - The project to download, defaulting to the currently open
 * project
 */
export function saveAndDownloadProjectToJSON(project?: Project): void {
  const savedProject = saveProject(project);
  const sanitizedProjectName = savedProject.name
    .trim()
    .replace(/\s+/g, "-")
    .replace(/[^a-zA-Z0-9_-]+/g, "")
    .replace(/^[-_]+|[-_]+$/g, "");
  const projectJSON = JSONUtils.stringify(savedProject);
  const projectBlob = new Blob([projectJSON], { type: "application/json" });
  const projectUrl = URL.createObjectURL(projectBlob);
  const projectLink = document.createElement("a");
  projectLink.download = `${sanitizedProjectName || "Untitled"}.tm4`;
  projectLink.href = projectUrl;
  projectLink.click();
  setTimeout(() => URL.revokeObjectURL(projectUrl), 60_000);
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
