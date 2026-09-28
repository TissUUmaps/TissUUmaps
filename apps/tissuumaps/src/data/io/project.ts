import { freeze } from "immer";

import {
  JSONUtils,
  type Project,
  type RawProject,
  SourceUtils,
  createProject,
} from "@tissuumaps/core";

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
    backgroundColor: structuredClone(project.backgroundColor),
  };
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
 */
export function loadProject(
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
 * app-relative (see `SourceUtils`).
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
  loadProject(await readProjectFile(file, options), projectSource);
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
 * Serializes a project to JSON and downloads it as a `.tmap` file
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
  projectLink.download = `${sanitizedProjectName || "Untitled"}.tmap`;
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
