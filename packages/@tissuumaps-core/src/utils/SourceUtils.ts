/**
 * Utility methods for resolving the `source` of data sources to URLs or file
 * system handles
 *
 * A source is one of the following, tried in this order:
 *
 * 1. A URL with a scheme (e.g. `https://…`, `blob:…`, `data:…`), which is
 *    normalized and returned as is.
 * 2. An app-relative path, prefixed with `//`, resolved against the base URL
 *    of the application (e.g. `//data/points.csv`).
 * 3. A workspace-relative path, prefixed with `/`, resolved to a file or
 *    directory within the open workspace (e.g. `/data/points.csv`). Without an
 *    open workspace it falls back to being app-relative.
 * 4. A project-relative path, without prefix, resolved against where the
 *    project was loaded from: a URL, or a workspace-relative path (e.g.
 *    `points.csv`, `./points.csv`, `../shared/points.csv`). Without a project
 *    source it falls back to being workspace-relative. Whenever it resolves to
 *    a workspace-relative path - against a workspace-relative project source,
 *    or through that fallback - it falls back further to being app-relative if
 *    no workspace is open.
 *
 * Resolution happens in two steps:
 *
 * 1. {@link SourceUtils.normalizeSource} classifies the source, applies the
 *    fallbacks and normalizes it, synchronously and without touching the file
 *    system. The result is either an absolute URL or a workspace-relative
 *    path. Sources referring to the same data normalize to the same string,
 *    and normalizing a normalized source yields it unchanged.
 * 2. {@link SourceUtils.resolveSource} opens the file or directory that a
 *    normalized workspace-relative path refers to;
 *    {@link SourceUtils.resolveSourceFile} and
 *    {@link SourceUtils.resolveSourceDirectory} accept only one of the two,
 *    and {@link SourceUtils.openSourceFile} also opens the file.
 *    URLs need no such step and are returned as is.
 *
 * Paths use `/` as separator and may contain `.` and `..` segments. Resolving
 * against a URL follows URL semantics: `..` segments beyond the root of the
 * URL's path are dropped silently. Resolving within the workspace collapses the
 * segments and fails if the path leaves the workspace.
 *
 * A relative path whose first segment contains a colon (e.g.
 * `sample1:ch2.tif`) is taken for a URL; write it as `./sample1:ch2.tif`
 * instead.
 */
export class SourceUtils {
  private static readonly _pathSep = "/";
  private static readonly _appPathPrefix = "//";
  private static readonly _workspacePathPrefix = SourceUtils._pathSep;
  private static readonly _urlSchemePattern = /^[a-z][a-z0-9+.-]*:/i;

  /**
   * Returns whether a normalized source refers to a file or directory within
   * the workspace
   *
   * @param normalizedSource - The normalized source (see
   *   {@link SourceUtils.normalizeSource})
   * @returns `true` for workspace-relative paths, `false` for URLs and
   *   app-relative paths
   */
  static isWorkspacePath(normalizedSource: string): boolean {
    return (
      normalizedSource.startsWith(SourceUtils._workspacePathPrefix) &&
      !normalizedSource.startsWith(SourceUtils._appPathPrefix)
    );
  }

  /**
   * Builds a workspace-relative path from the segments of a path within the
   * workspace, as returned by `FileSystemDirectoryHandle.resolve`
   *
   * @param segments - The path segments, from the workspace root down to the
   *   file or directory
   * @returns The workspace-relative path (with `/` prefix)
   */
  static makeWorkspacePath(segments: string[]): string {
    return (
      SourceUtils._workspacePathPrefix + segments.join(SourceUtils._pathSep)
    );
  }

  /**
   * Returns the directory that contains a normalized source
   *
   * @param normalizedSource - The normalized source (see
   *   {@link SourceUtils.normalizeSource})
   * @returns The normalized parent source and the name of the source within
   *   it, decoded for URLs; `null` if the source is the root of its URL's path
   *   or lies directly in the workspace, as the workspace root is no source
   */
  static getParentSource(
    normalizedSource: string,
  ): { parentSource: string; name: string } | null {
    if (SourceUtils.isWorkspacePath(normalizedSource)) {
      const segments = normalizedSource
        .substring(SourceUtils._workspacePathPrefix.length)
        .split(SourceUtils._pathSep);
      const name = segments.pop();
      if (name === undefined || segments.length === 0) {
        return null;
      }
      return { parentSource: SourceUtils.makeWorkspacePath(segments), name };
    }
    const url = new URL(normalizedSource);
    const segments = url.pathname
      .split(SourceUtils._pathSep)
      .filter((segment) => segment !== "");
    const name = segments.pop();
    if (name === undefined) {
      return null;
    }
    url.pathname = segments.join(SourceUtils._pathSep);
    return { parentSource: url.toString(), name: decodeURIComponent(name) };
  }

  /**
   * Normalizes a source, applying the fallbacks for missing project sources
   * and workspaces
   *
   * @param source - The source to normalize
   * @param workspace - The directory handle of the open workspace, if any
   * @param projectSource - Where the project was loaded from: its absolute URL,
   *   the workspace-relative path of the project file (with `/` prefix), or
   *   `null` for projects that were loaded from neither (e.g. uploaded ones)
   * @param options - Optional base URL that app-relative paths are resolved
   *   against (default `document.baseURI`); the application relies on the
   *   default, and the option is what keeps normalization a pure function of
   *   its arguments, testable without a `document`
   * @returns The absolute URL for URLs, app-relative paths, and paths that fall
   *   back to being app-relative or resolve against a project URL; or the
   *   normalized workspace-relative path (with `/` prefix) for paths within the
   *   open workspace
   * @throws Error if the source is empty or not a valid URL, or if it cannot be
   *   normalized as a project-relative, workspace-relative or app-relative path
   *   (see {@link SourceUtils._normalizeProjectPath},
   *   {@link SourceUtils._normalizeWorkspacePath} and
   *   {@link SourceUtils._normalizeAppPath})
   */
  static normalizeSource(
    source: string,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
    options?: { baseUrl?: string },
  ): string {
    if (source === "") {
      throw new Error("Empty source");
    }
    if (SourceUtils._urlSchemePattern.test(source)) {
      try {
        return new URL(source).toString();
      } catch (error) {
        throw new Error(`Invalid URL: ${source}`, { cause: error });
      }
    }
    if (source.startsWith(SourceUtils._appPathPrefix)) {
      return SourceUtils._normalizeAppPath(source, options);
    }
    if (source.startsWith(SourceUtils._workspacePathPrefix)) {
      return SourceUtils._normalizeWorkspacePath(source, workspace, options);
    }
    return SourceUtils._normalizeProjectPath(
      source,
      workspace,
      projectSource,
      options,
    );
  }

  /**
   * Resolves a normalized source to an absolute URL, a file handle or a
   * directory handle
   *
   * URLs are returned as is. A workspace-relative path is opened within the
   * workspace one directory at a time, and its last segment as a file, or as a
   * directory if it names one. No fallbacks apply here: the source has to be
   * normalized with {@link SourceUtils.normalizeSource} first, using the same
   * workspace. Nothing is thrown synchronously: the returned promise rejects
   * instead.
   *
   * @param normalizedSource - The normalized source to resolve: an absolute URL
   *   or a workspace-relative path (with `/` prefix), as returned by
   *   {@link SourceUtils.normalizeSource}
   * @param workspace - The directory handle of the open workspace, if any
   * @param options - Optional abort signal, checked upfront and between file
   *   system calls
   * @returns A promise that resolves to the absolute URL, or to the file or
   *   directory handle for sources within the workspace
   * @throws Error if a workspace-relative path is given without an open
   *   workspace, or if the path leaves the workspace or names its root
   * @throws DOMException if a segment names nothing within the workspace
   *   (`NotFoundError`), if a segment other than the last names a file
   *   (`TypeMismatchError`), or if the operation is aborted (`AbortError`)
   */
  static async resolveSource(
    normalizedSource: string,
    workspace: FileSystemDirectoryHandle | null,
    options?: { signal?: AbortSignal },
  ): Promise<string | FileSystemFileHandle | FileSystemDirectoryHandle> {
    const { signal } = options ?? {};
    signal?.throwIfAborted();
    if (!SourceUtils.isWorkspacePath(normalizedSource)) {
      return normalizedSource;
    }
    if (workspace === null) {
      throw new Error(
        `Cannot resolve workspace-relative path without workspace: ${normalizedSource}`,
      );
    }
    const segments = SourceUtils._collapseSegments(
      normalizedSource.substring(SourceUtils._workspacePathPrefix.length),
    );
    const entryName = segments.pop();
    if (entryName === undefined) {
      throw new Error(`Not a workspace file or directory: ${normalizedSource}`);
    }
    let dir = workspace;
    for (const dirName of segments) {
      dir = await dir.getDirectoryHandle(dirName);
      signal?.throwIfAborted();
    }
    let entry: FileSystemFileHandle | FileSystemDirectoryHandle;
    try {
      entry = await dir.getFileHandle(entryName);
    } catch (error) {
      const isDirectory =
        error instanceof DOMException && error.name === "TypeMismatchError";
      if (!isDirectory) {
        throw error;
      }
      signal?.throwIfAborted();
      entry = await dir.getDirectoryHandle(entryName);
    }
    signal?.throwIfAborted();
    return entry;
  }

  /**
   * Resolves a normalized source to an absolute URL or a file handle
   *
   * Like {@link SourceUtils.resolveSource}, for sources that have to refer to
   * a file.
   *
   * @param normalizedSource - See {@link SourceUtils.resolveSource}
   * @param workspace - See {@link SourceUtils.resolveSource}
   * @param options - See {@link SourceUtils.resolveSource}
   * @returns A promise that resolves to the absolute URL, or to the file handle
   *   for sources within the workspace
   * @throws See {@link SourceUtils.resolveSource}
   * @throws DOMException if the source refers to a directory within the
   *   workspace (`TypeMismatchError`)
   */
  static async resolveSourceFile(
    normalizedSource: string,
    workspace: FileSystemDirectoryHandle | null,
    options?: { signal?: AbortSignal },
  ): Promise<string | FileSystemFileHandle> {
    const resolvedSource = await SourceUtils.resolveSource(
      normalizedSource,
      workspace,
      options,
    );
    if (typeof resolvedSource !== "string" && resolvedSource.kind !== "file") {
      throw new DOMException(
        `Not a workspace file: ${normalizedSource}`,
        "TypeMismatchError",
      );
    }
    return resolvedSource;
  }

  /**
   * Resolves a normalized source to an absolute URL or an opened file
   *
   * Like {@link SourceUtils.resolveSourceFile}, for readers that take either
   * a URL or a `File`: a file in the workspace is opened right away.
   *
   * @param normalizedSource - See {@link SourceUtils.resolveSource}
   * @param workspace - See {@link SourceUtils.resolveSource}
   * @param options - See {@link SourceUtils.resolveSource}; the signal is
   *   also checked once the file is open, as opening it does not throw on
   *   abort
   * @returns A promise that resolves to the absolute URL, or to the opened
   *   file for sources within the workspace
   * @throws See {@link SourceUtils.resolveSourceFile}
   * @throws DOMException if the file cannot be opened, e.g. because it was
   *   removed (`NotFoundError`) or its read permission was revoked
   *   (`NotAllowedError`)
   * @throws The abort reason of the signal, if it is aborted once the file
   *   is open
   */
  static async openSourceFile(
    normalizedSource: string,
    workspace: FileSystemDirectoryHandle | null,
    options?: { signal?: AbortSignal },
  ): Promise<
    { url: string; file?: undefined } | { url?: undefined; file: File }
  > {
    const { signal } = options ?? {};
    const resolvedSource = await SourceUtils.resolveSourceFile(
      normalizedSource,
      workspace,
      options,
    );
    if (typeof resolvedSource === "string") {
      return { url: resolvedSource };
    }
    const file = await resolvedSource.getFile();
    signal?.throwIfAborted(); // getFile() does not throw on abort
    return { file };
  }

  /**
   * Resolves a normalized source to an absolute URL or a directory handle
   *
   * Like {@link SourceUtils.resolveSource}, for sources that have to refer to
   * a directory.
   *
   * @param normalizedSource - See {@link SourceUtils.resolveSource}
   * @param workspace - See {@link SourceUtils.resolveSource}
   * @param options - See {@link SourceUtils.resolveSource}
   * @returns A promise that resolves to the absolute URL, or to the directory
   *   handle for sources within the workspace
   * @throws See {@link SourceUtils.resolveSource}
   * @throws DOMException if the source refers to a file within the workspace
   *   (`TypeMismatchError`)
   */
  static async resolveSourceDirectory(
    normalizedSource: string,
    workspace: FileSystemDirectoryHandle | null,
    options?: { signal?: AbortSignal },
  ): Promise<string | FileSystemDirectoryHandle> {
    const resolvedSource = await SourceUtils.resolveSource(
      normalizedSource,
      workspace,
      options,
    );
    if (
      typeof resolvedSource !== "string" &&
      resolvedSource.kind !== "directory"
    ) {
      throw new DOMException(
        `Not a workspace directory: ${normalizedSource}`,
        "TypeMismatchError",
      );
    }
    return resolvedSource;
  }

  /**
   * Normalizes a project-relative path
   *
   * Against a project URL, the path is resolved to an absolute URL. Against a
   * workspace-relative project path, the path is resolved within the project
   * file's directory to a workspace-relative path. Without a project source,
   * the path is normalized as a workspace-relative path as is. Both of the
   * latter go through {@link SourceUtils._normalizeWorkspacePath}, including
   * its fallback for when no workspace is open.
   *
   * @param projectPath - The project-relative path (without prefix)
   * @param workspace - The directory handle of the open workspace, if any
   * @param projectSource - See {@link SourceUtils.normalizeSource}
   * @param options - Optional base URL for the app-relative fallback (see
   *   {@link SourceUtils._normalizeAppPath})
   * @returns The absolute URL, or the normalized workspace-relative path
   * @throws Error if the path does not form a valid URL with the project URL,
   *   if it leaves the workspace, or if it cannot be normalized as a
   *   workspace-relative path (see {@link SourceUtils._normalizeWorkspacePath})
   */
  private static _normalizeProjectPath(
    projectPath: string,
    workspace: FileSystemDirectoryHandle | null,
    projectSource: string | null,
    options?: { baseUrl?: string },
  ): string {
    if (projectSource !== null) {
      if (SourceUtils._urlSchemePattern.test(projectSource)) {
        try {
          return new URL(projectPath, projectSource).toString();
        } catch (error) {
          throw new Error(`Invalid project-relative path: ${projectPath}`, {
            cause: error,
          });
        }
      }
      const projectDirSegments = SourceUtils._collapseSegments(
        projectSource.substring(SourceUtils._workspacePathPrefix.length),
      ).slice(0, -1);
      const segments = SourceUtils._collapseSegments(
        projectPath,
        projectDirSegments,
      );
      return SourceUtils._normalizeWorkspacePath(
        SourceUtils.makeWorkspacePath(segments),
        workspace,
        options,
      );
    }
    return SourceUtils._normalizeWorkspacePath(
      SourceUtils._workspacePathPrefix + projectPath,
      workspace,
      options,
    );
  }

  /**
   * Normalizes a workspace-relative path
   *
   * With an open workspace, the path's segments are collapsed and the `/`
   * prefix is kept. Without one, the path is normalized as an app-relative
   * path instead (see {@link SourceUtils._normalizeAppPath}), dropping its `/`
   * prefix: `/data/points.csv` then resolves like `//data/points.csv`.
   *
   * @param workspacePath - The workspace-relative path (with `/` prefix)
   * @param workspace - The directory handle of the open workspace, if any
   * @param options - Optional base URL for the app-relative fallback (see
   *   {@link SourceUtils._normalizeAppPath})
   * @returns The normalized workspace-relative path, or the absolute URL if no
   *   workspace is open
   * @throws Error if the path lacks the `/` prefix, or if it leaves the
   *   workspace or names its root (with an open workspace) or cannot be
   *   normalized as an app-relative path (without one, see
   *   {@link SourceUtils._normalizeAppPath})
   */
  private static _normalizeWorkspacePath(
    workspacePath: string,
    workspace: FileSystemDirectoryHandle | null,
    options?: { baseUrl?: string },
  ): string {
    if (!workspacePath.startsWith(SourceUtils._workspacePathPrefix)) {
      throw new Error(`Invalid workspace-relative path: ${workspacePath}`);
    }
    const path = workspacePath.substring(
      SourceUtils._workspacePathPrefix.length,
    );
    if (workspace !== null) {
      const segments = SourceUtils._collapseSegments(path);
      if (segments.length === 0) {
        throw new Error(`Not a workspace file or directory: ${workspacePath}`);
      }
      return SourceUtils.makeWorkspacePath(segments);
    }
    return SourceUtils._normalizeAppPath(
      SourceUtils._appPathPrefix + path,
      options,
    );
  }

  /**
   * Normalizes an app-relative path to an absolute URL
   *
   * The path is resolved against the base URL like a relative URL, so `..`
   * segments can lead above the application and a `/` right after the prefix
   * makes the path absolute within the origin.
   *
   * @param appPath - The app-relative path (with `//` prefix)
   * @param options - Optional base URL to resolve against (default
   *   `document.baseURI`)
   * @returns The absolute URL
   * @throws Error if the path lacks the `//` prefix
   * @throws TypeError if the path does not form a valid URL with the base URL
   */
  private static _normalizeAppPath(
    appPath: string,
    options?: { baseUrl?: string },
  ): string {
    const { baseUrl = document.baseURI } = options ?? {};
    if (!appPath.startsWith(SourceUtils._appPathPrefix)) {
      throw new Error(`Invalid app-relative path: ${appPath}`);
    }
    const path = appPath.substring(SourceUtils._appPathPrefix.length);
    return new URL(path, baseUrl).toString();
  }

  /**
   * Splits a path into segments, dropping empty and `.` segments and
   * collapsing `..` segments against the segments preceding them
   *
   * @param path - The path to split, using `/` as separator
   * @param baseSegments - The segments of the directory the path is relative
   *   to, which `..` segments collapse against first (default: the root)
   * @returns The collapsed segments, empty if the path names the root itself
   * @throws Error if the path leads above the root
   */
  private static _collapseSegments(
    path: string,
    baseSegments: readonly string[] = [],
  ): string[] {
    const segments = [...baseSegments];
    for (const segment of path.split(SourceUtils._pathSep)) {
      if (segment === "" || segment === ".") {
        continue;
      }
      if (segment === "..") {
        if (segments.length === 0) {
          throw new Error(`Path escapes workspace: ${path}`);
        }
        segments.pop();
      } else {
        segments.push(segment);
      }
    }
    return segments;
  }
}
