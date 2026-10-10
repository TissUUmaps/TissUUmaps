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
 * Helpers work on normalized sources: {@link SourceUtils.isWorkspacePath}
 * classifies them, {@link SourceUtils.makeWorkspacePath} builds them,
 * {@link SourceUtils.getPathSegments}, {@link SourceUtils.getParentSource},
 * {@link SourceUtils.getStem} and {@link SourceUtils.getExtension} take them
 * apart, and
 * {@link SourceUtils.makeProjectPath} turns them back into project-relative
 * paths.
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
    const segments = SourceUtils._getWorkspaceSegments(normalizedSource);
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
   * Turns a workspace-relative path into a path relative to the project file,
   * if the project was loaded from the workspace
   *
   * The result normalizes back to the same workspace-relative path (see
   * {@link SourceUtils.normalizeSource}), so data sources stay valid when the
   * project and its data are moved together. A first segment that contains a
   * colon gets a `./` prefix where it would otherwise be taken for a URL.
   *
   * @param normalizedSource - The normalized source (see
   *   {@link SourceUtils.normalizeSource}); a workspace-relative path has to
   *   name a file or directory within the workspace, not its root
   * @param projectSource - Where the project was loaded from (see
   *   {@link SourceUtils.normalizeSource})
   * @returns The project-relative path (e.g. `points.csv` or
   *   `../shared/x.csv`, or `.` for the project file's own directory) if both
   *   the source and the project source are workspace-relative; the source
   *   unchanged otherwise
   */
  static makeProjectPath(
    normalizedSource: string,
    projectSource: string | null,
  ): string {
    if (
      projectSource === null ||
      !SourceUtils.isWorkspacePath(projectSource) ||
      !SourceUtils.isWorkspacePath(normalizedSource)
    ) {
      return normalizedSource;
    }
    const projectDirSegments = SourceUtils._getWorkspaceSegments(
      projectSource,
    ).slice(0, -1);
    const segments = SourceUtils._getWorkspaceSegments(normalizedSource);
    let commonLength = 0;
    while (
      commonLength < projectDirSegments.length &&
      commonLength < segments.length &&
      projectDirSegments[commonLength] === segments[commonLength]
    ) {
      commonLength++;
    }
    const relativeSegments = [
      ...Array<string>(projectDirSegments.length - commonLength).fill(".."),
      ...segments.slice(commonLength),
    ];
    const projectPath = relativeSegments.join(SourceUtils._pathSep);
    if (projectPath === "") {
      return ".";
    }
    return SourceUtils._urlSchemePattern.test(projectPath)
      ? `.${SourceUtils._pathSep}${projectPath}`
      : projectPath;
  }

  /**
   * Returns the path segments of a normalized source
   *
   * @param normalizedSource - The normalized source (see
   *   {@link SourceUtils.normalizeSource})
   * @returns The segments from the workspace root down to the file or
   *   directory, or the decoded segments of a URL's path, without its query and
   *   hash; empty for URLs without a path, such as `data:` URLs, and for the
   *   root of an origin
   * @throws TypeError if the source is neither a workspace-relative path nor
   *   an absolute URL
   * @throws Error if a workspace-relative path leads above the workspace root
   */
  static getPathSegments(normalizedSource: string): string[] {
    if (SourceUtils.isWorkspacePath(normalizedSource)) {
      return SourceUtils._getWorkspaceSegments(normalizedSource);
    }
    return SourceUtils._getURLSegments(new URL(normalizedSource)).map(
      (segment) => SourceUtils._decodeSegment(segment),
    );
  }

  /**
   * Returns the directory that contains a normalized source
   *
   * @param normalizedSource - The normalized source (see
   *   {@link SourceUtils.normalizeSource})
   * @returns The normalized parent source and the name of the source within
   *   it, decoded for URLs; `null` if the source is the root of its URL's path,
   *   is a URL without a path (such as a `data:` URL), or lies directly in the
   *   workspace, as the workspace root is no source
   * @throws See {@link SourceUtils.getPathSegments}
   */
  static getParentSource(
    normalizedSource: string,
  ): { parentSource: string; name: string } | null {
    if (SourceUtils.isWorkspacePath(normalizedSource)) {
      const segments = SourceUtils._getWorkspaceSegments(normalizedSource);
      const name = segments.pop();
      if (name === undefined || segments.length === 0) {
        return null;
      }
      return { parentSource: SourceUtils.makeWorkspacePath(segments), name };
    }
    const url = new URL(normalizedSource);
    const segments = SourceUtils._getURLSegments(url);
    const name = segments.pop();
    if (name === undefined) {
      return null;
    }
    url.pathname = segments.join(SourceUtils._pathSep);
    return {
      parentSource: url.toString(),
      name: SourceUtils._decodeSegment(name),
    };
  }

  /**
   * Returns the name of the file or directory a normalized source refers to,
   * without its last extension
   *
   * Only the last extension is dropped, so `cells.ome.zarr` gives `cells.ome`.
   * A leading dot does not start an extension, so `.hidden` stays as is.
   *
   * @param normalizedSource - The normalized source (see
   *   {@link SourceUtils.normalizeSource})
   * @returns The stem of the last path segment, decoded for URLs; an empty
   *   string for URLs without a path, such as `data:` URLs or the root of an
   *   origin
   * @throws See {@link SourceUtils.getPathSegments}
   */
  static getStem(normalizedSource: string): string {
    const name = SourceUtils.getPathSegments(normalizedSource).at(-1) ?? "";
    const extensionIndex = SourceUtils._getExtensionIndex(name);
    return extensionIndex !== -1 ? name.substring(0, extensionIndex) : name;
  }

  /**
   * Returns the last extension of the file or directory a normalized source
   * refers to, lower-cased for comparison
   *
   * Only the last extension is returned, so `cells.ome.zarr` gives `.zarr`.
   * A leading dot does not start an extension, so `.hidden` has none.
   *
   * @param normalizedSource - The normalized source (see
   *   {@link SourceUtils.normalizeSource})
   * @returns The extension including its dot (e.g. `.csv`), or an empty
   *   string if the last path segment has none
   * @throws See {@link SourceUtils.getPathSegments}
   */
  static getExtension(normalizedSource: string): string {
    const name = SourceUtils.getPathSegments(normalizedSource).at(-1) ?? "";
    const extensionIndex = SourceUtils._getExtensionIndex(name);
    return extensionIndex !== -1
      ? name.substring(extensionIndex).toLowerCase()
      : "";
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
      const projectDirSegments = SourceUtils._getWorkspaceSegments(
        projectSource,
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
    if (workspace !== null) {
      const segments = SourceUtils._getWorkspaceSegments(workspacePath);
      if (segments.length === 0) {
        throw new Error(`Not a workspace file or directory: ${workspacePath}`);
      }
      return SourceUtils.makeWorkspacePath(segments);
    }
    return SourceUtils._normalizeAppPath(
      SourceUtils._appPathPrefix +
        workspacePath.substring(SourceUtils._workspacePathPrefix.length),
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
   * Splits a workspace-relative path into its collapsed segments, the inverse
   * of {@link SourceUtils.makeWorkspacePath}
   *
   * @param workspacePath - The workspace-relative path (with `/` prefix)
   * @returns The segments from the workspace root down to the file or
   *   directory, empty for the root itself
   * @throws Error if the path leads above the workspace root
   */
  private static _getWorkspaceSegments(workspacePath: string): string[] {
    return SourceUtils._collapseSegments(
      workspacePath.substring(SourceUtils._workspacePathPrefix.length),
    );
  }

  /**
   * Splits the path of a URL into its segments
   *
   * @param url - The URL
   * @returns The non-empty, still encoded segments of the URL's path; empty
   *   for URLs without a hierarchical path, such as `data:` and `blob:` URLs,
   *   whose path does not start with `/`
   */
  private static _getURLSegments(url: URL): string[] {
    if (!url.pathname.startsWith(SourceUtils._pathSep)) {
      return [];
    }
    return url.pathname
      .split(SourceUtils._pathSep)
      .filter((segment) => segment !== "");
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

  /**
   * Returns where the last extension of a file or directory name starts
   *
   * @param name - The name
   * @returns The index of the extension's dot, or `-1` if the name has no
   *   extension; a leading dot does not start one
   */
  private static _getExtensionIndex(name: string): number {
    const extensionIndex = name.lastIndexOf(".");
    return extensionIndex > 0 ? extensionIndex : -1;
  }

  /**
   * Decodes a percent-encoded URL path segment
   *
   * @param segment - The encoded segment
   * @returns The decoded segment, or the segment as is if it holds a malformed
   *   escape (e.g. a stray `%`), which the URL parser lets through
   */
  private static _decodeSegment(segment: string): string {
    try {
      return decodeURIComponent(segment);
    } catch {
      return segment;
    }
  }
}
