import {
  JSONUtils,
  type Project,
  type ProjectProvider,
  type RawProject,
  SourceUtils,
} from "@tissuumaps/core";

/** The project type of TissUUmaps project files */
export const tissuumapsProjectType = "tissuumaps";

/**
 * Reads and writes TissUUmaps project files (`.tm4`), the native project
 * format of TissUUmaps
 *
 * A project file holds a project as JSON, with non-finite numbers encoded by
 * `JSONUtils`. Its relative data sources are relative to the file.
 */
export class TissUUmapsProjectProvider implements ProjectProvider {
  private static readonly _extension = ".tm4";
  private static readonly _mediaType = "application/json";

  readonly name = "TissUUmaps";
  readonly urlParam = "project";

  supports(normalizedSource: string): Promise<boolean> {
    return Promise.resolve(
      SourceUtils.getExtension(normalizedSource) ===
        TissUUmapsProjectProvider._extension,
    );
  }

  async load(
    normalizedSource: string,
    options?: {
      signal?: AbortSignal;
      workspace?: FileSystemDirectoryHandle | null;
    },
  ): Promise<{ project: RawProject; source: string }> {
    const { signal, workspace = null } = options ?? {};
    signal?.throwIfAborted();
    const { url, file } = await SourceUtils.openSourceFile(
      normalizedSource,
      workspace,
      { signal },
    );
    let rawProjectJSON: string;
    let source: string;
    if (file !== undefined) {
      rawProjectJSON = await file.text();
      signal?.throwIfAborted(); // text() does not throw on abort
      source = normalizedSource;
    } else {
      const response = await fetch(url, { signal });
      if (!response.ok) {
        throw new Error(
          `Failed to load project from ${url}: ${response.status} ${response.statusText}`,
        );
      }
      rawProjectJSON = await response.text(); // throws on abort
      // response.url is empty for responses that are not the result of a request
      source = response.url || url;
    }
    const project = JSONUtils.parse(rawProjectJSON) as RawProject; // TODO validate raw project data
    return { project, source };
  }

  /**
   * Writes a project to a project file
   *
   * The file is named after the project, with whitespace replaced by hyphens
   * and any other non-alphanumeric characters removed, falling back to
   * `Untitled`.
   *
   * @param project - The project to write
   * @returns A promise that resolves to the project file
   */
  toFile(project: Project): Promise<File> {
    const sanitizedProjectName = project.name
      .trim()
      .replace(/\s+/g, "-")
      .replace(/[^a-zA-Z0-9_-]+/g, "")
      .replace(/^[-_]+|[-_]+$/g, "");
    return Promise.resolve(
      new File(
        [JSONUtils.stringify(project)],
        `${sanitizedProjectName || "Untitled"}${TissUUmapsProjectProvider._extension}`,
        { type: TissUUmapsProjectProvider._mediaType },
      ),
    );
  }
}
